"""Local CPU pose adapter. Use on the single vision owner thread.

Ultralytics Results already contain original-image coordinates; scaling those
coordinates a second time would corrupt non-square inputs.
"""
from contextlib import contextmanager
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AppConfig
from .contracts import AssetVerificationError, FramePacket, OfflineNetworkError, PoseObservation
from .offline import offline_guard


@contextmanager
def guarded_runtime(root: Path, attempts: list[str]):
    """Deny downloads, including attempts swallowed by a dependency.

The guard patches process-wide Python APIs: inference/tracking calls must not
overlap in different threads. Capture may run on its separate local reader.
    """
    with offline_guard(root) as blocked:
        os.environ['YOLO_OFFLINE'] = 'true'  # spelling required by 8.3.203
        try:
            yield
        finally:
            os.environ['YOLO_OFFLINE'] = '1'
            attempts.extend(blocked)
        if blocked:
            raise OfflineNetworkError(f'Unexpected offline attempts: {blocked}')


def observation_from_result(packet: FramePacket, result, confidence: float) -> PoseObservation:
    """Adapt a postprocessed Ultralytics Results object without another resize.

Discard invalid or subpixel boxes before ByteTrack's aspect-ratio division.
Non-finite keypoints become zero placeholders with a false validity mask.
    """
    if tuple(result.orig_shape) != (packet.height, packet.width):
        raise ValueError('Pose result dimensions do not match the original frame')
    if not 0 <= confidence <= 1:
        raise ValueError('keypoint confidence must be in [0, 1]')
    if result.boxes is None or result.keypoints is None:
        raise ValueError('Pose result must contain boxes and COCO keypoints')
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32, copy=True)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32, copy=True)
    points = result.keypoints.data.cpu().numpy().astype(np.float32, copy=True)
    n = len(boxes)
    if boxes.shape != (n, 4) or scores.shape != (n,) or points.shape != (n, 17, 3):
        raise ValueError('Malformed pose output: expected boxes (N,4) and keypoints (N,17,3)')
    keep = (np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
            & (scores >= 0) & (scores <= 1))
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, packet.width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, packet.height)
    keep &= (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
    boxes, scores, points = boxes[keep], scores[keep], points[keep]
    xy, conf = points[..., :2], points[..., 2]
    mask = (np.isfinite(points).all(axis=2) & (conf >= confidence) & (conf <= 1)
            & (xy[..., 0] >= 0) & (xy[..., 0] < packet.width)
            & (xy[..., 1] >= 0) & (xy[..., 1] < packet.height))
    xy = np.nan_to_num(xy, nan=0, posinf=0, neginf=0)
    conf = np.clip(np.nan_to_num(conf, nan=0, posinf=0, neginf=0), 0, 1)
    return PoseObservation(packet.session_id, packet.sequence, packet.source_t_s,
                           packet.width, packet.height, boxes, scores, xy, conf, mask)


class PoseAdapter:
    def __init__(self, config: AppConfig, root_dir: Path | None = None, *, headless: bool = False):
        self.config = config
        self.root = (root_dir or Path.cwd()).resolve()
        self.network_attempts: list[str] = []
        self.warmup_ms = 0.0
        if config.model.device != 'cpu':
            raise ValueError('P1B supports CPU float32 inference only')
        with guarded_runtime(self.root, self.network_attempts):
            # Keep this before imports/model construction, also for standalone use.
            from .cli import verify_model_asset, _prepare_ultralytics, _check_dependencies
            self.asset = verify_model_asset(self.root / config.runtime.manifest_path,
                                             config.model.model_name, self.root)
            if self.asset.path != (self.root / config.model.model_path).resolve():
                raise AssetVerificationError('Configured model_path differs from verified asset')
            started = perf_counter()
            _prepare_ultralytics(self.root)
            if headless:
                _check_dependencies(include_qt=False)
            else:
                _check_dependencies()
            from ultralytics import YOLO
            self.model = YOLO(str(self.asset.path), task='pose')
            self.model.to('cpu').model.float()
            self.model_load_ms = (perf_counter() - started) * 1000

    def _predict(self, frame):
        return self.model.predict(frame, imgsz=self.config.model.imgsz,
                                  conf=self.config.model.det_conf_floor, device='cpu',
                                  half=False, batch=1, rect=False, verbose=False,
                                  save=False, show=False)[0]

    def warmup(self) -> float:
        started = perf_counter()
        with guarded_runtime(self.root, self.network_attempts):
            self._predict(np.zeros((self.config.model.imgsz, self.config.model.imgsz, 3), np.uint8))
        self.warmup_ms = (perf_counter() - started) * 1000
        return self.warmup_ms

    def infer(self, packet: FramePacket) -> PoseObservation:
        with guarded_runtime(self.root, self.network_attempts):
            result = self._predict(packet.copy_frame())
            return observation_from_result(packet, result, self.config.calibration.keypoint_confidence)
