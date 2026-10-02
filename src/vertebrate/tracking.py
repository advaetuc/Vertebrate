"""ByteTrack association with exact detection rows and source-time generations."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import TrackerConfig
from .contracts import EndOfStreamEvent, PersonTrackKey, PoseObservation, SessionResetEvent, TrackSample
from .pose import guarded_runtime


class _Detections:
    """ByteTrack's sliced results retain the original PoseObservation row."""
    def __init__(self, xywh, conf, cls, indices):
        self.xywh, self.conf, self.cls, self.indices = xywh, conf, cls, indices

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, selection):
        return _Detections(self.xywh[selection], self.conf[selection],
                           self.cls[selection], self.indices[selection])


def _new_tracker(config, fps):
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack

    class AlignedByteTracker(BYTETracker):
        def init_track(self, results, img=None):
            # 8.3.203 normally assigns arange *after* high/low filtering. Its
            # output idx is therefore not an index into the original results.
            boxes = np.column_stack((results.xywh, results.indices))
            return [STrack(box, score, cls) for box, score, cls in
                    zip(boxes, results.conf, results.cls)]

    args = SimpleNamespace(track_high_thresh=config.track_high_thresh,
                           track_low_thresh=config.track_low_thresh,
                           new_track_thresh=config.new_track_thresh,
                           track_buffer=config.track_buffer, match_thresh=config.match_thresh,
                           fuse_score=True)
    return AlignedByteTracker(args, frame_rate=fps)


class ByteTrackManager:
    """One active session; only matched current detections produce evidence.

Use a single vision owner: Ultralytics' numeric ID counter is process-global.
Closed/reset session IDs are fenced to reject delayed frames from old queues.
    """
    def __init__(self, config: TrackerConfig, root_dir: Path | None = None, frame_rate: float = 30):
        if not np.isfinite(frame_rate) or not 0 < frame_rate <= 240:
            raise ValueError('frame_rate must be finite in (0, 240]')
        self.config, self.frame_rate = config, frame_rate
        self.root = (root_dir or Path.cwd()).resolve()
        self.network_attempts: list[str] = []
        self.session_id = None
        self.tracker = None
        self.last_matched_source_t_s: dict[int, float] = {}
        self.generations: dict[int, int] = {}
        self._closed: set[str] = set()
        self._last_sequence = -1
        self._last_source_t_s = -1.0

    def _reset(self, session_id):
        if session_id in self._closed:
            raise ValueError('Cannot reopen a closed tracking session')
        if self.session_id == session_id:
            raise ValueError('Session reset must use a new ID')
        if self.session_id is not None:
            self._closed.add(self.session_id)
        from .cli import _prepare_ultralytics
        _prepare_ultralytics(self.root)
        self.tracker = _new_tracker(self.config, self.frame_rate)
        self.session_id = session_id
        self.last_matched_source_t_s.clear()
        self.generations.clear()
        self._last_sequence, self._last_source_t_s = -1, -1.0

    def handle_event(self, event: SessionResetEvent | EndOfStreamEvent) -> None:
        with guarded_runtime(self.root, self.network_attempts):
            if isinstance(event, SessionResetEvent):
                self._reset(event.new_session_id)
            elif isinstance(event, EndOfStreamEvent):
                if event.session_id != self.session_id:
                    raise ValueError('EOF does not belong to active tracking session')
                self._closed.add(event.session_id)
                self.tracker = None
            else:
                raise TypeError('Expected a lifecycle stream event')

    def update(self, observation: PoseObservation) -> tuple[TrackSample, ...]:
        with guarded_runtime(self.root, self.network_attempts):
            if observation.session_id in self._closed:
                raise ValueError('Observation belongs to a closed tracking session')
            if observation.session_id != self.session_id:
                self._reset(observation.session_id)
            if (observation.sequence <= self._last_sequence
                    or observation.source_t_s < self._last_source_t_s):
                raise ValueError('Tracking requires increasing sequence and nondecreasing source time')
            boxes = observation.boxes_xyxy
            wh = boxes[:, 2:] - boxes[:, :2]
            indices = np.flatnonzero((wh >= 1).all(axis=1))
            xywh = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, wh))
            results = _Detections(xywh[indices], observation.box_confidences[indices],
                                  np.zeros(len(indices)), indices)
            rows = self.tracker.update(results)
            samples = []
            seen_ids, seen_indices = set(), set()
            for row in rows:
                if row.shape != (8,) or not np.isfinite(row).all():
                    raise RuntimeError('Unexpected ByteTrack output layout for pinned version')
                tracker_id, index = int(row[4]), int(row[7])
                if (row[4] != tracker_id or row[7] != index or index not in indices
                        or tracker_id in seen_ids or index in seen_indices):
                    raise RuntimeError('Invalid/duplicate ByteTrack detection correspondence')
                # Do not trust an output row alone: require a current-frame match.
                track = next(t for t in self.tracker.tracked_stracks if t.track_id == tracker_id)
                if track.frame_id != self.tracker.frame_id:
                    continue
                seen_ids.add(tracker_id)
                seen_indices.add(index)
                generation = self.generations.get(tracker_id, 1)
                last = self.last_matched_source_t_s.get(tracker_id)
                if last is not None and observation.source_t_s - last > self.config.application_identity_expiry_s:
                    generation += 1
                self.generations[tracker_id] = generation
                self.last_matched_source_t_s[tracker_id] = observation.source_t_s
                samples.append(TrackSample(
                    PersonTrackKey(observation.session_id, tracker_id, generation),
                    observation.sequence, observation.source_t_s, row[:4],
                    float(observation.box_confidences[index]), observation.keypoints_xy[index],
                    observation.keypoint_confidences[index], observation.keypoint_valid_mask[index], True))
            self._last_sequence = observation.sequence
            self._last_source_t_s = observation.source_t_s
            return tuple(samples)
