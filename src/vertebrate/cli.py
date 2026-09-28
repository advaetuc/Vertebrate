"""Local-only P0 command line and strict offline preflight."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

from .config import AppConfig, config_sha256, load_config
from .contracts import (AssetVerificationError, CheckResult, ConfigRecord,
                        ConfigValidationError, DoctorResult, ModelAssetRecord, Status)
from .offline import offline_guard


def verify_model_asset(manifest_path: Path, model_name: str, root_dir: Path) -> ModelAssetRecord:
    """Verify an existing local file before importing/constructing any model."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or manifest.get('schema_version') != '1.0':
            raise ValueError('expected manifest schema_version 1.0')
        models = manifest.get('models')
        if not isinstance(models, list) or not all(isinstance(x, dict) for x in models):
            raise ValueError('models must be a list of objects')
        matches = [m for m in models if m.get('name') == model_name]
        if len(matches) != 1:
            raise ValueError(f'expected exactly one manifest entry for {model_name}')
        entry = matches[0]
        relative = entry.get('relative_path')
        if not isinstance(relative, str) or not relative or '://' in relative or Path(relative).is_absolute():
            raise ValueError('relative_path must be a local relative path')
        root = root_dir.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'{relative}: asset path escapes project root')
        if not path.is_file():
            raise ValueError(f'{path}: missing local model file; provision it during setup')
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f'{path}: model file is empty')
        expected_size = entry.get('size_bytes')
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f'{path}: size_bytes must be a nonnegative integer')
        if expected_size > 0 and size != expected_size:
            raise ValueError(f'{path}: size mismatch: expected {expected_size}, got {size}')
        expected_hash = entry.get('sha256')
        if not isinstance(expected_hash, str) or re.fullmatch('[0-9a-f]{64}', expected_hash) is None:
            raise ValueError(f'{path}: missing/placeholder/invalid lowercase SHA-256 in manifest')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected_hash:
            raise ValueError(f'{path}: SHA-256 mismatch: expected {expected_hash}, got {digest}')
        if entry.get('task') != 'pose':
            raise ValueError(f'{path}: manifest task must be pose')
        return ModelAssetRecord(model_name, path, entry['task'], digest, size)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise AssetVerificationError(f'Asset verification failed ({manifest_path}, {model_name}): {exc}') from exc


def _local(root: Path, path: str | Path) -> Path:
    return (root / path).resolve()


def _check_runtime(cfg: AppConfig, root: Path) -> str:
    outbox = _local(root, cfg.runtime.outbox_dir)
    logs = _local(root, cfg.runtime.logs_dir)
    try:
        outbox.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        probe = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', prefix='.doctor-', dir=outbox, delete=False) as stream:
                probe = Path(stream.name)
                stream.write(b'VERTEBRATE offline preflight\n')
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if probe is not None:
                probe.unlink()
    except OSError as exc:
        raise RuntimeError(f'Runtime/outbox writability failed: outbox={outbox}, logs={logs}: {exc}') from exc
    return f'outbox write/fsync/remove succeeded: {outbox}; logs: {logs}'


def _check_tracker(cfg: AppConfig, root: Path) -> dict:
    import yaml
    path = _local(root, cfg.tracker.tracker_config_path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    expected = {k: v for k, v in asdict(cfg.tracker).items()
                if k not in ('tracker_config_path', 'application_identity_expiry_s')}
    expected['fuse_score'] = True
    if not isinstance(data, dict) or data.keys() != expected.keys():
        raise ConfigValidationError(f'{path}: tracker YAML keys must be {sorted(expected)}')
    for key, value in expected.items():
        if type(data[key]) is not type(value) or data[key] != value:
            raise ConfigValidationError(f'{path}: {key} must match configuration value {value!r}')
    return data


def _prepare_ultralytics(root: Path) -> None:
    # 8.3.203 accepts the string 'true', not '1', for its import-time DNS check.
    # The requested public flags remain 1 outside the guarded library operations.
    os.environ['YOLO_OFFLINE'] = 'true'
    folder = Path(os.environ['YOLO_CONFIG_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    settings = {
        'settings_version': '0.0.6', 'datasets_dir': str(root / 'data'),
        'weights_dir': str(root / 'models'), 'runs_dir': str(root / 'runtime/logs/runs'),
        'uuid': '0' * 64, 'api_key': '', 'openai_api_key': '',
    }
    settings.update({key: False for key in ('sync', 'clearml', 'comet', 'dvc', 'hub',
                    'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb', 'vscode_msg', 'openvino_msg')})
    # Dedicated application settings; never touch the user's global settings.
    (folder / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')


def _check_dependencies() -> str:
    opencv = sorted(d.metadata['Name'] for d in metadata.distributions()
                    if re.sub(r'[-_.]+', '-', d.metadata.get('Name', '').lower()).startswith('opencv-'))
    if opencv != ['opencv-python']:
        raise RuntimeError(f'Expected only opencv-python; conflicting/missing OpenCV distributions: {opencv}')
    versions = {}
    for module, distribution in (('numpy', 'numpy'), ('cv2', 'opencv-python'),
                                 ('torch', 'torch'), ('torchvision', 'torchvision'),
                                 ('lap', 'lap'), ('PySide6.QtCore', 'PySide6'),
                                 ('PySide6.QtGui', 'PySide6'), ('PySide6.QtWidgets', 'PySide6'),
                                 ('ultralytics', 'ultralytics')):
        importlib.import_module(module)
        versions[distribution] = metadata.version(distribution)
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError('P0 requires a CPU-only PyTorch build')
    import ultralytics.utils as utils
    utils.SETTINGS.update({key: False for key in
                           ('sync', 'hub', 'clearml', 'comet', 'dvc', 'mlflow', 'neptune',
                            'raytune', 'tensorboard', 'wandb', 'vscode_msg') if key in utils.SETTINGS})
    utils.ONLINE = False
    utils.AUTOINSTALL = False
    return json.dumps(versions, sort_keys=True)


def _smoke(cfg: AppConfig, asset: ModelAssetRecord, tracker: dict, root: Path) -> str:
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    model = YOLO(str(asset.path), task='pose')
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    results = model.track(frame, persist=True, tracker=str(_local(root, cfg.tracker.tracker_config_path)),
                          imgsz=cfg.model.imgsz, conf=cfg.model.det_conf_floor,
                          device='cpu', verbose=False, save=False, show=False)
    if len(results) != 1:
        raise RuntimeError('Pose inference must return exactly one result')
    result = results[0]
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or boxes.data.ndim != 2 or boxes.data.shape[1] not in (6, 7):
        raise RuntimeError('Invalid boxes tensor shape')
    if keypoints is None or tuple(keypoints.data.shape) != (len(boxes), 17, 3):
        raise RuntimeError('Invalid pose tensor shape; expected (N, 17, 3)')
    for tensor in (boxes.data, keypoints.data):
        if not np.isfinite(tensor.cpu().numpy()).all():
            raise RuntimeError('Non-finite inference output')
    if not model.predictor.trackers or not isinstance(model.predictor.trackers[0], BYTETracker):
        raise RuntimeError('model.track did not initialize BYTETracker')

    # A blank image normally has zero detections, so additionally exercise real
    # association with controlled detections on two frames (not fake inference).
    byte = BYTETracker(SimpleNamespace(**tracker), frame_rate=30)
    ids = []
    for offset in (0, 2):
        detections = Boxes(np.array([[100 + offset, 80, 180 + offset, 300, .9, 0]], dtype=np.float32),
                           orig_shape=(512, 512))
        assigned = byte.update(detections)
        if assigned.shape != (1, 8) or not np.isfinite(assigned).all():
            raise RuntimeError(f'Invalid ByteTrack assignment shape: {assigned.shape}')
        ids.append(int(assigned[0, 4]))
    if ids[0] <= 0 or ids[0] != ids[1]:
        raise RuntimeError(f'ByteTrack identity not preserved: {ids}')
    return (f'CPU pose boxes={tuple(boxes.data.shape)}, keypoints={tuple(keypoints.data.shape)}; '
            f'ByteTrack two-frame assignment shape=(1, 8), stable ID={ids[0]}')


def run_doctor(config_path: str | Path = 'config/demo.json', manifest_path: str | Path | None = None,
               root_dir: Path | None = None) -> DoctorResult:
    root = (root_dir or Path.cwd()).resolve()
    checks = []
    record = None
    stage = 'configuration'
    with offline_guard(root) as attempts:
        try:
            cfg = load_config(_local(root, config_path))
            record = ConfigRecord(cfg.profile, cfg.revision, config_sha256(cfg))
            checks.append(CheckResult(stage, Status.PASS, record.sha256))
            # Asset failures are resolved before any third-party import or model load.
            stage = 'model_asset'
            asset = verify_model_asset(_local(root, manifest_path or cfg.runtime.manifest_path),
                                       cfg.model.model_name, root)
            if asset.path != _local(root, cfg.model.model_path):
                raise AssetVerificationError('Configured model_path does not match verified manifest asset')
            checks.append(CheckResult(stage, Status.PASS, f'{asset.path}: {asset.sha256} ({asset.size_bytes} bytes)'))
            stage = 'runtime'
            checks.append(CheckResult(stage, Status.PASS, _check_runtime(cfg, root)))
            stage = 'tracker_config'
            tracker = _check_tracker(cfg, root)
            checks.append(CheckResult(stage, Status.PASS, 'ByteTrack YAML matches JSON tracker settings'))
            stage = 'dependencies'
            _prepare_ultralytics(root)
            checks.append(CheckResult(stage, Status.PASS, _check_dependencies()))
            stage = 'inference_and_assignment'
            checks.append(CheckResult(stage, Status.PASS, _smoke(cfg, asset, tracker, root)))
        except Exception as exc:
            checks.append(CheckResult(stage, Status.FAIL, f'{type(exc).__name__}: {exc}'))
        finally:
            os.environ['YOLO_OFFLINE'] = '1'
        if attempts:
            checks.append(CheckResult('offline_guard', Status.FAIL, f'Blocked network/process attempts: {attempts}'))
        else:
            checks.append(CheckResult('offline_guard', Status.PASS, 'zero network/process attempts'))
    status = Status.FAIL if any(c.status == Status.FAIL for c in checks) else Status.PASS
    return DoctorResult(status, True, tuple(checks), tuple(attempts), record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='vertebrate', description='VERTEBRATE local desktop foundation')
    parser.add_argument('--source', help='Reserved for P4 capture source')
    parser.add_argument('--config', default='config/demo.json')
    sub = parser.add_subparsers(dest='command')
    doctor = sub.add_parser('doctor', help='Offline preflight (always denies network)')
    doctor.add_argument('--offline', action='store_true', help='Explicit offline mode; also the default')
    doctor.add_argument('--config', default='config/demo.json')
    doctor.add_argument('--manifest', default=None, help='Override config runtime.manifest_path')
    benchmark = sub.add_parser('benchmark')
    benchmark.add_argument('--manifest', default='data/manifests/dev.json')
    benchmark.add_argument('--device', default='cpu')
    benchmark.add_argument('--output', default='reports/benchmark.json')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--manifest', default='data/manifests/holdout.json')
    evaluate.add_argument('--config', default='config/demo.json')
    evaluate.add_argument('--output', default='reports/evaluation.json')
    evaluate.add_argument('--fail-on-gates', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        # Keep stdout machine-readable even if a dependency logs on first import.
        with redirect_stdout(sys.stderr):
            result = run_doctor(args.config, args.manifest)
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.status == Status.PASS else 1
    if args.command in ('benchmark', 'evaluate'):
        print(json.dumps({'status': Status.NOT_IMPLEMENTED,
                          'message': 'Not implemented until ' + ('P1B' if args.command == 'benchmark' else 'P5')}))
        return 2
    try:
        cfg = load_config(args.config)
    except ConfigValidationError as exc:
        print(json.dumps({'status': Status.FAIL, 'error': str(exc)}))
        return 1
    print(json.dumps({'status': Status.NOT_IMPLEMENTED, 'message': 'GUI launch not implemented until P4',
                      'profile': cfg.profile, 'config_sha256': config_sha256(cfg)}))
    return 2
