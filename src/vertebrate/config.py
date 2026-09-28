"""Immutable configuration, strict JSON parsing, and canonical revisions."""

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .contracts import ConfigValidationError


def _fail(message: str) -> None:
    raise ConfigValidationError(message)


def _number(name: str, value: Any, low: float = 0, high: float | None = None,
            *, inclusive: bool = False, integer: bool = False) -> None:
    if type(value) not in ((int,) if integer else (int, float)):
        _fail(f"{name} must be a {'whole number' if integer else 'finite number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (value < low if inclusive else value <= low) or (high is not None and value > high):
        _fail(f"{name} must be finite and {'>=' if inclusive else '>'} {low}"
              + (f" and <= {high}" if high is not None else ""))


def _text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        _fail(f"{name} must be a nonempty string without NUL characters")


def _path(name: str, value: Any) -> None:
    _text(name, value)
    if '://' in value:
        _fail(f"{name} must be a local filesystem path")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "yolo11n-pose.pt"
    model_path: str = "models/yolo11n-pose.pt"
    task: str = "pose"
    imgsz: int = 512
    device: str = "cpu"
    det_conf_floor: float = 0.10

    def __post_init__(self) -> None:
        _text('model_name', self.model_name)
        _path('model_path', self.model_path)
        if self.task != 'pose' or self.device != 'cpu':
            _fail('P0 requires task=pose and device=cpu')
        _number('imgsz', self.imgsz, integer=True)
        if self.imgsz % 32:
            _fail('imgsz must be a positive multiple of 32')
        _number('det_conf_floor', self.det_conf_floor, high=1)
        if self.det_conf_floor == 1:
            _fail('det_conf_floor must be strictly less than 1')


@dataclass(frozen=True)
class TrackerConfig:
    tracker_type: str = "bytetrack"
    tracker_config_path: str = "config/bytetrack.yaml"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    application_identity_expiry_s: float = 0.75

    def __post_init__(self) -> None:
        if self.tracker_type != 'bytetrack':
            _fail('tracker_type must be bytetrack')
        _path('tracker_config_path', self.tracker_config_path)
        for name in ('track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'match_thresh'):
            _number(name, getattr(self, name), high=1, inclusive=True)
        _number('track_buffer', self.track_buffer, integer=True)
        _number('application_identity_expiry_s', self.application_identity_expiry_s)
        if self.track_low_thresh > self.track_high_thresh:
            _fail('track_low_thresh must not exceed track_high_thresh')


@dataclass(frozen=True)
class CalibrationConfig:
    keypoint_confidence: float = 0.35
    min_initial_person_height_px_at_720p: float = 160.0
    upright_acquisition_s: float = 1.0
    upright_min_theta_deg: float = 60.0
    upright_min_height_ratio_q: float = 0.80
    baseline_max_hip_drift_h0: float = 0.05
    baseline_height_reset_fraction: float = 0.20
    min_torso_length_h0: float = 0.08
    filter_tau_s: float = 0.06
    velocity_window_s: float = 0.25
    velocity_min_observations: int = 3
    velocity_min_span_s: float = 0.12
    rapid_descent_v_h0_per_s: float = 0.60
    rapid_descent_min_duration_s: float = 0.12
    rapid_descent_min_estimates: int = 2
    initial_downward_displacement_d: float = 0.15
    completed_descent_d: float = 0.25
    low_torso_max_theta_deg: float = 35.0
    shape_min_width_height_ratio_r: float = 1.20
    shape_max_height_ratio_q: float = 0.60
    ground_margin_h0: float = 0.25
    down_state_dwell_s: float = 0.20
    descent_deadline_s: float = 1.20
    stillness_max_speed_m_h0_per_s: float = 0.05
    stillness_pair_target_dt_s: float = 0.20
    stillness_pair_min_dt_s: float = 0.15
    stillness_pair_max_dt_s: float = 0.30
    one_second_max_displacement_b_h0: float = 0.04
    one_second_min_dt_s: float = 0.80
    one_second_max_dt_s: float = 1.20
    stillness_duration_s: float = 3.0
    max_observation_gap_s: float = 0.25
    max_down_verification_s: float = 12.0
    recovery_dwell_s: float = 2.0

    def __post_init__(self) -> None:
        fractions = {'keypoint_confidence', 'upright_min_height_ratio_q',
                     'baseline_height_reset_fraction', 'shape_max_height_ratio_q'}
        integers = {'velocity_min_observations', 'rapid_descent_min_estimates'}
        for item in fields(self):
            name = item.name
            _number(name, getattr(self, name), high=90 if name.endswith('_deg') else (1 if name in fractions else None),
                    inclusive=name.endswith('_deg'), integer=name in integers)
        if self.completed_descent_d < self.initial_downward_displacement_d:
            _fail('completed_descent_d must be >= initial_downward_displacement_d')
        if self.stillness_duration_s >= self.max_down_verification_s:
            _fail('stillness_duration_s must be < max_down_verification_s')
        if not self.stillness_pair_min_dt_s < self.stillness_pair_max_dt_s:
            _fail('stillness_pair_min_dt_s must be < stillness_pair_max_dt_s')
        if not self.stillness_pair_min_dt_s <= self.stillness_pair_target_dt_s <= self.stillness_pair_max_dt_s:
            _fail('stillness_pair_target_dt_s must lie between min and max')
        if self.one_second_min_dt_s >= self.one_second_max_dt_s:
            _fail('one_second_min_dt_s must be < one_second_max_dt_s')
        if self.velocity_min_span_s > self.velocity_window_s:
            _fail('velocity_min_span_s must be <= velocity_window_s')


@dataclass(frozen=True)
class RuntimeConfig:
    capture_queue_capacity: int = 2
    incident_queue_capacity: int = 8
    outbox_dir: str = "runtime/outbox"
    logs_dir: str = "runtime/logs"
    manifest_path: str = "models/manifest.json"
    context_staleness_minutes: float = 30.0
    enable_optional_clip: bool = False
    max_media_buffer_bytes: int = 134217728

    def __post_init__(self) -> None:
        for name in ('capture_queue_capacity', 'incident_queue_capacity', 'max_media_buffer_bytes'):
            _number(name, getattr(self, name), integer=True)
        for name in ('outbox_dir', 'logs_dir', 'manifest_path'):
            _path(name, getattr(self, name))
        _number('context_staleness_minutes', self.context_staleness_minutes)
        if type(self.enable_optional_clip) is not bool:
            _fail('enable_optional_clip must be a boolean')


@dataclass(frozen=True)
class AppConfig:
    profile: str = "demo-3s"
    revision: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if self.profile not in ('demo-3s', 'observation-10s'):
            _fail('profile must be demo-3s or observation-10s')
        _number('revision', self.revision, integer=True)
        for name, cls in _SECTIONS.items():
            if type(getattr(self, name)) is not cls:
                _fail(f'{name} must be {cls.__name__}')


_SECTIONS = {'model': ModelConfig, 'tracker': TrackerConfig,
             'calibration': CalibrationConfig, 'runtime': RuntimeConfig}


def _construct(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        _fail(f'{cls.__name__} must be a JSON object')
    unknown = data.keys() - {f.name for f in fields(cls)}
    if unknown:
        _fail(f'{cls.__name__}: unknown keys: {sorted(unknown)}')
    return cls(**data)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_config(path: str | Path) -> AppConfig:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda v: _fail(f'non-finite JSON value: {v}'))
        if not isinstance(data, dict):
            _fail('AppConfig must be a JSON object')
        for name, cls in _SECTIONS.items():
            if name in data:
                data[name] = _construct(cls, data[name])
        return _construct(AppConfig, data)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ConfigValidationError(f'{path}: {exc}') from exc


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_sha256(cfg: AppConfig) -> str:
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, allow_nan=False,
                           separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def increment_revision(cfg: AppConfig, **overrides: Any) -> AppConfig:
    if 'revision' in overrides:
        _fail('revision is incremented automatically; do not override it')
    for name, cls in _SECTIONS.items():
        if name in overrides and isinstance(overrides[name], dict):
            overrides[name] = _construct(cls, asdict(getattr(cfg, name)) | overrides[name])
    try:
        return replace(cfg, revision=cfg.revision + 1, **overrides)
    except TypeError as exc:
        raise ConfigValidationError(str(exc)) from exc
