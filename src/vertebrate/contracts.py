"""Small immutable records shared by configuration and preflight consumers."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_IMPLEMENTED = "not_implemented"


class ConfigValidationError(ValueError):
    """A configuration violates the application contract."""


class AssetVerificationError(ValueError):
    """A local asset does not match its recorded manifest."""


class OfflineNetworkError(RuntimeError):
    """An operation tried to use the network during offline preflight."""


@dataclass(frozen=True)
class ModelAssetRecord:
    name: str
    path: Path
    task: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ConfigRecord:
    profile: str
    revision: int
    sha256: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class DoctorResult:
    status: Status
    offline: bool
    checks: tuple[CheckResult, ...]
    network_attempts: tuple[str, ...]
    config: ConfigRecord | None = None


class VertebrateError(ValueError):
    """Base exception for P1 pipeline/manifest contracts."""


class SourceTimingError(VertebrateError):
    def __init__(self, message: str):
        super().__init__(message + '\nConvert locally to CFR, then retry: '
                         'ffmpeg -i "input.mp4" -vf fps=30 -c:v libx264 '
                         '-pix_fmt yuv420p "output_cfr.mp4"')


class CaptureError(VertebrateError):
    """A local frame source failed or a lifecycle operation is invalid."""


class ManifestValidationError(VertebrateError):
    """Invalid dataset metadata, labels, or local file integrity."""


class SourceKind(StrEnum):
    LIVE = 'live'
    VIDEO = 'video'


class TimingMode(StrEnum):
    LIVE_MONOTONIC = 'live_monotonic'
    PTS_VERIFIED = 'pts_verified'
    FRAME_INDEX_FPS_FALLBACK = 'frame_index_fps_fallback'


def _identity(session_id, sequence, source_t_s):
    from .validation import integer, number, text
    text(session_id, 'session_id')
    integer(sequence, 'sequence')
    number(source_t_s, 'source_t_s')


def _array(value, name, shape, kind):
    # Lazy import preserves P0's asset-before-dependency-import ordering.
    import numpy as np
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f'{name} must be an ndarray with shape {shape}')
    if ((kind == 'float' and value.dtype.kind != 'f') or
        (kind == 'bool' and value.dtype != np.bool_) or
        (kind == 'uint8' and value.dtype != np.uint8)):
        raise ValueError(f'{name} must have {kind} dtype')
    if kind == 'float' and not np.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')
    # Immutable bytes backing prevents even setflags(write=True) bypasses, and
    # isolates retained data from the producer's mutable source buffer.
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(shape)


def _probabilities(value, name):
    if ((value < 0) | (value > 1)).any():
        raise ValueError(f'{name} must lie in [0, 1]')


def _boxes(value):
    if (value[..., 2:] < value[..., :2]).any():
        raise ValueError('xyxy boxes must have x2 >= x1 and y2 >= y1')


@dataclass(frozen=True)
class PersonTrackKey:
    session_id: str
    tracker_id: int
    generation: int

    def __post_init__(self):
        from .validation import integer, text
        text(self.session_id, 'session_id')
        integer(self.tracker_id, 'tracker_id')
        integer(self.generation, 'generation', 1)


@dataclass(frozen=True, eq=False)
class FramePacket:
    session_id: str
    sequence: int
    source_kind: SourceKind
    source_name: str
    width: int
    height: int
    frame_bgr: 'np.ndarray'
    source_t_s: float
    acquired_monotonic_ns: int
    timing_mode: TimingMode = TimingMode.FRAME_INDEX_FPS_FALLBACK
    session_start_utc: str | None = None
    session_start_monotonic_ns: int | None = None
    recorded_start_utc: str | None = None
    # Seek sessions start at time zero; these retain original clip label alignment.
    source_frame_index: int | None = None
    source_time_offset_s: float = 0.0

    def __post_init__(self):
        from .validation import integer, number, text, utc_datetime
        _identity(self.session_id, self.sequence, self.source_t_s)
        text(self.source_name, 'source_name')
        integer(self.width, 'width', 1)
        integer(self.height, 'height', 1)
        integer(self.acquired_monotonic_ns, 'acquired_monotonic_ns', 1)
        if not isinstance(self.source_kind, SourceKind) or not isinstance(self.timing_mode, TimingMode):
            raise ValueError('source_kind and timing_mode must be enum values')
        if self.source_kind == SourceKind.VIDEO and self.timing_mode == TimingMode.LIVE_MONOTONIC:
            raise ValueError('video requires a file timing mode')
        for name in ('session_start_utc', 'recorded_start_utc'):
            if getattr(self, name) is not None:
                utc_datetime(getattr(self, name))
        if self.session_start_monotonic_ns is not None:
            integer(self.session_start_monotonic_ns, 'session_start_monotonic_ns', 1)
            if self.acquired_monotonic_ns < self.session_start_monotonic_ns:
                raise ValueError('acquisition predates session start')
        if self.source_frame_index is not None:
            integer(self.source_frame_index, 'source_frame_index')
        number(self.source_time_offset_s, 'source_time_offset_s')
        object.__setattr__(self, 'frame_bgr', _array(self.frame_bgr, 'frame_bgr', (self.height, self.width, 3), 'uint8'))

    def copy_frame(self) -> 'np.ndarray':
        return self.frame_bgr.copy()


@dataclass(frozen=True, eq=False)
class PoseObservation:
    session_id: str
    sequence: int
    source_t_s: float
    width: int
    height: int
    boxes_xyxy: 'np.ndarray'
    box_confidences: 'np.ndarray'
    keypoints_xy: 'np.ndarray'
    keypoint_confidences: 'np.ndarray'
    keypoint_valid_mask: 'np.ndarray'

    def __post_init__(self):
        from .validation import integer
        _identity(self.session_id, self.sequence, self.source_t_s)
        integer(self.width, 'width', 1)
        integer(self.height, 'height', 1)
        if getattr(self.boxes_xyxy, 'ndim', None) != 2:
            raise ValueError('boxes_xyxy must have shape (N, 4)')
        n = self.boxes_xyxy.shape[0]
        for name, shape, kind in (
            ('boxes_xyxy', (n, 4), 'float'), ('box_confidences', (n,), 'float'),
            ('keypoints_xy', (n, 17, 2), 'float'), ('keypoint_confidences', (n, 17), 'float'),
            ('keypoint_valid_mask', (n, 17), 'bool')):
            object.__setattr__(self, name, _array(getattr(self, name), name, shape, kind))
        _boxes(self.boxes_xyxy)
        _probabilities(self.box_confidences, 'box_confidences')
        _probabilities(self.keypoint_confidences, 'keypoint_confidences')


@dataclass(frozen=True, eq=False)
class TrackSample:
    person_key: PersonTrackKey
    sequence: int
    source_t_s: float
    bbox_xyxy: 'np.ndarray'
    box_confidence: float
    keypoints_xy: 'np.ndarray'
    keypoint_confidences: 'np.ndarray'
    keypoint_valid_mask: 'np.ndarray'
    observation_valid: bool
    matched_detection: bool = True
    predicted_only: bool = False

    def __post_init__(self):
        from .validation import number
        if not isinstance(self.person_key, PersonTrackKey):
            raise ValueError('person_key must be PersonTrackKey')
        _identity(self.person_key.session_id, self.sequence, self.source_t_s)
        number(self.box_confidence, 'box_confidence', maximum=1)
        for name in ('observation_valid', 'matched_detection', 'predicted_only'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be bool')
        if self.observation_valid and (not self.matched_detection or self.predicted_only):
            raise ValueError('Absent or predicted-only detection cannot be observation_valid')
        for name, shape, kind in (
            ('bbox_xyxy', (4,), 'float'), ('keypoints_xy', (17, 2), 'float'),
            ('keypoint_confidences', (17,), 'float'), ('keypoint_valid_mask', (17,), 'bool')):
            object.__setattr__(self, name, _array(getattr(self, name), name, shape, kind))
        _boxes(self.bbox_xyxy)
        _probabilities(self.keypoint_confidences, 'keypoint_confidences')


@dataclass(frozen=True)
class SessionResetEvent:
    previous_session_id: str | None
    new_session_id: str
    reason: str
    source_kind: SourceKind
    source_name: str

    def __post_init__(self):
        from .validation import text
        for name in ('new_session_id', 'reason', 'source_name'):
            text(getattr(self, name), name)
        if self.previous_session_id is not None:
            text(self.previous_session_id, 'previous_session_id')
            if self.previous_session_id == self.new_session_id:
                raise ValueError('Reset must create a new session ID')
        if not isinstance(self.source_kind, SourceKind):
            raise ValueError('source_kind must be SourceKind')


@dataclass(frozen=True)
class EndOfStreamEvent:
    session_id: str
    source_name: str
    total_frames_emitted: int
    last_sequence: int | None
    last_source_t_s: float | None

    def __post_init__(self):
        from .validation import integer, text
        text(self.session_id, 'session_id')
        text(self.source_name, 'source_name')
        integer(self.total_frames_emitted, 'total_frames_emitted')
        if self.total_frames_emitted == 0:
            if self.last_sequence is not None or self.last_source_t_s is not None:
                raise ValueError('Empty stream must have no last frame')
        else:
            _identity(self.session_id, self.last_sequence, self.last_source_t_s)
            if self.last_sequence != self.total_frames_emitted - 1:
                raise ValueError('last_sequence must match total_frames_emitted')


# Consumers must close incomplete evidence as inconclusive on EOF/reset; events
# contain no fabricated frame or additional source-time duration.
StreamEvent: TypeAlias = SessionResetEvent | EndOfStreamEvent
QueueItem: TypeAlias = FramePacket | StreamEvent
