"""Immutable source-time dataset annotations and strict local manifest I/O."""
from dataclasses import asdict, dataclass, fields
from functools import wraps
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

from ..contracts import ManifestValidationError
from ..validation import integer, number, text, utc_datetime


def _validated(fn):
    @wraps(fn)
    def check(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, TypeError, OSError, OverflowError) as exc:
            if isinstance(exc, ManifestValidationError):
                raise
            raise ManifestValidationError(str(exc)) from exc
    return check


@dataclass(frozen=True)
class TimeInterval:
    start_s: float
    end_s: float

    @_validated
    def __post_init__(self):
        number(self.start_s, 'start_s')
        number(self.end_s, 'end_s')
        if self.end_s < self.start_s:
            raise ValueError('Interval requires start_s <= end_s')


@dataclass(frozen=True)
class FrameTimingSpec:
    fps: float
    frame_count: int
    duration_s: float
    cfr_verified: bool

    @_validated
    def __post_init__(self):
        number(self.fps, 'fps', maximum=240)
        integer(self.frame_count, 'frame_count', 1)
        number(self.duration_s, 'duration_s')
        if self.fps == 0 or self.duration_s == 0:
            raise ValueError('fps and duration_s must be > 0')
        if type(self.cfr_verified) is not bool:
            raise ValueError('cfr_verified must be bool')
        if abs(self.duration_s - self.frame_count / self.fps) > 1 / self.fps + 1e-9:
            raise ValueError('duration_s inconsistent with frame_count/fps')


@dataclass(frozen=True)
class PersonAnnotation:
    person_label: str
    fall_expected: bool
    already_down_from_start: bool = False
    fall_onset_interval_s: TimeInterval | None = None
    down_posture_interval_s: TimeInterval | None = None
    stillness_eligible_interval_s: TimeInterval | None = None
    recovery_interval_s: TimeInterval | None = None
    visibility_gaps_s: tuple[TimeInterval, ...] = ()
    initial_bbox_xyxy: tuple[float, float, float, float] | None = None

    @_validated
    def __post_init__(self):
        text(self.person_label, 'person_label')
        if type(self.fall_expected) is not bool or type(self.already_down_from_start) is not bool:
            raise ValueError('fall_expected and already_down_from_start must be bool')
        for name in _INTERVAL_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, TimeInterval):
                raise ValueError(f'{name} must be TimeInterval or None')
        object.__setattr__(self, 'visibility_gaps_s', tuple(self.visibility_gaps_s))
        last_end = -1
        for gap in self.visibility_gaps_s:
            if not isinstance(gap, TimeInterval) or gap.start_s < last_end:
                raise ValueError('visibility_gaps_s must be ordered nonoverlapping TimeIntervals')
            last_end = gap.end_s
        if self.initial_bbox_xyxy is not None:
            box = tuple(self.initial_bbox_xyxy)
            if len(box) != 4:
                raise ValueError('initial_bbox_xyxy must contain four coordinates')
            for value in box:
                number(value, 'initial_bbox_xyxy coordinate')
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError('initial_bbox_xyxy must be ordered xyxy')
            object.__setattr__(self, 'initial_bbox_xyxy', box)


_INTERVAL_FIELDS = ('fall_onset_interval_s', 'down_posture_interval_s',
                    'stillness_eligible_interval_s', 'recovery_interval_s')


@dataclass(frozen=True)
class ClipManifestRow:
    clip_id: str
    relative_path: str
    sha256: str
    subject_id: str
    session_group: str
    source_group: str
    camera_view: str
    eligibility: str
    exclusion_reason: str | None
    people_count: int
    frame_timing: FrameTimingSpec
    recorded_start_utc: str | None
    person_annotations: tuple[PersonAnnotation, ...]

    @_validated
    def __post_init__(self):
        for name in ('clip_id', 'relative_path', 'subject_id', 'session_group', 'source_group', 'camera_view'):
            text(getattr(self, name), name)
        path = Path(self.relative_path)
        windows = PureWindowsPath(self.relative_path)
        if path.is_absolute() or windows.drive or windows.root or '..' in windows.parts or '..' in path.parts or '://' in self.relative_path:
            raise ValueError('relative_path must stay inside the local dataset root')
        if not isinstance(self.sha256, str) or re.fullmatch('[0-9a-f]{64}', self.sha256) is None:
            raise ValueError('sha256 must be 64 lowercase hex characters')
        if self.eligibility not in ('supported', 'excluded_out_of_envelope', 'ambiguity_stress_test'):
            raise ValueError('Invalid eligibility')
        if self.exclusion_reason is not None:
            text(self.exclusion_reason, 'exclusion_reason')
        if self.eligibility == 'excluded_out_of_envelope' and self.exclusion_reason is None:
            raise ValueError('Excluded clips require exclusion_reason')
        integer(self.people_count, 'people_count')
        if not isinstance(self.frame_timing, FrameTimingSpec):
            raise ValueError('frame_timing must be FrameTimingSpec')
        if self.eligibility == 'supported' and not self.frame_timing.cfr_verified:
            raise ValueError('Supported clips require verified CFR timing')
        if self.recorded_start_utc is not None:
            utc_datetime(self.recorded_start_utc)
        object.__setattr__(self, 'person_annotations', tuple(self.person_annotations))
        if len(self.person_annotations) != self.people_count:
            raise ValueError('people_count must match person_annotations count')
        labels = set()
        limit = self.frame_timing.duration_s + 1 / self.frame_timing.fps
        for person in self.person_annotations:
            if not isinstance(person, PersonAnnotation):
                raise ValueError('person_annotations must contain PersonAnnotation')
            if person.person_label in labels:
                raise ValueError('Duplicate person_label')
            labels.add(person.person_label)
            intervals = [getattr(person, f) for f in _INTERVAL_FIELDS] + list(person.visibility_gaps_s)
            if any(i is not None and i.end_s > limit + 1e-9 for i in intervals):
                raise ValueError('Annotation interval exceeds clip duration plus one frame')
            if person.fall_expected and self.eligibility == 'supported':
                onset, down, still = (getattr(person, f) for f in _INTERVAL_FIELDS[:3])
                if any(i is None for i in (onset, down, still)):
                    raise ValueError('Supported expected fall requires onset, down-posture and stillness intervals')
                if not onset.start_s <= down.start_s <= still.start_s:
                    raise ValueError('Fall onset, down-posture and stillness intervals must be ordered')


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    partition: str
    clips: tuple[ClipManifestRow, ...]

    @_validated
    def __post_init__(self):
        if self.schema_version != '1.0' or self.partition not in ('dev', 'holdout'):
            raise ValueError('Expected schema_version=1.0 and partition dev or holdout')
        object.__setattr__(self, 'clips', tuple(self.clips))
        if not all(isinstance(c, ClipManifestRow) for c in self.clips):
            raise ValueError('clips must contain ClipManifestRow')
        if len({c.clip_id for c in self.clips}) != len(self.clips):
            raise ValueError('Duplicate clip_id')


def _construct(cls, data):
    if not isinstance(data, dict):
        raise ManifestValidationError(f'{cls.__name__} must be a JSON object')
    unknown = data.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ManifestValidationError(f'{cls.__name__}: unknown keys {sorted(unknown)}')
    try:
        return cls(**data)
    except TypeError as exc:
        raise ManifestValidationError(f'{cls.__name__}: missing/invalid required fields: {exc}') from exc


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ManifestValidationError(f'Duplicate JSON key {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ManifestValidationError(f'Non-finite JSON value {value}')


@_validated
def load_dataset_manifest(path: Path, verify_files: bool = False, root_dir: Path | None = None) -> DatasetManifest:
    """Paths resolve against explicit root_dir or current project working directory."""
    data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                      parse_constant=_reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get('clips'), list):
        raise ManifestValidationError('Manifest must be an object with clips list')
    clips = []
    for raw in data['clips']:
        if not isinstance(raw, dict):
            raise ManifestValidationError('Clip must be an object')
        row = dict(raw)
        if not isinstance(row.get('person_annotations'), list):
            raise ManifestValidationError('person_annotations is required and must be a list')
        people = []
        for annotation in row['person_annotations']:
            if not isinstance(annotation, dict):
                raise ManifestValidationError('Person annotation must be an object')
            person = dict(annotation)
            for name in _INTERVAL_FIELDS:
                if person.get(name) is not None:
                    person[name] = _construct(TimeInterval, person[name])
            if not isinstance(person.get('visibility_gaps_s', []), list):
                raise ManifestValidationError('visibility_gaps_s must be a list')
            person['visibility_gaps_s'] = tuple(_construct(TimeInterval, gap) for gap in person.get('visibility_gaps_s', []))
            people.append(_construct(PersonAnnotation, person))
        row['person_annotations'] = tuple(people)
        row['frame_timing'] = _construct(FrameTimingSpec, row.get('frame_timing'))
        clips.append(_construct(ClipManifestRow, row))
    data['clips'] = tuple(clips)
    manifest = _construct(DatasetManifest, data)
    if verify_files:
        root = (Path(root_dir) if root_dir is not None else Path.cwd()).resolve()
        for row in manifest.clips:
            clip_path = (root / row.relative_path).resolve()
            if not clip_path.is_relative_to(root) or not clip_path.is_file():
                raise ManifestValidationError(f'{row.clip_id}: missing/local path violation: {clip_path}')
            with clip_path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != row.sha256:
                raise ManifestValidationError(f'{row.clip_id}: SHA-256 mismatch for {clip_path}: {actual}')
    return manifest


@_validated
def save_dataset_manifest(manifest: DatasetManifest, path: Path) -> None:
    if not isinstance(manifest, DatasetManifest):
        raise ManifestValidationError('Expected DatasetManifest')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
