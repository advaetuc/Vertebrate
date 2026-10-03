"""Frozen schema-1.0 local incident records and strict JSON round trips.

The blueprint's illustrative keys are preserved. Optional *_reason/*_error
fields supply the explicit diagnostics it requires for unavailable data.
Replay candidate IDs remain deterministic; persisted incident IDs are UUIDv4.
"""
from dataclasses import dataclass, field, fields
from datetime import timedelta
import json
from pathlib import PurePosixPath, PureWindowsPath
from uuid import UUID, uuid4

from .config import AppConfig
from .context import Environment, environment_at
from .validation import integer, number, text, utc_datetime, utc_string


def choice(value, values, name):
    if type(value) is not str or value not in values:
        raise ValueError(f'{name} must be one of {values}')


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be boolean')


def relative_path(value):
    text(value, 'relative path')
    path = PurePosixPath(value)
    if ('\\' in value or ':' in value or path.is_absolute() or PureWindowsPath(value).drive
            or any(p in ('', '.', '..') or p.endswith((' ', '.')) or PureWindowsPath(p).is_reserved()
                   or any(c in p for c in '<>"|?*') for p in value.split('/'))):
        raise ValueError('Media/config paths must be safe relative paths')


@dataclass(frozen=True)
class Source:
    kind: str
    name: str
    recorded_start_utc: str | None = None

    def __post_init__(self):
        choice(self.kind, ('live', 'video'), 'source.kind')
        text(self.name, 'source.name')
        if self.recorded_start_utc is not None:
            utc_datetime(self.recorded_start_utc)


@dataclass(frozen=True)
class Person:
    tracker_id: int
    generation: int

    def __post_init__(self):
        integer(self.tracker_id, 'tracker_id')
        integer(self.generation, 'generation', 1)


@dataclass(frozen=True)
class Timing:
    fall_onset_source_s: float
    down_posture_source_s: float
    stillness_start_source_s: float
    confirmed_source_s: float
    fall_onset_utc: str | None
    confirmed_at_utc: str
    fall_onset_utc_reason: str | None = field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        times = [self.fall_onset_source_s, self.down_posture_source_s, self.stillness_start_source_s, self.confirmed_source_s]
        for value in times:
            number(value, 'source time')
        if times != sorted(times):
            raise ValueError('Require fall_onset <= down_posture <= stillness_start <= confirmed')
        confirmed = utc_datetime(self.confirmed_at_utc)
        if self.fall_onset_utc is not None:
            if utc_datetime(self.fall_onset_utc) > confirmed:
                raise ValueError('Fall onset UTC must not follow analysis confirmation UTC')
            if self.fall_onset_utc_reason is not None:
                raise ValueError('Known fall onset UTC cannot have an unknown reason')
        elif self.fall_onset_utc_reason is not None:
            text(self.fall_onset_utc_reason, 'fall_onset_utc_reason')


@dataclass(frozen=True)
class Location:
    label: str
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self):
        text(self.label, 'location.label')
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError('Latitude and longitude must both be present or null')
        if self.latitude is not None:
            number(self.latitude, 'latitude', -90, 90)
            number(self.longitude, 'longitude', -180, 180)


@dataclass(frozen=True)
class Configuration:
    profile: str
    revision: int
    snapshot_path: str = 'config.json'

    def __post_init__(self):
        choice(self.profile, ('demo-3s', 'observation-10s'), 'profile')
        integer(self.revision, 'revision', 1)
        relative_path(self.snapshot_path)


@dataclass(frozen=True)
class Kinematics:
    peak_downward_velocity_h0_per_s: float
    net_hip_descent_h0: float
    torso_angle_deg: float
    bbox_width_height_ratio: float
    bbox_height_h0_ratio: float
    stillness_observed_s: float
    common_core_keypoints: int

    def __post_init__(self):
        for name in ('peak_downward_velocity_h0_per_s', 'net_hip_descent_h0', 'bbox_width_height_ratio',
                     'bbox_height_h0_ratio', 'stillness_observed_s'):
            number(getattr(self, name), name)
        number(self.torso_angle_deg, 'torso_angle_deg', 0, 90)
        integer(self.common_core_keypoints, 'common_core_keypoints', 6)
        if self.common_core_keypoints > 8:
            raise ValueError('Common core contains at most 8 landmarks')
        if self.bbox_width_height_ratio == 0 or self.bbox_height_h0_ratio == 0:
            raise ValueError('Box ratios must be positive')


@dataclass(frozen=True)
class Evidence:
    rapid_descent: bool
    low_posture: bool
    near_floor: bool
    sustained_stillness: bool

    def __post_init__(self):
        for f in fields(self):
            boolean(getattr(self, f.name), f.name)
            if not getattr(self, f.name):
                raise ValueError('Confirmed incident requires all ordered evidence gates')


@dataclass(frozen=True)
class Media:
    snapshot_path: str | None = 'snapshot.jpg'
    clip_path: str | None = None
    clip_status: str = 'not_requested'
    snapshot_error: str | None = field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        for path in (self.snapshot_path, self.clip_path):
            if path is not None:
                relative_path(path)
        choice(self.clip_status, ('not_requested', 'saved', 'truncated', 'unavailable'), 'clip_status')
        if (self.clip_status in ('saved', 'truncated')) != (self.clip_path is not None):
            raise ValueError('Clip status and path disagree')
        if self.snapshot_path is None:
            text(self.snapshot_error, 'snapshot_error')
        elif self.snapshot_error is not None:
            raise ValueError('A snapshot error requires snapshot_path=null')


@dataclass(frozen=True)
class Delivery:
    destination: str = 'local_inbox'
    status: str = 'pending'
    external_dispatch: bool = False

    def __post_init__(self):
        choice(self.destination, ('local_inbox',), 'destination')
        choice(self.status, ('pending', 'saved', 'save_failed'), 'delivery.status')
        if self.external_dispatch is not False:
            raise ValueError('External dispatch must be false')


@dataclass(frozen=True)
class Acknowledgement:
    status: str = 'unacknowledged'
    at_utc: str | None = None

    def __post_init__(self):
        choice(self.status, ('unacknowledged', 'acknowledged'), 'acknowledgement.status')
        if self.status == 'acknowledged':
            utc_datetime(self.at_utc)
        elif self.at_utc is not None:
            raise ValueError('Unacknowledged incident cannot have acknowledgement time')


@dataclass(frozen=True)
class Incident:
    session_id: str
    source: Source
    person: Person
    timing: Timing
    location: Location
    configuration: Configuration
    kinematics: Kinematics
    evidence: Evidence
    environment: Environment
    media: Media = field(default_factory=Media)
    delivery: Delivery = field(default_factory=Delivery)
    acknowledgement: Acknowledgement = field(default_factory=Acknowledgement)
    incident_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = '1.0'
    event_type: str = 'suspected_fall_with_stillness'
    mode: str = 'local_demo'

    def __post_init__(self):
        text(self.session_id, 'session_id')
        text(self.incident_id, 'incident_id')
        parsed = UUID(self.incident_id)
        if parsed.version != 4 or str(parsed) != self.incident_id:
            raise ValueError('incident_id must be a canonical UUIDv4')
        for name, cls in _SECTIONS.items():
            if type(getattr(self, name)) is not cls:
                raise ValueError(f'{name} must be {cls.__name__}')
        choice(self.schema_version, ('1.0',), 'schema_version')
        choice(self.event_type, ('suspected_fall_with_stillness',), 'event_type')
        choice(self.mode, ('local_demo',), 'mode')
        if self.source.kind == 'video' and self.source.recorded_start_utc is None and self.timing.fall_onset_utc is not None:
            raise ValueError('Historical event UTC is unknown without explicit recording UTC')
        if self.acknowledgement.at_utc is not None and utc_datetime(self.acknowledgement.at_utc) < utc_datetime(self.timing.confirmed_at_utc):
            raise ValueError('Acknowledgement must not precede confirmation')
        observed = self.timing.confirmed_source_s - self.timing.stillness_start_source_s
        if abs(observed - self.kinematics.stillness_observed_s) > 1e-9:
            raise ValueError('Stillness duration disagrees with source-time evidence')
        paths = [self.configuration.snapshot_path, self.media.snapshot_path, self.media.clip_path]
        present = [p for p in paths if p is not None]
        if len({p.casefold() for p in present}) != len(present) or any(p.lower() == 'incident.json' for p in present):
            raise ValueError('Incident media/config paths collide')


_SECTIONS = {'source': Source, 'person': Person, 'timing': Timing, 'location': Location,
             'configuration': Configuration, 'kinematics': Kinematics, 'evidence': Evidence,
             'environment': Environment, 'media': Media, 'delivery': Delivery, 'acknowledgement': Acknowledgement}


def incident_to_dict(incident: Incident) -> dict:
    def convert(value):
        return {f.name: convert(getattr(value, f.name)) if hasattr(getattr(value, f.name), '__dataclass_fields__') else getattr(value, f.name)
                for f in fields(value) if not (f.metadata.get('omit_none') and getattr(value, f.name) is None)}
    if type(incident) is not Incident:
        raise ValueError('Expected Incident')
    return convert(incident)


def incident_to_json(incident: Incident) -> str:
    return json.dumps(incident_to_dict(incident), sort_keys=True, indent=2, allow_nan=False)


def _construct(cls, data):
    if type(data) is not dict:
        raise ValueError(f'{cls.__name__} must be an object')
    allowed = {f.name for f in fields(cls)}
    # JSON documents require every illustrative field; optional diagnostic
    # extensions are the only fields whose absence is permitted on load.
    required = {f.name for f in fields(cls) if not f.metadata.get('omit_none')}
    if data.keys() - allowed or required - data.keys():
        raise ValueError(f'{cls.__name__}: unknown or missing fields')
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def incident_from_dict(data: dict) -> Incident:
    if type(data) is not dict:
        raise ValueError('Incident must be an object')
    values = dict(data)
    for name, cls in _SECTIONS.items():
        values[name] = _construct(cls, values.get(name))
    return _construct(Incident, values)


def incident_from_json(value: str) -> Incident:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = item
        return result
    def bad_constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return incident_from_dict(json.loads(value, object_pairs_hook=pairs, parse_constant=bad_constant))


class IncidentFactory:
    """One UUIDv4 per candidate token; retries reuse that immutable incident."""
    def __init__(self):
        self._incidents = {}

    def create(self, alert, features, frame, config: AppConfig, *, confirmed_at_utc: str,
               environment: Environment | None = None, location: Location | None = None) -> Incident:
        token = (alert.person_key, alert.incident_id)
        if token in self._incidents:
            return self._incidents[token]
        if (features.person_key != alert.person_key or frame.session_id != alert.person_key.session_id
                or features.sequence != alert.sequence or frame.sequence != alert.sequence
                or features.source_t_s != alert.confirmed_source_t_s or frame.source_t_s != alert.confirmed_source_t_s):
            raise ValueError('Incident evidence/snapshot must match confirmation frame and identity')
        if alert.stillness_start_source_t_s is None or features.common_core_keypoints is None:
            raise ValueError('Incident requires observed stillness start and common-core count')
        offset = frame.source_time_offset_s
        anchor = frame.recorded_start_utc if frame.source_kind.value == 'video' else frame.session_start_utc
        event_seconds = alert.onset_source_t_s + (offset if frame.source_kind.value == 'video' else 0)
        onset_utc = utc_string(utc_datetime(anchor) + timedelta(seconds=event_seconds)) if anchor else None
        k = features.filtered
        incident = Incident(
            session_id=alert.person_key.session_id,
            source=Source(frame.source_kind.value, frame.source_name, frame.recorded_start_utc),
            person=Person(alert.person_key.tracker_id, alert.person_key.generation),
            timing=Timing(alert.onset_source_t_s + offset, alert.down_source_t_s + offset,
                          alert.stillness_start_source_t_s + offset, alert.confirmed_source_t_s + offset,
                          onset_utc, confirmed_at_utc, None if onset_utc else 'source_utc_unavailable'),
            location=location or Location('Unspecified'),
            configuration=Configuration(config.profile, config.revision),
            kinematics=Kinematics(alert.peak_v_h0_per_s, k.d, k.theta, k.r, k.q,
                                  alert.confirmed_source_t_s - alert.stillness_start_source_t_s,
                                  features.common_core_keypoints),
            evidence=Evidence(True, True, k.g, True),
            environment=environment_at(environment or Environment(), confirmed_at_utc,
                                       config.runtime.context_staleness_minutes))
        self._incidents[token] = incident
        return incident
