from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import re
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

import pytest

from vertebrate.incidents import incident_from_dict, incident_from_json, incident_to_dict, incident_to_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def illustrative():
    blueprint = (ROOT / 'docs/03-validated-blueprint-v2.md').read_text(encoding='utf-8')
    section = blueprint.split('## 10. Incident schema and local dispatch')[1]
    return json.loads(re.search(r'```json\s*(.*?)\s*```', section, re.S).group(1))


def test_exact_illustrative_schema_round_trip(illustrative):
    incident = incident_from_dict(illustrative)
    assert incident_to_dict(incident) == illustrative
    assert incident_from_json(incident_to_json(incident)) == incident
    with pytest.raises(FrozenInstanceError):
        incident.mode = 'live'


@pytest.mark.parametrize('field', ['schema_version', 'incident_id', 'source', 'person', 'timing', 'location',
                                 'configuration', 'kinematics', 'evidence', 'environment', 'media',
                                 'delivery', 'acknowledgement'])
def test_missing_required_fields(illustrative, field):
    illustrative.pop(field)
    with pytest.raises(ValueError):
        incident_from_dict(illustrative)


@pytest.mark.parametrize('section,field,value', [
    (None, 'schema_version', '2.0'), (None, 'incident_id', str(uuid5(NAMESPACE_URL, 'bad'))),
    (None, 'incident_id', '../escape'), (None, 'mode', 'real_ems'), (None, 'event_type', 'diagnosis'),
    ('source', 'kind', 'rtsp'), ('person', 'tracker_id', True), ('person', 'generation', 0),
    ('configuration', 'revision', 0), ('configuration', 'profile', 'unknown'),
    ('kinematics', 'torso_angle_deg', 91), ('kinematics', 'net_hip_descent_h0', float('nan')),
    ('kinematics', 'bbox_width_height_ratio', float('inf')), ('kinematics', 'common_core_keypoints', 5),
    ('kinematics', 'common_core_keypoints', 9), ('kinematics', 'common_core_keypoints', True),
    ('evidence', 'near_floor', False), ('evidence', 'rapid_descent', 1),
    ('delivery', 'external_dispatch', True), ('delivery', 'status', 'delivered'),
    ('environment', 'source', 'external_api'), ('environment', 'relative_humidity_pct', 101),
    ('location', 'latitude', 91), ('acknowledgement', 'status', 'read'),
    ('timing', 'confirmed_at_utc', '2026-10-02T12:00:00'),
    ('timing', 'fall_onset_utc', '2026-10-03T00:00:00Z'),
    ('timing', 'fall_onset_source_s', 13), ('timing', 'down_posture_source_s', 15),
    ('timing', 'stillness_start_source_s', 17), ('timing', 'confirmed_source_s', -1),
    ('kinematics', 'stillness_observed_s', 4),
])
def test_rejects_types_enums_ranges_and_time_order(illustrative, section, field, value):
    (illustrative if section is None else illustrative[section])[field] = value
    with pytest.raises(ValueError):
        incident_from_dict(illustrative)


@pytest.mark.parametrize('path', ['../snapshot.jpg', '/snapshot.jpg', 'C:/snapshot.jpg', 'file://x',
                                 'a/../../x', 'a\\x', './x', 'a//x', 'incident.json', 'CONFIG.JSON',
                                 'NUL', 'CON.jpg', 'x.jpg.', 'x.jpg ', 'x?.jpg'])
def test_unsafe_media_paths(illustrative, path):
    illustrative['media']['snapshot_path'] = path
    with pytest.raises(ValueError):
        incident_from_dict(illustrative)


def test_unknown_duplicate_and_nonfinite_json_fields(illustrative):
    illustrative['unexpected'] = True
    with pytest.raises(ValueError, match='unknown'):
        incident_from_dict(illustrative)
    with pytest.raises(ValueError, match='Duplicate'):
        incident_from_json('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(ValueError, match='Non-finite'):
        incident_from_json('{"x":NaN}')


def test_acknowledgement_order_and_snapshot_failure_reason(illustrative):
    illustrative['acknowledgement'] = {'status': 'acknowledged', 'at_utc': '2026-09-28T10:29:00Z'}
    with pytest.raises(ValueError, match='precede'):
        incident_from_dict(illustrative)
    illustrative['acknowledgement']['at_utc'] = '2026-09-28T10:31:00Z'
    illustrative['media']['snapshot_path'] = None
    with pytest.raises(ValueError, match='snapshot_error'):
        incident_from_dict(illustrative)
    illustrative['media']['snapshot_error'] = 'disk unavailable'
    assert incident_from_dict(illustrative).media.snapshot_path is None


def test_uuid4_and_json_finite(illustrative):
    base = incident_from_dict(illustrative)
    first, second = replace(base, incident_id=str(uuid4())), replace(base, incident_id=str(uuid4()))
    assert first.incident_id != second.incident_id
    assert UUID(first.incident_id).version == 4
    json.loads(incident_to_json(first), parse_constant=lambda _: pytest.fail('Non-finite JSON'))
