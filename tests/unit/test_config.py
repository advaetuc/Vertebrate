from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path

import pytest

from vertebrate.config import (AppConfig, CalibrationConfig, ConfigValidationError,
                               ModelConfig, RuntimeConfig, TrackerConfig, config_sha256,
                               config_to_dict, increment_revision, load_config)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('filename,profile,duration', [
    ('demo.json', 'demo-3s', 3.0), ('observation-10s.json', 'observation-10s', 10.0)])
def test_canonical_profiles(filename, profile, duration):
    cfg = load_config(ROOT / 'config' / filename)
    assert cfg.profile == profile
    assert cfg.calibration.stillness_duration_s == duration
    assert len(fields(CalibrationConfig)) == 34
    assert cfg.model.imgsz == 512
    assert cfg.runtime.max_media_buffer_bytes == 128 * 1024 * 1024
    with pytest.raises(FrozenInstanceError):
        cfg.revision = 2
    with pytest.raises(FrozenInstanceError):
        cfg.model.imgsz = 640


def test_hash_and_revision(tmp_path):
    cfg = load_config(ROOT / 'config/demo.json')
    path = tmp_path / 'reordered.json'
    path.write_text(json.dumps(config_to_dict(cfg), sort_keys=True, indent=4))
    assert config_sha256(cfg) == config_sha256(load_config(path))
    assert len(config_sha256(cfg)) == 64
    changed = increment_revision(cfg, calibration={'stillness_duration_s': 4.0})
    assert changed.revision == 2 and cfg.revision == 1
    assert changed.calibration.stillness_duration_s == 4
    assert changed.calibration.recovery_dwell_s == cfg.calibration.recovery_dwell_s
    assert config_sha256(changed) != config_sha256(cfg)
    with pytest.raises(ConfigValidationError):
        increment_revision(cfg, revision=7)
    with pytest.raises(ConfigValidationError):
        increment_revision(cfg, bogus=True)


@pytest.mark.parametrize('cls,overrides', [
    (ModelConfig, {'imgsz': 500}), (ModelConfig, {'imgsz': True}),
    (ModelConfig, {'det_conf_floor': 0}), (ModelConfig, {'det_conf_floor': 1}),
    (ModelConfig, {'det_conf_floor': float('nan')}), (ModelConfig, {'task': 'detect'}),
    (ModelConfig, {'model_path': 'https://example.com/model.pt'}),
    (TrackerConfig, {'track_high_thresh': 1.1}),
    (TrackerConfig, {'track_low_thresh': .5, 'track_high_thresh': .2}),
    (TrackerConfig, {'track_buffer': 0}),
    (CalibrationConfig, {'completed_descent_d': .1}),
    (CalibrationConfig, {'stillness_duration_s': 12}),
    (CalibrationConfig, {'stillness_pair_min_dt_s': .3}),
    (CalibrationConfig, {'stillness_pair_target_dt_s': .4}),
    (CalibrationConfig, {'velocity_min_span_s': .5}),
    (CalibrationConfig, {'upright_min_theta_deg': 91}),
    (CalibrationConfig, {'keypoint_confidence': 1.5}),
    (RuntimeConfig, {'enable_optional_clip': 'false'}),
    (RuntimeConfig, {'capture_queue_capacity': 2.5}),
    (AppConfig, {'revision': 0}), (AppConfig, {'profile': 'unknown'}),
])
def test_reject_invalid_dataclasses(cls, overrides):
    with pytest.raises(ConfigValidationError):
        cls(**overrides)


@pytest.mark.parametrize('item', fields(CalibrationConfig))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_every_calibration_number_is_validated(item, bad):
    with pytest.raises(ConfigValidationError):
        CalibrationConfig(**{item.name: bad})


@pytest.mark.parametrize('item', [f for f in fields(CalibrationConfig) if f.name.endswith('_s')])
def test_zero_durations_rejected(item):
    with pytest.raises(ConfigValidationError):
        CalibrationConfig(**{item.name: 0})


@pytest.mark.parametrize('payload', [
    '{"calibration":{"filter_tau_s":NaN}}',
    '{"calibration":{"filter_tau_s":Infinity}}',
    '{"calibration":{"filter_tau_s":1e999}}',
    '{"mystery":1}', '{"model":{"mystery":1}}',
    '{"runtime":{"incident_queue_capacity":0}}',
    '{"tracker":{"track_low_thresh":0.9}}',
    '{"revision":1,"revision":2}', '[]', '{"model":null}', '{broken',
])
def test_strict_json(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(payload)
    with pytest.raises(ConfigValidationError):
        load_config(path)
