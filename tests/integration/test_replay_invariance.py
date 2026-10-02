"""Synthetic trajectory replay plus a real offline capture/pose/tracker runner.

Recorded trajectories exercise feature/FSM decisions, not model accuracy.
The separate fresh-process manifest test executes real local inference.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from vertebrate.config import AppConfig
from vertebrate.contracts import (EndOfStreamEvent, PersonTrackKey, PoseObservation,
                                  SessionResetEvent, SourceKind, TrackSample)
from vertebrate.fsm import State
from vertebrate.pipeline import HeadlessPipeline

ROOT = Path(__file__).resolve().parents[2]
KEY = PersonTrackKey('replay', 1, 1)


def trajectory(t, *, key=KEY, mode='fall'):
    p = min(1., max(0., (t - 1) / (.6 if mode != 'slow' else 5.)))
    if mode == 'down':
        p = 1.
    if mode == 'sit':
        p = min(p, .4)
    if mode == 'recover' and t >= 2.:
        p = max(0., 1. - (t - 2) / .5)
    xy = np.full((17, 2), 100., dtype=float)
    for a, b, midpoint in ((5, 6, (100 - 80 * p, 150 + 200 * p)),
                           (11, 12, (100., 250 + 100 * p)),
                           (13, 14, (100., 275 + 75 * p)), (15, 16, (100., 300 + 50 * p))):
        xy[a], xy[b] = (midpoint[0] - 10, midpoint[1]), (midpoint[0] + 10, midpoint[1])
    return TrackSample(key, round(t * 10), t, np.array([0., 100., 80. + 140 * p, 300. - 100 * p]),
                       .9, xy, np.ones(17), np.ones(17, bool), True)


def start_pipeline():
    pipe = HeadlessPipeline(root_dir=ROOT)
    pipe.handle_event(SessionResetEvent(None, KEY.session_id, 'start', SourceKind.VIDEO, 'synthetic-trajectory'))
    return pipe


def replay(*, pace=0., mode='fall', end=7., omit=(), successor_at=None):
    pipe = start_pipeline()
    decisions = []
    for i in range(round(end * 10) + 1):
        t = i / 10
        key = replace(KEY, generation=2) if successor_at is not None and t >= successor_at else KEY
        samples = () if i in omit else (trajectory(t, key=key, mode=mode),)
        decisions.extend(pipe.process_tracks(samples, sequence=i, source_t_s=t))
        if pace:
            time.sleep(pace)
    return pipe, decisions


def test_exact_replay_invariance_under_5_and_300_fps_target_pacing():
    # Actual deliberate wall delays, not mocked integration or measured FPS claims.
    # Processing overhead adds to target periods; decisions must still be identical.
    before = time.perf_counter()
    slow, slow_decisions = replay(pace=1 / 5, end=6.)
    slow_elapsed = time.perf_counter() - before
    before = time.perf_counter()
    fast, fast_decisions = replay(pace=1 / 300, end=6.)
    fast_elapsed = time.perf_counter() - before
    assert slow_elapsed > 10 and fast_elapsed < slow_elapsed
    assert len(slow.alerts) == 1
    assert slow.alerts == fast.alerts
    assert slow_decisions == fast_decisions
    assert slow.transitions == fast.transitions
    assert slow.alerts[0].confirmed_source_t_s == 5.6


@pytest.mark.parametrize('mode', ['down', 'slow', 'sit', 'recover'])
def test_synthetic_negative_trajectories_have_no_alert(mode):
    pipe, decisions = replay(mode=mode, end=8.)
    assert pipe.alerts == []
    if mode == 'down':
        assert all(d.state == State.ACQUIRING for d in decisions)
        assert any(d.reason == 'already_down' for d in decisions)


def test_missing_track_at_confirmation_cannot_add_stillness_credit():
    pipe, _ = replay(end=7., omit=(56,))
    assert pipe.alerts == []
    assert any(d.state == State.VERIFYING_DOWN for d in pipe.transitions)


def test_identity_generation_change_requires_new_baseline():
    pipe, decisions = replay(successor_at=4., end=8.)
    assert pipe.alerts == []
    assert any(d.reason == 'identity_expired' and d.inconclusive for d in decisions)
    assert all(d.state == State.ACQUIRING for d in decisions if d.person_key.generation == 2)


def test_eof_closes_incomplete_candidate_at_last_source_time():
    pipe, _ = replay(end=5.5)
    result = pipe.handle_event(EndOfStreamEvent(KEY.session_id, 'synthetic', 56, 55, 5.5))
    assert pipe.alerts == []
    assert result[0].source_t_s == 5.5 and result[0].inconclusive
    with pytest.raises(ValueError, match='open'):
        pipe.process_tracks((trajectory(5.6),), sequence=56, source_t_s=5.6)


def test_reset_and_error_do_not_carry_candidate_to_new_source():
    pipe, _ = replay(end=4.)
    result = pipe.handle_event(SessionResetEvent(KEY.session_id, 'new-session', 'seek', SourceKind.VIDEO, 'local'))
    assert result[0].inconclusive
    value = trajectory(0, key=PersonTrackKey('new-session', 1, 1), mode='down')
    assert pipe.process_tracks((value,), sequence=0, source_t_s=0)[0].state == State.ACQUIRING
    pipe.source_error()
    assert pipe.closed and pipe.alerts == []


def test_two_people_have_independent_histories_and_alerts():
    pipe = start_pipeline()
    second = PersonTrackKey(KEY.session_id, 2, 1)
    for i in range(70):
        t = i / 10
        pipe.process_tracks((trajectory(t), trajectory(t, key=second, mode='down')),
                            sequence=i, source_t_s=t)
    assert len(pipe.alerts) == 1 and pipe.alerts[0].person_key == KEY


def test_missing_people_expire_and_stale_generation_is_rejected():
    pipe, _ = replay(end=3., omit=tuple(range(20, 31)))
    with pytest.raises(ValueError, match='Stale'):
        pipe.process_tracks((trajectory(3.1),), sequence=31, source_t_s=3.1)


def test_recorded_pose_replay_uses_real_bytetrack_and_feature_fsm():
    pipe = start_pipeline()
    for i in range(70):
        s = trajectory(i / 10)
        obs = PoseObservation(KEY.session_id, i, i / 10, 1280, 720,
                              s.bbox_xyxy[None, :], np.array([.9]), s.keypoints_xy[None, :],
                              s.keypoint_confidences[None, :], s.keypoint_valid_mask[None, :])
        pipe.process_observation(obs)
    assert len(pipe.alerts) == 1
    assert pipe.tracker.network_attempts == []


def test_real_manifest_runner_in_fresh_process_has_no_qt_or_network(tmp_path):
    # Actual three files, local model inference, real ByteTrack, CaptureWorker,
    # and shared features/FSM. No detector mock, no synthetic accuracy claim.
    target = tmp_path / 'result.json'
    code = '''import json, sys
from pathlib import Path
from vertebrate.evaluation.runner import run_manifest
report = run_manifest('data/manifests/dev.json')
assert not any(n == 'PySide6' or n.startswith('PySide6.') for n in sys.modules)
Path(sys.argv[1]).write_text(json.dumps(report, allow_nan=False))
'''
    result = subprocess.run([sys.executable, '-c', code, str(target)], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(target.read_text())
    assert len(report['clips']) == 3 and report['network_attempts'] == []
    for clip in report['clips']:
        assert clip['frames'] == 100 and clip['dropped_frames'] == 0
        assert clip['network_attempts'] == []
    assert report['model_sha256'] == '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'


def test_runner_missing_manifest_fails_before_model_load(tmp_path, monkeypatch):
    from vertebrate.evaluation import runner
    def forbidden(*args, **kwargs):
        pytest.fail('Model must not load before manifest verification')
    monkeypatch.setattr(runner, 'PoseAdapter', forbidden)
    with pytest.raises(ValueError):
        runner.run_manifest(tmp_path / 'absent.json')
