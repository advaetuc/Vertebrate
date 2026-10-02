"""Ordered guards and exact source-time boundaries with controlled features."""
from dataclasses import replace
import math

import pytest

from vertebrate.config import CalibrationConfig
from vertebrate.contracts import PersonTrackKey
from vertebrate.features import Baseline, FeatureSummary, Kinematics
from vertebrate.fsm import PersonFSM, State

KEY = PersonTrackKey('test-session', 1, 1)


def feature(t, posture='down', *, v=0., m=0., b=0., key=KEY, history_reset=False):
    k = Kinematics(0., 2., .5, .5, True) if posture == 'down' else Kinematics(90., .4, 1., 0., True)
    return FeatureSummary(key, round(t * 1000), t, Baseline(200., 250., 300., 0., 1.),
                          (20., 350.) if posture == 'down' else (100., 150.),
                          (100., 350.) if posture == 'down' else (100., 250.),
                          (20., 350.), (100., 350.), k, k, v, m, b, .2, 1., history_reset)


def descend(fsm):
    assert fsm.update(feature(0, 'up')).state == State.MONITORING
    for t in (.1, .2, .3):
        result = fsm.update(feature(t, v=.8))
    assert result.state == State.DESCENT
    return result


def still(fsm):
    descend(fsm)
    for t in (.4, .5, .6):
        result = fsm.update(feature(t))
    assert result.state == State.VERIFYING_DOWN
    result = fsm.update(feature(.7))
    assert result.state == State.STILLNESS
    return result


def alert(fsm):
    still(fsm)
    results = [fsm.update(feature(i / 10)) for i in range(8, 38)]
    assert sum(r.alert is not None for r in results) == 1
    assert results[-1].alert is not None
    return results[-1]


def test_ordered_fall_and_zero_initial_stillness_credit():
    fsm = PersonFSM(KEY)
    result = alert(fsm)
    assert result.alert.onset_source_t_s == .1
    assert result.alert.down_source_t_s == .4
    assert result.alert.confirmed_source_t_s == 3.7
    assert result.alert.peak_v_h0_per_s == .8
    assert fsm.incident_id == result.alert.incident_id


@pytest.mark.parametrize('v', [None, -.8, 0, .59, math.nan, math.inf])
def test_slow_lie_sit_or_unknown_speed_never_arms_descent(v):
    fsm = PersonFSM(KEY)
    fsm.update(feature(0, 'up'))
    for i in range(1, 100):
        result = fsm.update(feature(i / 10, v=v))
        assert result.alert is None
    assert result.state == State.MONITORING


def test_already_down_without_baseline_is_informational():
    fsm = PersonFSM(KEY)
    for i in range(60):
        result = fsm.update(replace(feature(i / 10), baseline=None))
        assert result.state == State.ACQUIRING and result.reason == 'already_down'
        assert result.alert is None


def test_initial_displacement_required_and_rapid_evidence_must_be_contiguous():
    fsm = PersonFSM(KEY)
    fsm.update(feature(0, 'up'))
    for t in (.1, .2, .3):
        value = feature(t, v=.8)
        assert fsm.update(replace(value, raw=replace(value.raw, d=.14))).state == State.MONITORING
    fsm.update(feature(.4, v=.1))
    assert fsm.update(feature(.5, v=.8)).state == State.MONITORING
    assert fsm.update(feature(.6, v=.8)).state == State.MONITORING
    assert fsm.update(feature(.7, v=.8)).state == State.DESCENT


@pytest.mark.parametrize('field,value', [('d', .24), ('theta', 36.), ('g', False)])
def test_low_posture_requires_every_guard(field, value):
    fsm = PersonFSM(KEY)
    descend(fsm)
    for i in range(4, 15):
        f = feature(i / 10)
        result = fsm.update(replace(f, filtered=replace(f.filtered, **{field: value})))
        assert result.state == State.DESCENT
    assert fsm.update(feature(1.5)).reason == 'descent_deadline'


@pytest.mark.parametrize('r,q,low', [(1.19, .61, False), (1.2, .9, True), (.8, .6, True)])
def test_shape_gate_is_width_or_height(r, q, low):
    fsm = PersonFSM(KEY)
    descend(fsm)
    for t in (.4, .5, .6):
        f = feature(t)
        result = fsm.update(replace(f, filtered=replace(f.filtered, r=r, q=q)))
    assert (result.state == State.VERIFYING_DOWN) is low


@pytest.mark.parametrize('m,b', [(None, 0), (0, None), (.051, 0), (0, .041), (math.nan, 0), (0, math.inf), (-1, 0)])
def test_motion_or_unknown_resets_stillness_without_credit(m, b):
    fsm = PersonFSM(KEY)
    still(fsm)
    result = fsm.update(feature(.8, m=m, b=b))
    assert result.state == State.VERIFYING_DOWN and result.alert is None
    assert fsm.update(feature(.9)).reason == 'stillness_started'


def test_invalid_frame_at_confirmation_boundary_precedes_duration():
    fsm = PersonFSM(KEY)
    still(fsm)
    for i in range(8, 37):
        fsm.update(feature(i / 10))
    result = fsm.missing(3700, 3.7)
    assert result.state == State.VERIFYING_DOWN and result.alert is None
    # Even stale feature inputs cannot bypass the required fresh drift history.
    for i in range(38, 45):
        assert fsm.update(feature(i / 10)).state == State.VERIFYING_DOWN
    assert fsm.update(feature(4.5)).reason == 'stillness_started'
    for i in range(46, 75):
        assert fsm.update(feature(i / 10)).alert is None
    assert fsm.update(feature(7.5)).alert is not None


def test_repeated_short_invalid_frames_cannot_hide_long_loss():
    fsm = PersonFSM(KEY)
    still(fsm)
    assert fsm.missing(800, .8).state == State.VERIFYING_DOWN
    assert fsm.missing(900, .9).state == State.VERIFYING_DOWN
    result = fsm.missing(1000, 1.)
    assert result.state == State.ACQUIRING and result.inconclusive


@pytest.mark.parametrize('gap,reset', [(.25, False), (.25001, True), (20, True)])
def test_source_gap_boundary(gap, reset):
    fsm = PersonFSM(KEY)
    still(fsm)
    result = fsm.update(feature(.7 + gap))
    assert (result.state == State.ACQUIRING) is reset
    assert result.alert is None


def test_down_deadline_wins_over_stillness_completion():
    fsm = PersonFSM(KEY)
    still(fsm)
    for i in range(8, 96):
        fsm.update(feature(i / 10, m=1.))
    assert fsm.update(feature(9.6)).state == State.STILLNESS
    for i in range(97, 126):
        assert fsm.update(feature(i / 10)).alert is None
    result = fsm.update(feature(12.6))  # down entry .6 + 12; also stillness 9.6 + 3
    assert result.reason == 'down_deadline' and result.alert is None


@pytest.mark.parametrize('when,expected', [(1., State.DESCENT), (1.6, State.ACQUIRING)])
def test_loss_of_low_applies_original_descent_deadline_immediately(when, expected):
    fsm = PersonFSM(KEY)
    still(fsm)
    for i in range(8, round(when * 10)):
        fsm.update(feature(i / 10))
    f = feature(when)
    result = fsm.update(replace(f, filtered=replace(f.filtered, g=False)))
    assert result.state == expected and result.alert is None


def test_no_repeat_alert_and_full_upright_recovery_before_rearm():
    fsm = PersonFSM(KEY)
    first = alert(fsm).alert
    for i in range(38, 50):
        assert fsm.update(feature(i / 10)).alert is None
    for i in range(50, 70):
        assert fsm.update(feature(i / 10, 'up')).state == State.ALERTED
    result = fsm.update(feature(7., 'up'))
    assert result.state == State.ACQUIRING and result.reset_baseline
    assert fsm.incident_id == first.incident_id


def test_unstable_or_invalid_recovery_has_no_credit():
    fsm = PersonFSM(KEY)
    alert(fsm)
    for i in range(38, 48):
        fsm.update(feature(i / 10, 'up'))
    assert fsm.missing(4800, 4.8).state == State.ALERTED
    for i in range(49, 69):
        f = feature(i / 10, 'up')
        f = replace(f, hips_raw=(100 + i * 20, 250))
        assert fsm.update(f).state == State.ALERTED


def test_alert_survives_long_visibility_loss():
    fsm = PersonFSM(KEY)
    incident = alert(fsm).alert.incident_id
    assert fsm.missing(20000, 20).state == State.ALERTED
    assert fsm.incident_id == incident


def test_eof_and_source_error_never_extend_last_frame():
    for reason in ('end_of_stream', 'source_reset', 'source_error'):
        fsm = PersonFSM(KEY)
        still(fsm)
        result = fsm.close(reason)
        assert result.source_t_s == .7 and result.alert is None and result.inconclusive
        with pytest.raises(ValueError, match='closed'):
            fsm.update(feature(10))


def test_identity_cannot_transfer_candidate():
    fsm = PersonFSM(KEY)
    still(fsm)
    successor = PersonTrackKey(KEY.session_id, KEY.tracker_id, 2)
    with pytest.raises(ValueError, match='Identity'):
        fsm.update(feature(.8, key=successor))
    fresh = PersonFSM(successor)
    assert fresh.update(replace(feature(.8, key=successor), baseline=None)).state == State.ACQUIRING


@pytest.mark.parametrize('t,sequence', [(.7, 800), (.6, 800), (.8, 700)])
def test_bad_timestamp_or_sequence_closes_evidence(t, sequence):
    fsm = PersonFSM(KEY)
    still(fsm)
    with pytest.raises(ValueError, match='increasing'):
        fsm.update(replace(feature(t), sequence=sequence))
    assert fsm.closed


def test_fsm_never_reads_wall_clock(monkeypatch):
    import time
    def forbidden():
        raise AssertionError('FSM must not consult wall time')
    with monkeypatch.context() as patch:
        patch.setattr(time, 'time', forbidden)
        patch.setattr(time, 'monotonic', forbidden)
        patch.setattr(time, 'monotonic_ns', forbidden)
        assert alert(PersonFSM(KEY)).alert.confirmed_source_t_s == 3.7
