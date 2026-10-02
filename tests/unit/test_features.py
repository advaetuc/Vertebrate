"""Deterministic geometry and source-time tests; no detector mocks or network."""
from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from vertebrate.config import CalibrationConfig
from vertebrate.contracts import PersonTrackKey, TrackSample
from vertebrate.features import (BaselineBuilder, FeatureExtractor, midpoint,
                                  normalized_velocity, torso_angle)
from vertebrate.smoothing import TimestampAwareFilter

C = CalibrationConfig()


def sample(t, *, sequence=None, height=200., width=80., shift=(0., 0.), missing=(),
           key=None, valid=True, shoulder=(100., 150.), hip=(100., 250.), ankle_y=300.):
    xy = np.tile([100., 200.], (17, 1)).astype(float)
    for a, b, point in ((5, 6, shoulder), (11, 12, hip), (13, 14, (100., 275.)), (15, 16, (100., ankle_y))):
        xy[a], xy[b] = (point[0] - 10, point[1]), (point[0] + 10, point[1])
    xy += np.array(shift)
    mask = np.ones(17, bool)
    mask[list(missing)] = False
    return TrackSample(key or PersonTrackKey('session', 1, 1),
                       int(round(t * 1000)) if sequence is None else sequence, t,
                       np.array([60., 100., 60. + width, 100. + height]), .9,
                       xy, np.ones(17), mask, valid)


def acquire(owner):
    for i in range(11):
        result = owner.update(sample(i / 10))
    return result


@pytest.mark.parametrize('dt', [.01, .06, .20, .25])
def test_filter_alpha_math(dt):
    filt = TimestampAwareFilter()
    assert filt.update((10, 20), 0) == (10, 20)
    alpha = 1 - math.exp(-dt / .06)
    assert filt.update((30, 0), dt) == pytest.approx((10 + 20 * alpha, 20 - 20 * alpha))


@pytest.mark.parametrize('point', [None, (math.nan, 0), (0, math.inf), (0, -math.inf), (1,), ('bad', 2)])
def test_invalid_points_reset_filter(point):
    filt = TimestampAwareFilter()
    filt.update((10, 20), 0)
    assert filt.update(point, .1) is None
    assert filt.update((100, 200), .2) == (100, 200)


def test_filter_missing_flag_and_gap_boundary():
    filt = TimestampAwareFilter()
    filt.update((0, 0), 0)
    assert filt.update((100, 200), .1, valid=False) is None
    filt.update((0, 0), .2)
    assert filt.update((100, 200), .450001) == (100, 200)
    filt.reset()
    assert filt.point is None


@pytest.mark.parametrize('time', [0., -.1, math.nan, math.inf])
def test_filter_rejects_bad_time_and_clears(time):
    filt = TimestampAwareFilter()
    filt.update((1, 2), 0)
    with pytest.raises(ValueError):
        filt.update((3, 4), time)
    assert filt.point is None


@pytest.mark.parametrize('kwargs', [{'tau_s': 0}, {'tau_s': -1}, {'tau_s': math.nan},
                                  {'max_observation_gap_s': 0}, {'max_observation_gap_s': math.inf}])
def test_filter_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        TimestampAwareFilter(**kwargs)


def test_baseline_waits_full_interval_and_publishes_medians():
    builder = BaselineBuilder()
    for i in range(10):
        assert builder.update(sample(i / 10, height=200 + i % 3, shift=(0, i % 3))) is None
    baseline = builder.update(sample(1., height=201, shift=(0, 1)))
    assert (baseline.H0, baseline.Hy0, baseline.Fy0) == (201., 251., 301.)
    assert baseline.start_source_t_s == 0 and baseline.end_source_t_s == 1
    with pytest.raises(FrozenInstanceError):
        baseline.H0 = 5


def test_irregular_acquisition_cadence_no_interpolation():
    builder = BaselineBuilder()
    for t in (0., .17, .34, .51, .68, .85):
        assert builder.update(sample(t)) is None
    assert builder.update(sample(1.02)).H0 == 200


@pytest.mark.parametrize('missing', [(5,), (6,), (11,), (12,), (15,), (16,)])
def test_baseline_requires_bilateral_body_and_ankles(missing):
    builder = BaselineBuilder()
    for i in range(15):
        assert builder.update(sample(i / 10, missing=missing)) is None


@pytest.mark.parametrize('kwargs', [{'height': 159.}, {'shoulder': (100., 249.)},
                                  {'shoulder': (20., 250.)}, {'valid': False}])
def test_baseline_rejects_tiny_degenerate_down_or_invalid(kwargs):
    builder = BaselineBuilder()
    for i in range(15):
        assert builder.update(sample(i / 10, **kwargs)) is None


def test_minimum_initial_height_scales_with_source_resolution():
    builder = BaselineBuilder(frame_height=360)
    for i in range(11):
        result = builder.update(sample(i / 10, height=80.))
    assert result.H0 == 80


@pytest.mark.parametrize('axis', [0, 1])
def test_baseline_hip_drift_resets_acquisition_and_monitoring(axis):
    builder = BaselineBuilder()
    acquire(builder)
    shift = [0, 0]
    shift[axis] = 11  # > .05 * 200; slow enough not rapid descent.
    assert builder.update(sample(1.2, shift=shift)) is None
    for i in range(1, 11):
        result = builder.update(sample(1.2 + i / 10, shift=shift))
    assert result.Hy0 == 250 + shift[1]


def test_baseline_uses_full_interval_drift_not_only_endpoint():
    builder = BaselineBuilder()
    for t, x in [(0., 0), (.2, 6), (.4, 12), (.6, 6), (.8, 0), (1., 0)]:
        assert builder.update(sample(t, shift=(x, 0))) is None


def test_baseline_height_growth_resets_but_boundary_does_not():
    builder = BaselineBuilder()
    acquire(builder)
    assert builder.update(sample(1.1, height=240.)) is not None
    assert builder.update(sample(1.2, height=242.)) is None


@pytest.mark.parametrize('kwargs', [{'height': 100.}, {'shoulder': (20., 250.)}, {'shift': (0, 20)}])
def test_collapsing_geometry_does_not_apply_depth_reset(kwargs):
    builder = BaselineBuilder()
    baseline = acquire(builder)
    assert builder.update(sample(1.1, **kwargs)) == baseline
    # Candidate latch preserves H0 throughout the descent/down plateau.
    for i in range(12, 23):
        assert builder.update(sample(i / 10, **kwargs), candidate_active=True) == baseline


def test_gap_zone_and_explicit_candidate_freeze():
    builder = BaselineBuilder()
    baseline = acquire(builder)
    assert builder.update(sample(1.3), candidate_active=True) == baseline
    assert builder.update(sample(1.4), candidate_active=True, in_zone=False) is None
    builder = BaselineBuilder()
    acquire(builder)
    assert builder.update(sample(1.251)) is None


@pytest.mark.parametrize('pair,missing', [((5, 6), (5,)), ((5, 6), (6,)), ((11, 12), (11,)), ((11, 12), (12,))])
def test_midpoints_never_substitute_one_side(pair, missing):
    assert midpoint(sample(0, missing=missing), pair) is None


def test_midpoint_uses_confidence_and_prediction_flags():
    value = sample(0)
    conf = value.keypoint_confidences.copy()
    conf[5] = .34
    assert midpoint(replace(value, keypoint_confidences=conf), (5, 6)) is None
    assert midpoint(replace(value, observation_valid=False, predicted_only=True), (11, 12)) is None


@pytest.mark.parametrize('S,H,expected', [((0, 0), (0, 10), 90), ((0, 0), (10, 0), 0),
                                       ((10, 0), (0, 10), 45), ((0, 10), (10, 0), 45)])
def test_torso_angle_quadrants(S, H, expected):
    assert torso_angle(S, H) == pytest.approx(expected)


def test_raw_and_filtered_geometry_arithmetic_and_frozen_baseline():
    features = FeatureExtractor()
    initial = acquire(features)
    result = features.update(sample(1.2, width=240, height=100,
                                   shoulder=(40, 280), hip=(140, 280)), candidate_active=True)
    assert result.baseline == initial.baseline
    assert result.shoulders_raw == (40, 280) and result.hips_raw == (140, 280)
    assert result.raw.theta == 0
    assert result.raw.r == 2.4 and result.raw.q == .5 and result.raw.d == .15 and result.raw.g is True
    alpha = 1 - math.exp(-.2 / .06)
    S = (100 - 60 * alpha, 150 + 130 * alpha)
    H = (100 + 40 * alpha, 250 + 30 * alpha)
    assert result.shoulders_filtered == pytest.approx(S)
    assert result.hips_filtered == pytest.approx(H)
    assert result.filtered.theta == pytest.approx(torso_angle(S, H))
    assert result.filtered.d == pytest.approx((H[1] - 250) / 200)
    assert result.filtered.g is True
    assert initial.raw.g is True  # hip=250 is exactly Fy0 - .25 H0


def test_ground_below_threshold_and_degenerate_bbox_unknown():
    features = FeatureExtractor()
    acquire(features)
    result = features.update(sample(1.1, shift=(0, -1)), candidate_active=True)
    assert result.raw.g is False
    result = features.update(sample(1.2, height=0), candidate_active=True)
    assert result.raw.theta is None and result.v is None and result.m is None


def test_fixed_H0_raw_velocity_despite_smoothing_and_box_collapse():
    features = FeatureExtractor(replace(C, filter_tau_s=1.))
    acquire(features)
    for i in range(1, 6):
        dt = i * .05
        result = features.update(sample(1 + dt, shift=(0, 120 * dt), height=200 - i * 20), candidate_active=True)
    assert result.baseline.H0 == 200
    assert result.v == pytest.approx(.6)
    assert result.hips_filtered[1] < result.hips_raw[1]
    assert result.raw.q == .5


def test_velocity_window_minimum_observations_and_span():
    assert normalized_velocity([0, .2], [0, 20], 200, C) is None
    assert normalized_velocity([0, .04, .08], [0, 4, 8], 200, C) is None
    assert normalized_velocity([1000, 1000.06, 1000.12], [0, 6, 12], 200, C) == pytest.approx(.5)
    assert normalized_velocity([0, .06, .12], [12, 6, 0], 200, C) == pytest.approx(-.5)


@pytest.mark.parametrize('times,ys,H0', [([0, 0, .2], [1, 2, 3], 200), ([0, .2, .1], [1, 2, 3], 200),
                                      ([0, .1, math.nan], [1, 2, 3], 200), ([0, .1, .2], [1, 2, math.inf], 200),
                                      ([0, .1, .2], [1, 2], 200), ([0, .1, .2], [1, 2, 3], 0)])
def test_regression_rejects_unsafe_inputs(times, ys, H0):
    with pytest.raises(ValueError):
        normalized_velocity(times, ys, H0, C)


def test_stillness_common_core_actual_dt_and_one_second_drift():
    features = FeatureExtractor()
    acquire(features)
    for i in range(1, 12):
        result = features.update(sample(1 + i / 10, shift=(i * .5, 0), missing=(0, 1, 2, 3, 4, 7, 8, 9, 10)), candidate_active=True)
    assert result.m == pytest.approx(.025)
    assert result.stillness_pair_dt_s == pytest.approx(.2)
    assert result.b == pytest.approx(.025)
    assert result.drift_pair_dt_s == pytest.approx(1.)


def test_pair_chooses_nearest_target_and_uses_actual_interval():
    features = FeatureExtractor()
    acquire(features)
    features.invalidate(preserve_baseline=True)
    for t in [1.03, 1.11, 1.21, 1.28]:
        result = features.update(sample(t, shift=(10 * (t - 1.03), 0)), candidate_active=True)
    assert result.stillness_pair_dt_s == pytest.approx(.17)
    assert result.m == pytest.approx(.05)
    assert result.b is None


def test_stillness_uses_median_over_core_only():
    features = FeatureExtractor()
    acquire(features)
    value = sample(1.2)
    xy = value.keypoints_xy.copy()
    xy[5] += [20, 0]  # one moving landmark does not redefine the median
    xy[0] += [10000, 0]  # face is not part of stillness core
    result = features.update(replace(value, keypoints_xy=xy), candidate_active=True)
    assert result.m == 0


def test_pair_requires_two_common_lower_joints_not_two_different_sets():
    features = FeatureExtractor()
    acquire(features)
    features.invalidate(preserve_baseline=True)
    features.update(sample(1.1, missing=(15, 16)), candidate_active=True)
    features.update(sample(1.2, missing=(13, 14)), candidate_active=True)
    result = features.update(sample(1.3, missing=(13, 14)), candidate_active=True)
    assert result.m is None


@pytest.mark.parametrize('missing', [(5,), (11,), (13, 14, 15)])
def test_short_occlusion_resets_motion_and_one_second_history(missing):
    features = FeatureExtractor()
    acquire(features)
    assert features.update(sample(1.1), candidate_active=True).b == 0
    bad = features.update(sample(1.2, missing=missing), candidate_active=True)
    assert bad.m is None and bad.b is None and bad.history_reset
    recovered = features.update(sample(1.3), candidate_active=True)
    assert recovered.m is None and recovered.b is None


def test_intermediate_landmark_occlusion_cannot_be_bridged():
    features = FeatureExtractor()
    acquire(features)
    features.invalidate(preserve_baseline=True)
    features.update(sample(1.1, missing=(15, 16)), candidate_active=True)
    features.update(sample(1.2, missing=(13, 16)), candidate_active=True)
    result = features.update(sample(1.3, missing=(15, 16)), candidate_active=True)
    assert result.m is None


def test_long_gap_resets_filter_velocity_and_stillness_history():
    features = FeatureExtractor()
    baseline = acquire(features).baseline
    result = features.update(sample(1.251, shift=(20, 20)), candidate_active=True)
    assert result.baseline == baseline and result.history_reset
    assert result.v is None and result.m is None and result.b is None
    assert result.hips_filtered == result.hips_raw


@pytest.mark.parametrize('kwargs', [{'valid': False}, {'missing': (5,)}, {'shoulder': (100., 249.)}])
def test_invalid_pose_resets_velocity_without_one_sided_evidence(kwargs):
    features = FeatureExtractor()
    acquire(features)
    assert features.update(sample(1.1, **kwargs), candidate_active=True).v is None
    assert features.update(sample(1.2), candidate_active=True).v is None


@pytest.mark.parametrize('key', [PersonTrackKey('new', 1, 1), PersonTrackKey('session', 2, 1), PersonTrackKey('session', 1, 2)])
def test_identity_changes_discard_baseline_and_histories(key):
    features = FeatureExtractor()
    acquire(features)
    result = features.update(sample(0, key=key))
    assert result.baseline is None and result.history_reset
    assert result.v is None and result.m is None and result.b is None


@pytest.mark.parametrize('sequence,time', [(1000, 1.1), (1100, 1.), (1100, .9)])
def test_bad_order_resets_and_rejects(sequence, time):
    features = FeatureExtractor()
    acquire(features)
    with pytest.raises(ValueError, match='increasing'):
        features.update(sample(time, sequence=sequence))
    assert features.builder.baseline is None


def test_explicit_missing_observation_resets_acquisition_and_motion():
    features = FeatureExtractor()
    acquire(features)
    features.invalidate()
    result = features.update(sample(1.1))
    assert result.baseline is None and result.v is None and result.b is None


def test_summary_is_immutable_and_uses_source_time_only():
    features = FeatureExtractor()
    result = acquire(features)
    assert result.source_t_s == 1
    with pytest.raises(FrozenInstanceError):
        result.v = 1
    assert isinstance(result.hips_raw, tuple)


@pytest.mark.parametrize('dt,known', [(.149, False), (.15, True), (.30, True), (.301, False)])
def test_stillness_pair_window_boundaries(dt, known):
    features = FeatureExtractor()
    acquire(features)
    features.invalidate(preserve_baseline=True)
    features.update(sample(1.1), candidate_active=True)
    # Keep capture gaps under .25; middle frame is too recent for pairing.
    features.update(sample(1.1 + dt - .1), candidate_active=True)
    result = features.update(sample(1.1 + dt), candidate_active=True)
    assert (result.m is not None) is known
    if known:
        assert result.stillness_pair_dt_s == pytest.approx(dt)


@pytest.mark.parametrize('duration,known', [(.79, False), (.8, True), (1.2, True)])
def test_one_second_pair_minimum_and_actual_displacement(duration, known):
    features = FeatureExtractor()
    acquire(features)
    features.invalidate(preserve_baseline=True)
    # Irregular intervals prove displacement uses actual observed points.
    for i, dt in enumerate(np.linspace(0, duration, 7)):
        result = features.update(sample(float(1.1 + dt), sequence=1100 + i,
                                       shift=(10 * float(dt), 0)), candidate_active=True)
    assert (result.b is not None) is known
    if known:
        assert result.b == pytest.approx(10 * result.drift_pair_dt_s / 200)
        assert .8 - 1e-12 <= result.drift_pair_dt_s <= 1.2 + 1e-12


def test_velocity_discards_older_values():
    features = FeatureExtractor()
    acquire(features)
    for i in range(1, 9):
        t = 1 + i * .05
        # Early step must not pollute the final .25 second regression.
        result = features.update(sample(t, shift=(0, 20 if i > 1 else 0)), candidate_active=True)
    assert result.v == pytest.approx(0)


def test_exact_gap_boundary_ignores_binary_roundoff():
    filt = TimestampAwareFilter()
    filt.update((0, 0), .3)
    assert filt.update((10, 10), .55)[0] < 10


def test_features_run_under_offline_guard():
    from pathlib import Path
    from vertebrate.offline import offline_guard
    with offline_guard(Path(__file__).resolve().parents[2]) as attempts:
        features = FeatureExtractor()
        result = acquire(features)
        assert result.v == 0 and result.m == 0 and result.b == 0
    assert attempts == []
