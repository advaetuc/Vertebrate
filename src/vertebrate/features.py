"""Blueprint section 7 geometry and source-time evidence, one owner per person.

FeatureExtractor.update returns immutable raw/filtered summaries. P2B must pass
candidate_active=True after candidate entry to freeze calibration, and in_zone
from the configured scene zone. Neither input implements an event FSM here.
"""
from collections import deque
from dataclasses import dataclass
import math
from statistics import median

import numpy as np

from .config import CalibrationConfig
from .contracts import PersonTrackKey, TrackSample
from .smoothing import TimestampAwareFilter
from .validation import integer, number

CORE = (5, 6, 11, 12)
LEGS = (13, 14, 15, 16)
Point = tuple[float, float]


def midpoint(sample: TrackSample, indices: tuple[int, int], confidence: float = .35) -> Point | None:
    if not sample.observation_valid or not sample.matched_detection or sample.predicted_only:
        return None
    if not all(sample.keypoint_valid_mask[i] and sample.keypoint_confidences[i] >= confidence for i in indices):
        return None
    return tuple(float(x) for x in np.mean(sample.keypoints_xy[list(indices)], axis=0))


def torso_angle(shoulders: Point, hips: Point) -> float:
    return math.degrees(math.atan2(abs(shoulders[1] - hips[1]), abs(shoulders[0] - hips[0])))


def normalized_velocity(times, hip_y, H0: float, config: CalibrationConfig) -> float | None:
    """OLS on raw hip y, centered in source time for numerical stability."""
    number(H0, 'H0')
    if H0 == 0:
        raise ValueError('H0 must be positive')
    t, y = np.asarray(times, dtype=float), np.asarray(hip_y, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('Regression requires matching finite one-dimensional observations')
    if len(t) > 1 and (np.diff(t) <= 0).any():
        raise ValueError('Regression requires strictly positive time spans')
    if len(t) < config.velocity_min_observations or t[-1] - t[0] < config.velocity_min_span_s - 1e-12:
        return None
    t = t - t[0]
    centered = t - t.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered) / H0)


@dataclass(frozen=True)
class Baseline:
    H0: float
    Hy0: float
    Fy0: float
    start_source_t_s: float
    end_source_t_s: float


class BaselineBuilder:
    """Median calibration from contiguous full-body upright samples.

Transient falling geometry and rapid downward motion suppress baseline updates;
the caller's candidate latch freezes it across the entire descent/down event.
    """
    def __init__(self, config: CalibrationConfig | None = None, frame_height: int = 720):
        integer(frame_height, 'frame_height', 1)
        self.config = config or CalibrationConfig()
        self.frame_height = frame_height
        self.reset()

    def reset(self):
        self.baseline = None
        self._window = deque()
        self._previous = None
        self._person_key = None
        self._sequence = -1

    def invalidate(self, *, preserve_baseline=False):
        self._window.clear()
        self._previous = None
        if not preserve_baseline:
            self.baseline = None

    def update(self, sample: TrackSample, *, candidate_active=False, in_zone=True) -> Baseline | None:
        c = self.config
        if self._person_key != sample.person_key:
            self.reset()
            self._person_key = sample.person_key
        if self._previous is not None and (sample.sequence <= self._sequence or sample.source_t_s <= self._previous[0]):
            self.reset()
            raise ValueError('Baseline requires increasing sequence and source time')
        previous = self._previous
        self._sequence = sample.sequence
        S, H, F = (midpoint(sample, pair, c.keypoint_confidence) for pair in ((5, 6), (11, 12), (15, 16)))
        height = float(sample.bbox_xyxy[3] - sample.bbox_xyxy[1])
        theta = torso_angle(S, H) if S is not None and H is not None else None
        self._previous = (sample.source_t_s, height, theta, H)
        if not in_zone:
            self.baseline = None
            self._window.clear()
            return None
        gap = previous is not None and sample.source_t_s - previous[0] > c.max_observation_gap_s + 1e-12
        if gap:
            self._window.clear()
            if not candidate_active:
                self.baseline = None
        if candidate_active:
            self._window.clear()
            return self.baseline
        if (S is None or H is None or F is None or height <= 0
                or (self.baseline is None and height < c.min_initial_person_height_px_at_720p * self.frame_height / 720)):
            self._window.clear()
            self.baseline = None
            return None
        falling = False
        if self.baseline is not None and previous is not None and previous[2] is not None and not gap:
            dt = sample.source_t_s - previous[0]
            falling = (height < previous[1] - 1e-9 or theta < previous[2] - 1e-9
                       or (H[1] - previous[3][1]) / (dt * self.baseline.H0) >= c.rapid_descent_v_h0_per_s)
        if theta < c.upright_min_theta_deg or falling:
            self._window.clear()
            return self.baseline
        # Retain the sample just before the boundary as well, so irregular
        # cadence can complete an interval without inventing an interpolant.
        while len(self._window) > 1 and sample.source_t_s - self._window[1][0] >= c.upright_acquisition_s - 1e-12:
            self._window.popleft()
        entry = (sample.source_t_s, height, H, F[1], math.dist(S, H))
        proposed = [*self._window, entry]
        H0 = median(e[1] for e in proposed)
        drift = max(math.dist(a[2], b[2]) for a in proposed for b in proposed)
        stable = (drift <= c.baseline_max_hip_drift_h0 * H0 + 1e-9
                  and all(e[4] >= c.min_torso_length_h0 * H0 for e in proposed))
        changed = (self.baseline is not None
                   and abs(height / self.baseline.H0 - 1) > c.baseline_height_reset_fraction + 1e-12)
        if not stable or changed:
            self.baseline = None
            self._window.clear()
            if entry[4] >= c.min_torso_length_h0 * height:
                self._window.append(entry)
            return None
        if self.baseline is not None and height / self.baseline.H0 < c.upright_min_height_ratio_q:
            self._window.clear()
            return self.baseline
        self._window.append(entry)
        if sample.source_t_s - self._window[0][0] >= c.upright_acquisition_s - 1e-12:
            self.baseline = Baseline(float(H0), float(median(e[2][1] for e in proposed)),
                                     float(median(e[3] for e in proposed)),
                                     self._window[0][0], sample.source_t_s)
        return self.baseline


@dataclass(frozen=True)
class Kinematics:
    theta: float | None = None
    r: float | None = None
    q: float | None = None
    d: float | None = None
    g: bool | None = None


@dataclass(frozen=True)
class FeatureSummary:
    person_key: PersonTrackKey
    sequence: int
    source_t_s: float
    baseline: Baseline | None
    shoulders_raw: Point | None
    hips_raw: Point | None
    shoulders_filtered: Point | None
    hips_filtered: Point | None
    raw: Kinematics
    filtered: Kinematics
    v: float | None
    m: float | None
    b: float | None
    stillness_pair_dt_s: float | None
    drift_pair_dt_s: float | None
    history_reset: bool


def _geometry(sample, S, H, baseline, config):
    w, h = sample.bbox_xyxy[2:] - sample.bbox_xyxy[:2]
    if S is None or H is None or w <= 0 or h <= 0:
        return Kinematics()
    if baseline is not None and math.dist(S, H) < config.min_torso_length_h0 * baseline.H0:
        return Kinematics()
    return Kinematics(torso_angle(S, H), float(w / h),
                      float(h / baseline.H0) if baseline else None,
                      (H[1] - baseline.Hy0) / baseline.H0 if baseline else None,
                      max(S[1], H[1]) >= baseline.Fy0 - config.ground_margin_h0 * baseline.H0 if baseline else None)


class FeatureExtractor:
    """One per person. Changed identity resets all retained evidence.

Missing/invalid observations must be submitted (observation_valid=False), or
call invalidate() when no TrackSample exists. No absent frame is interpolated.
    """
    def __init__(self, config: CalibrationConfig | None = None, frame_height: int = 720):
        self.config = config or CalibrationConfig()
        self.builder = BaselineBuilder(self.config, frame_height)
        self._S = TimestampAwareFilter(self.config.filter_tau_s, self.config.max_observation_gap_s)
        self._H = TimestampAwareFilter(self.config.filter_tau_s, self.config.max_observation_gap_s)
        self.reset()

    def reset(self):
        self.builder.reset()
        self.person_key = None
        self._last = None
        self._clear_history()

    def _clear_history(self):
        self._S.reset()
        self._H.reset()
        self._velocity = deque()
        self._motion = deque()

    def invalidate(self, *, preserve_baseline=False):
        self._clear_history()
        self.builder.invalidate(preserve_baseline=preserve_baseline)

    def update(self, sample: TrackSample, *, candidate_active=False, in_zone=True) -> FeatureSummary:
        c = self.config
        reset = self.person_key != sample.person_key
        if reset:
            self.reset()
            self.person_key = sample.person_key
        if self._last is not None:
            if sample.sequence <= self._last[0] or sample.source_t_s <= self._last[1]:
                self.reset()
                raise ValueError('Features require increasing sequence and source time')
            if sample.source_t_s - self._last[1] > c.max_observation_gap_s + 1e-12:
                self._clear_history()
                reset = True
        self._last = (sample.sequence, sample.source_t_s)
        baseline = self.builder.update(sample, candidate_active=candidate_active, in_zone=in_zone)
        S, H = (midpoint(sample, pair, c.keypoint_confidence) for pair in ((5, 6), (11, 12)))
        w, h = sample.bbox_xyxy[2:] - sample.bbox_xyxy[:2]
        reference = baseline.H0 if baseline else h
        valid = (in_zone and S is not None and H is not None and w > 0 and h > 0
                 and math.dist(S, H) >= c.min_torso_length_h0 * reference)
        if not valid:
            self._clear_history()
            return FeatureSummary(sample.person_key, sample.sequence, sample.source_t_s, baseline,
                                  S, H, None, None, Kinematics(), Kinematics(), None, None, None, None, None, True)
        Sf, Hf = self._S.update(S, sample.source_t_s), self._H.update(H, sample.source_t_s)
        self._velocity.append((sample.source_t_s, H[1]))
        while self._velocity and sample.source_t_s - self._velocity[0][0] > c.velocity_window_s + 1e-12:
            self._velocity.popleft()
        mask = sample.keypoint_valid_mask & (sample.keypoint_confidences >= c.keypoint_confidence)
        if not mask[list(CORE)].all() or mask[list(LEGS)].sum() < 2:
            self._motion.clear()
            reset = True
        else:
            self._motion.append((sample, mask))
        horizon = max(c.one_second_max_dt_s, c.stillness_pair_max_dt_s)
        while self._motion and sample.source_t_s - self._motion[0][0].source_t_s > horizon + 1e-12:
            self._motion.popleft()
        v = normalized_velocity(*zip(*self._velocity), baseline.H0, c) if baseline else None
        m = b = pair_dt = drift_dt = None
        if baseline:
            pairs = []
            common = mask.copy()
            for previous, previous_mask in reversed(self._motion):
                common &= previous_mask  # never bridge a landmark's temporary occlusion
                dt = sample.source_t_s - previous.source_t_s
                if (c.stillness_pair_min_dt_s - 1e-12 <= dt <= c.stillness_pair_max_dt_s + 1e-12
                        and common[list(CORE)].all() and common[list(LEGS)].sum() >= 2):
                    indices = [*CORE, *(i for i in LEGS if common[i])]
                    speed = float(np.median(np.linalg.norm(sample.keypoints_xy[indices] - previous.keypoints_xy[indices], axis=1)) / (baseline.H0 * dt))
                    pairs.append((abs(dt - c.stillness_pair_target_dt_s), dt, speed))
            if pairs:
                _, pair_dt, m = min(pairs)
            prior = [(abs(sample.source_t_s - p.source_t_s - 1), p) for p, _ in self._motion
                     if c.one_second_min_dt_s - 1e-12 <= sample.source_t_s - p.source_t_s <= c.one_second_max_dt_s + 1e-12]
            if prior:
                _, p = min(prior, key=lambda item: (item[0], -item[1].source_t_s))
                b = math.dist(H, midpoint(p, (11, 12), c.keypoint_confidence)) / baseline.H0
                drift_dt = sample.source_t_s - p.source_t_s
        return FeatureSummary(sample.person_key, sample.sequence, sample.source_t_s, baseline, S, H, Sf, Hf,
                              _geometry(sample, S, H, baseline, c), _geometry(sample, Sf, Hf, baseline, c),
                              v, m, b, pair_dt, drift_dt, reset)
