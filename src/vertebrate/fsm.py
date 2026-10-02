"""Deterministic section-8 FSM. All times are source times; no wall clock."""
from dataclasses import dataclass
from enum import StrEnum
import math
from uuid import NAMESPACE_URL, uuid5

from .config import CalibrationConfig
from .contracts import PersonTrackKey
from .features import FeatureSummary
from .validation import integer, number

EPS = 1e-12


class State(StrEnum):
    ACQUIRING = 'ACQUIRING'
    MONITORING = 'MONITORING'
    DESCENT = 'DESCENT'
    VERIFYING_DOWN = 'VERIFYING_DOWN'
    STILLNESS = 'STILLNESS'
    ALERTED = 'ALERTED'


@dataclass(frozen=True)
class CandidateAlert:
    incident_id: str
    person_key: PersonTrackKey
    onset_source_t_s: float
    down_source_t_s: float
    confirmed_source_t_s: float
    sequence: int
    peak_v_h0_per_s: float


@dataclass(frozen=True)
class Decision:
    person_key: PersonTrackKey
    sequence: int | None
    source_t_s: float | None
    state: State
    reason: str
    alert: CandidateAlert | None = None
    inconclusive: bool = False
    reset_baseline: bool = False


class PersonFSM:
    def __init__(self, person_key: PersonTrackKey, config: CalibrationConfig | None = None):
        self.person_key = person_key
        self.config = config or CalibrationConfig()
        self.state = State.ACQUIRING
        self.incident_id = None
        self._event_number = 0
        self._last_t = self._last_valid_t = self._last_sequence = None
        self.closed = False
        self._clear_candidate()

    @property
    def candidate_active(self):
        return self.state in (State.DESCENT, State.VERIFYING_DOWN, State.STILLNESS)

    @property
    def freeze_baseline(self):
        return self.candidate_active or self.state == State.ALERTED

    def _clear_candidate(self):
        self._rapid_start = None
        self._rapid_count = 0
        self._onset = self._descent_entry = None
        self._low_start = self._down_time = self._down_entry = None
        self._still_start = None
        self._peak_v = 0.
        self._recovery = []
        self._motion_after = -1.

    def _decision(self, reason, *, alert=None, inconclusive=False, reset_baseline=False):
        return Decision(self.person_key, self._last_sequence, self._last_t, self.state,
                        reason, alert, inconclusive, reset_baseline)

    def _acquire(self, reason, *, inconclusive=False):
        self.state = State.ACQUIRING
        self._clear_candidate()
        return self._decision(reason, inconclusive=inconclusive, reset_baseline=True)

    def close(self, reason='end_of_stream') -> Decision:
        """Close at the last observed source time, without adding EOF dwell."""
        was_candidate = self.candidate_active
        self.closed = True
        if self.state != State.ALERTED:
            self.state = State.ACQUIRING
        self._clear_candidate()
        return self._decision(reason, inconclusive=was_candidate, reset_baseline=True)

    def missing(self, sequence: int, source_t_s: float) -> Decision:
        return self._step(None, sequence, source_t_s, False)

    def update(self, features: FeatureSummary, *, recovery_valid=True) -> Decision:
        if features.person_key != self.person_key:
            raise ValueError('Identity mismatch: a new generation requires a fresh FSM and baseline')
        return self._step(features, features.sequence, features.source_t_s, recovery_valid)

    def _step(self, f, sequence, t, recovery_valid):
        if self.closed:
            raise ValueError('Source closed; create a new session FSM')
        integer(sequence, 'sequence')
        number(t, 'source_t_s')
        if self._last_t is not None and (t <= self._last_t or sequence <= self._last_sequence):
            self.close('invalid_timestamp')
            raise ValueError('FSM requires increasing source time and sequence')
        gap = self._last_t is not None and t - self._last_t > self.config.max_observation_gap_s + EPS
        self._last_t, self._last_sequence = t, sequence
        c = self.config
        k = f.filtered if f else None
        valid = (f is not None and f.baseline is not None and f.hips_raw is not None and k.g is not None
                 and all(v is not None and math.isfinite(v) for v in (k.theta, k.q, k.r, k.d)))
        if gap and self.state != State.ALERTED:
            return self._acquire('observation_gap', inconclusive=self.candidate_active)
        if not valid:
            self._recovery.clear()
            self._rapid_start, self._rapid_count = None, 0
            self._low_start = self._still_start = None
            self._motion_after = t + c.one_second_min_dt_s
            if self.state == State.ALERTED:
                return self._decision('alert_latched_visibility_lost')
            if self.candidate_active:
                if self._last_valid_t is None or t - self._last_valid_t > c.max_observation_gap_s + EPS:
                    return self._acquire('observation_gap', inconclusive=True)
                if self.state == State.STILLNESS:
                    self.state = State.VERIFYING_DOWN
                expired = self._deadline(t)
                return expired or self._decision('invalid_observation')
            down = f is not None and f.raw.theta is not None and f.raw.theta <= c.low_torso_max_theta_deg
            was_monitoring = self.state == State.MONITORING
            self.state = State.ACQUIRING
            return self._decision('already_down' if down else 'insufficient_evidence', reset_baseline=was_monitoring)
        # A stream of invalid frames must not hide a long observation loss.
        lost = self._last_valid_t is not None and t - self._last_valid_t > c.max_observation_gap_s + EPS
        self._last_valid_t = t
        if lost and self.candidate_active:
            return self._acquire('observation_gap', inconclusive=True)
        if gap or f.history_reset:
            self._recovery.clear()
            self._still_start = None
            self._motion_after = t + c.one_second_min_dt_s
            if self.state == State.STILLNESS:
                self.state = State.VERIFYING_DOWN
        upright = k.theta >= c.upright_min_theta_deg and k.q >= c.upright_min_height_ratio_q
        # Recovery precedes deadline, posture and duration completion.
        if self.freeze_baseline and upright and recovery_valid and not f.history_reset:
            self._recovery.append((t, f.hips_raw))
            drift = max(math.dist(a[1], b[1]) for a in self._recovery for b in self._recovery)
            if drift > c.baseline_max_hip_drift_h0 * f.baseline.H0:
                self._recovery = [(t, f.hips_raw)]
            if t - self._recovery[0][0] >= c.recovery_dwell_s - EPS:
                return self._acquire('recovered')
        else:
            self._recovery.clear()
        if self.state == State.ALERTED:
            return self._decision('alert_latched')
        expired = self._deadline(t)
        if expired:
            return expired
        if self.state == State.ACQUIRING:
            if upright:
                self.state = State.MONITORING
                return self._decision('baseline_acquired')
            return self._decision('already_down' if k.theta <= c.low_torso_max_theta_deg else 'acquiring')
        if self.state == State.MONITORING:
            rapid = f.v is not None and math.isfinite(f.v) and f.v >= c.rapid_descent_v_h0_per_s
            if rapid and not f.history_reset:
                if self._rapid_start is None:
                    self._rapid_start = t
                    self._peak_v = 0.
                self._peak_v = max(self._peak_v, f.v)
                self._rapid_count += 1
                if (self._rapid_count >= c.rapid_descent_min_estimates
                        and t - self._rapid_start >= c.rapid_descent_min_duration_s - EPS
                        and f.raw.d is not None and f.raw.d >= c.initial_downward_displacement_d):
                    self.state = State.DESCENT
                    self._onset, self._descent_entry = self._rapid_start, t
                    return self._decision('descent_observed')
            else:
                self._rapid_start, self._rapid_count = None, 0
            return self._decision('monitoring')
        if f.v is not None and math.isfinite(f.v):
            self._peak_v = max(self._peak_v, f.v)
        low = (k.d >= c.completed_descent_d and k.theta <= c.low_torso_max_theta_deg
               and (k.r >= c.shape_min_width_height_ratio_r or k.q <= c.shape_max_height_ratio_q) and k.g)
        motion = (not f.history_reset and t >= self._motion_after - EPS
                  and f.m is not None and math.isfinite(f.m) and 0 <= f.m <= c.stillness_max_speed_m_h0_per_s
                  and f.b is not None and math.isfinite(f.b) and 0 <= f.b <= c.one_second_max_displacement_b_h0)
        if self.state == State.DESCENT:
            self._low_start = (t if self._low_start is None else self._low_start) if low else None
            if self._low_start is not None and t - self._low_start >= c.down_state_dwell_s - EPS:
                self.state = State.VERIFYING_DOWN
                if self._down_entry is None:
                    self._down_time, self._down_entry = self._low_start, t
                return self._decision('down_posture_observed')
            return self._decision('descent')
        if not low:
            self._still_start = self._low_start = None
            if t - self._descent_entry >= c.descent_deadline_s - EPS:
                return self._acquire('low_posture_lost_after_deadline', inconclusive=True)
            self.state = State.DESCENT
            return self._decision('low_posture_lost')
        if not motion:
            self.state = State.VERIFYING_DOWN
            self._still_start = None
            return self._decision('motion_or_incomplete_history')
        if self.state == State.VERIFYING_DOWN:
            self.state = State.STILLNESS
            self._still_start = t  # no retroactive credit from acquisition/down dwell
            return self._decision('stillness_started')
        if t - self._still_start >= c.stillness_duration_s - EPS:
            self.state = State.ALERTED
            self._event_number += 1
            self.incident_id = str(uuid5(NAMESPACE_URL, repr((self.person_key, self._event_number, self._onset))))
            alert = CandidateAlert(self.incident_id, self.person_key, self._onset, self._down_time, t, sequence, self._peak_v)
            return self._decision('suspected_fall_with_sustained_stillness', alert=alert)
        return self._decision('stillness')

    def _deadline(self, t):
        if not self.candidate_active:
            return None
        if self._down_entry is not None and t - self._down_entry >= self.config.max_down_verification_s - EPS:
            return self._acquire('down_deadline', inconclusive=True)
        if self.state == State.DESCENT and t - self._descent_entry >= self.config.descent_deadline_s - EPS:
            return self._acquire('descent_deadline', inconclusive=True)
        return None
