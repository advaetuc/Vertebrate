"""Source-time exponential point smoothing; no interpolation across invalidity."""
import math

from .validation import number


class TimestampAwareFilter:
    def __init__(self, tau_s: float = .06, max_observation_gap_s: float = .25):
        for name, value in (('tau_s', tau_s), ('max_observation_gap_s', max_observation_gap_s)):
            number(value, name)
            if value == 0:
                raise ValueError(f'{name} must be positive')
        self.tau_s = tau_s
        self.max_observation_gap_s = max_observation_gap_s
        self.reset()

    def reset(self) -> None:
        self.point = None
        self.source_t_s = None

    def update(self, point, source_t_s: float, *, valid: bool = True) -> tuple[float, float] | None:
        try:
            number(source_t_s, 'source_t_s')
            if self.source_t_s is not None and source_t_s <= self.source_t_s:
                raise ValueError('Filter requires strictly increasing source time')
        except ValueError:
            self.reset()
            raise
        if not valid or point is None:
            self.reset()
            return None
        try:
            if len(point) != 2:
                raise ValueError('Expected (x, y)')
            current = tuple(float(x) for x in point)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return None
        if not all(math.isfinite(x) for x in current):
            self.reset()
            return None
        if self.point is None or source_t_s - self.source_t_s > self.max_observation_gap_s + 1e-12:
            filtered = current
        else:
            alpha = -math.expm1(-(source_t_s - self.source_t_s) / self.tau_s)
            filtered = tuple(alpha * p + (1 - alpha) * prev for p, prev in zip(current, self.point))
        self.point, self.source_t_s = filtered, source_t_s
        return filtered
