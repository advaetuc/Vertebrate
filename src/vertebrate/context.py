"""Optional manual environment context; never an input to fall detection."""
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import math

from .validation import number, text, utc_datetime, utc_string

SOURCES = ('manual_measured', 'manual_simulated', 'missing')
METHODS = ('nws_simple_averaged', 'nws_rothfusz_adjusted', 'out_of_demo_range', None)
APPLIES_TO = ('demo_scenario', 'current_scene', 'recorded_scene')


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class HeatIndex:
    heat_index_c: float | None
    heat_index_method: str | None
    reason: str | None = None


def heat_index(temp_c, relative_humidity_pct) -> HeatIndex:
    """Blueprint §9 NWS calculation, full precision and explicit null reasons."""
    if temp_c is None or relative_humidity_pct is None:
        return HeatIndex(None, None, 'missing_temperature_or_humidity')
    if not finite(temp_c) or not finite(relative_humidity_pct):
        return HeatIndex(None, None, 'invalid_temperature_or_humidity')
    if not 0 <= relative_humidity_pct <= 100:
        return HeatIndex(None, None, 'invalid_relative_humidity')
    if not 20 <= temp_c <= 40 or not 10 <= relative_humidity_pct <= 90:
        return HeatIndex(None, 'out_of_demo_range', 'outside_20_40c_10_90rh')
    T, R = 1.8 * temp_c + 32, relative_humidity_pct
    simple = .5 * (T + 61 + 1.2 * (T - 68) + .094 * R)
    screen = (simple + T) / 2
    if screen < 80:
        return HeatIndex((screen - 32) / 1.8, 'nws_simple_averaged')
    value = (-42.379 + 2.04901523*T + 10.14333127*R - .22475541*T*R
             - .00683783*T*T - .05481717*R*R + .00122874*T*T*R
             + .00085282*T*R*R - .00000199*T*T*R*R)
    if R < 13 and 80 <= T <= 112:
        value -= ((13 - R) / 4) * math.sqrt((17 - abs(T - 95)) / 17)
    if R > 85 and 80 <= T <= 87:
        value += ((R - 85) / 10) * ((87 - T) / 5)
    return HeatIndex((value - 32) / 1.8, 'nws_rothfusz_adjusted')


@dataclass(frozen=True)
class Environment:
    ambient_temp_c: float | None = None
    relative_humidity_pct: float | None = None
    heat_index_c: float | None = None
    heat_index_method: str | None = None
    source: str = 'missing'
    applies_to: str = 'demo_scenario'
    observed_at_utc: str | None = None
    stale: bool | None = None
    # Optional reason extensions accompany unknown/failed values in schema 1.0.
    heat_index_reason: str | None = field(default=None, metadata={'omit_none': True})

    def __post_init__(self):
        if self.source not in SOURCES or self.applies_to not in APPLIES_TO or self.heat_index_method not in METHODS:
            raise ValueError('Invalid environment source, applies_to or heat_index_method')
        for name in ('ambient_temp_c', 'relative_humidity_pct', 'heat_index_c'):
            if getattr(self, name) is not None and not finite(getattr(self, name)):
                raise ValueError(f'{name} must be finite or null')
        if self.relative_humidity_pct is not None and not 0 <= self.relative_humidity_pct <= 100:
            raise ValueError('relative_humidity_pct must be in [0, 100] or null')
        if self.observed_at_utc is not None:
            utc_datetime(self.observed_at_utc)
        if self.stale is not None and type(self.stale) is not bool:
            raise ValueError('stale must be boolean or null')
        if (self.observed_at_utc is None) != (self.stale is None):
            raise ValueError('stale is unknown exactly when observation time is unknown')
        if self.heat_index_c is None:
            if self.heat_index_reason is None:
                object.__setattr__(self, 'heat_index_reason', 'missing_context')
            text(self.heat_index_reason, 'heat_index_reason')
            if self.heat_index_method not in (None, 'out_of_demo_range'):
                raise ValueError('Missing heat index cannot claim a successful NWS calculation')
        elif (self.heat_index_method not in METHODS[:2] or self.heat_index_reason is not None
              or self.ambient_temp_c is None or self.relative_humidity_pct is None):
            raise ValueError('Computed heat index requires measurements, NWS method and no error')
        if self.source == 'missing' and any(v is not None for v in
                (self.ambient_temp_c, self.relative_humidity_pct, self.heat_index_c, self.observed_at_utc)):
            raise ValueError('Missing context cannot contain measured values')


def build_environment(temp_c=None, relative_humidity_pct=None, *, source='missing',
                      applies_to='demo_scenario', observed_at_utc=None, now_utc=None,
                      staleness_minutes=30.) -> Environment:
    number(staleness_minutes, 'staleness_minutes')
    if staleness_minutes == 0:
        raise ValueError('staleness_minutes must be positive')
    if source == 'missing':
        return Environment(applies_to=applies_to)
    result = heat_index(temp_c, relative_humidity_pct)
    # JSON cannot retain NaN/Inf. Invalid entries become null with the reason.
    temp = temp_c if finite(temp_c) else None
    rh = relative_humidity_pct if finite(relative_humidity_pct) and 0 <= relative_humidity_pct <= 100 else None
    stale = None
    if observed_at_utc is not None:
        observed = utc_datetime(observed_at_utc)
        now = utc_datetime(now_utc) if now_utc is not None else datetime.now(timezone.utc)
        if observed > now:
            raise ValueError('Context observation time cannot be in the future')
        stale = (now - observed).total_seconds() >= staleness_minutes * 60
    return Environment(temp, rh, result.heat_index_c, result.heat_index_method, source,
                       applies_to, observed_at_utc, stale, result.reason)


def environment_at(environment: Environment, confirmed_at_utc: str, staleness_minutes=30.) -> Environment:
    """Freeze freshness at confirmation, not when an operator opened the editor."""
    number(staleness_minutes, 'staleness_minutes')
    if staleness_minutes == 0:
        raise ValueError('staleness_minutes must be positive')
    confirmed = utc_datetime(confirmed_at_utc)
    if environment.observed_at_utc is None:
        return environment
    age = (confirmed - utc_datetime(environment.observed_at_utc)).total_seconds()
    if age < 0:
        raise ValueError('Context observation cannot follow confirmation')
    return replace(environment, stale=age >= staleness_minutes * 60)
