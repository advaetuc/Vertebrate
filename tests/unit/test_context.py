from dataclasses import FrozenInstanceError
import math

import pytest

from vertebrate.context import Environment, build_environment, heat_index


@pytest.mark.parametrize('fahrenheit,rh,expected', [(90., 70., 105.9220206),
                                                  (100., 10., 94.122482662), (80., 90., 86.3418917)])
def test_nws_reference_regressions(fahrenheit, rh, expected):
    result = heat_index((fahrenheit - 32) / 1.8, rh)
    assert result.heat_index_c * 1.8 + 32 == pytest.approx(expected, abs=.001)
    assert result.heat_index_method == 'nws_rothfusz_adjusted'


def test_lower_branch_is_averaged_not_unaveraged_simple():
    result = heat_index(20, 50)
    T = 68
    simple = .5 * (T + 61 + 1.2 * (T - 68) + .094 * 50)
    assert result.heat_index_c == ((simple + T) / 2 - 32) / 1.8
    assert result.heat_index_method == 'nws_simple_averaged'


@pytest.mark.parametrize('temp,rh', [(None, 50), (30, None), (math.nan, 50), (30, math.inf),
                                    (True, 50), ('30', 50), (30, -1), (30, 101), (10**1000, 50)])
def test_invalid_or_missing_context_never_produces_heat_index(temp, rh):
    result = heat_index(temp, rh)
    assert result.heat_index_c is None and result.reason
    env = build_environment(temp, rh, source='manual_measured')
    assert env.heat_index_c is None and env.heat_index_reason


@pytest.mark.parametrize('temp,rh', [(19.99, 50), (40.01, 50), (30, 9.9), (30, 90.1), (-10, 99)])
def test_outside_envelope_retains_finite_inputs_with_reason(temp, rh):
    result = build_environment(temp, rh, source='manual_simulated')
    assert result.ambient_temp_c == temp and result.relative_humidity_pct == rh
    assert result.heat_index_method == 'out_of_demo_range' and result.heat_index_c is None


@pytest.mark.parametrize('temp,rh', [(20, 10), (40, 90), (20, 90), (40, 10)])
def test_demo_envelope_inclusive_boundaries(temp, rh):
    assert heat_index(temp, rh).heat_index_c is not None


@pytest.mark.parametrize('now,stale', [('2026-10-02T12:29:59Z', False), ('2026-10-02T12:30:00Z', True),
                                    ('2026-10-02T13:00:00Z', True)])
def test_thirty_minute_staleness_retains_measurements(now, stale):
    env = build_environment(30., 50., source='manual_measured', observed_at_utc='2026-10-02T12:00:00Z', now_utc=now)
    assert env.stale is stale and env.heat_index_c is not None
    assert env.observed_at_utc == '2026-10-02T12:00:00Z'


def test_configurable_staleness_and_unknown_time():
    assert build_environment(30, 50, source='manual_measured').stale is None
    assert build_environment(30, 50, source='manual_simulated', observed_at_utc='2026-10-02T12:00:00Z',
                             now_utc='2026-10-02T12:10:00Z', staleness_minutes=10).stale is True
    assert Environment().source == 'missing'
    assert build_environment().heat_index_reason == 'missing_context'


@pytest.mark.parametrize('kwargs', [{'source': 'weather_api'}, {'applies_to': 'unknown'},
                                  {'staleness_minutes': 0}, {'staleness_minutes': math.nan},
                                  {'observed_at_utc': 'bad'},
                                  {'observed_at_utc': '2026-10-02T12:01:00Z', 'now_utc': '2026-10-02T12:00:00Z'}])
def test_invalid_metadata_rejected(kwargs):
    with pytest.raises(ValueError):
        build_environment(30, 50, **({'source': 'manual_measured'} | kwargs))


def test_frozen_and_no_internal_rounding():
    env = build_environment((90 - 32) / 1.8, 70, source='manual_simulated')
    assert env.heat_index_c != round(env.heat_index_c, 2)
    with pytest.raises(FrozenInstanceError):
        env.stale = True
