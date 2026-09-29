"""Small P1 validation primitives; P0 configuration remains independent."""
from datetime import datetime, timezone
import math


def text(value, name):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{name} must be a nonempty string without NUL')


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def number(value, name, minimum=0, maximum=None):
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be a finite number')
    try:
        valid = math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite, >= {minimum}' + (f' and <= {maximum}' if maximum is not None else ''))


def utc_datetime(value: str) -> datetime:
    text(value, 'UTC timestamp')
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
            raise ValueError('timezone must be UTC')
        return dt.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f'Invalid ISO-8601 UTC timestamp {value!r}: {exc}') from exc


def utc_string(dt: datetime) -> str:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('UTC clock must return a timezone-aware datetime')
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
