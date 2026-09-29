"""Source time is the sole evidence clock. UTC is separate provenance."""
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import time
import uuid

from .contracts import CaptureError, SourceTimingError, TimingMode
from .validation import integer, number, text, utc_datetime, utc_string


class LiveSourceClock:
    def __init__(self, monotonic_ns_fn: Callable[[], int] = time.monotonic_ns,
                 utc_now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.monotonic_ns_fn = monotonic_ns_fn
        self.utc_now_fn = utc_now_fn
        self.session_id = None

    def start_session(self, session_id: str | None = None) -> str:
        session_id = str(uuid.uuid4()) if session_id is None else session_id
        text(session_id, 'session_id')
        stamp = self.monotonic_ns_fn()
        integer(stamp, 'monotonic_ns', 1)
        self.session_start_utc = utc_string(self.utc_now_fn())
        self.session_start_monotonic_ns = stamp
        self.session_id = session_id
        self.sequence = 0
        self.last_source_t_s = 0.0
        self._last_ns = stamp
        return session_id

    def tick(self) -> tuple[int, float, int]:
        if self.session_id is None:
            raise CaptureError('Start the live clock session before tick')
        stamp = self.monotonic_ns_fn()
        integer(stamp, 'monotonic_ns', 1)
        if stamp < self._last_ns:
            raise SourceTimingError('Live monotonic clock moved backwards')
        source_t_s = (stamp - self.session_start_monotonic_ns) / 1e9
        result = self.sequence, source_t_s, stamp
        self.sequence += 1
        self.last_source_t_s = source_t_s
        self._last_ns = stamp
        return result

    def estimate_event_utc(self, source_t_s: float) -> str:
        if self.session_id is None:
            raise CaptureError('Start the live clock session before UTC estimation')
        number(source_t_s, 'source_t_s')
        return utc_string(utc_datetime(self.session_start_utc) + timedelta(seconds=source_t_s))


class VideoFileClock:
    def __init__(self, verified_fps: float, recorded_start_utc: str | None = None,
                 cfr_tolerance_s: float | None = None):
        try:
            number(verified_fps, 'verified_fps', maximum=240)
            if verified_fps == 0:
                raise ValueError('verified_fps must be > 0')
            tolerance = max(0.5 / verified_fps, 0.005) if cfr_tolerance_s is None else cfr_tolerance_s
            number(tolerance, 'cfr_tolerance_s')
            if tolerance == 0:
                raise ValueError('cfr_tolerance_s must be > 0')
            if recorded_start_utc is not None:
                utc_datetime(recorded_start_utc)
        except (ValueError, TypeError) as exc:
            raise SourceTimingError(str(exc)) from exc
        self.verified_fps = verified_fps
        self.recorded_start_utc = recorded_start_utc
        self.cfr_tolerance_s = tolerance
        self.reset()

    def reset(self, first_frame_index: int = 0):
        integer(first_frame_index, 'first_frame_index')
        self._next_index = first_frame_index
        self.last_source_t_s = None
        self._last_pts = None
        self._fallback = False
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def compute_frame_time(self, frame_index: int, pts_seconds: float | None) -> tuple[float, TimingMode]:
        if self.paused:
            raise CaptureError('Video clock is paused; frame advancement is blocked')
        try:
            integer(frame_index, 'frame_index')
            if frame_index != self._next_index:
                raise ValueError(f'Expected consecutive frame index {self._next_index}, got {frame_index}; reset on seek')
            if pts_seconds is not None:
                number(pts_seconds, 'PTS')
            expected = frame_index / self.verified_fps
            missing = pts_seconds is None or (frame_index > 0 and pts_seconds == 0)
            if not missing:
                if abs(pts_seconds - expected) > self.cfr_tolerance_s + 1e-12:
                    raise ValueError(f'Non-CFR PTS {pts_seconds} at frame {frame_index}; expected {expected}')
                if frame_index > 0 and self.last_source_t_s is not None:
                    if pts_seconds <= self.last_source_t_s:
                        raise ValueError('PTS must be strictly increasing; frozen/backward timestamp')
                    if abs(pts_seconds - self.last_source_t_s - 1 / self.verified_fps) > self.cfr_tolerance_s + 1e-12:
                        raise ValueError('VFR frame-to-frame PTS jitter exceeds CFR tolerance')
            fallback = self._fallback or missing
            value = expected if fallback or frame_index == 0 else float(pts_seconds)
            mode = TimingMode.FRAME_INDEX_FPS_FALLBACK if fallback else TimingMode.PTS_VERIFIED
        except (ValueError, TypeError) as exc:
            raise SourceTimingError(str(exc)) from exc
        self._fallback = fallback
        self.last_source_t_s = value
        self._last_pts = pts_seconds
        self._next_index += 1
        return value, mode

    def estimate_event_utc(self, source_t_s: float) -> str | None:
        number(source_t_s, 'source_t_s')
        if self.recorded_start_utc is None:
            return None
        return utc_string(utc_datetime(self.recorded_start_utc) + timedelta(seconds=source_t_s))
