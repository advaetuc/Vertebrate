"""Optional playback media only. Source-time sampling never changes detection.

One vision owner feeds the recorder. Immutable pixel allocations are charged to
a thread-safe budget for their entire lifetime, including queued writer payloads.
Two incidents share pixels rather than copying the ring. No continuous disk video.
"""
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

import cv2
import numpy as np

from .contracts import FramePacket
from .validation import integer, number, text

WIDTH, HEIGHT, FPS, RETENTION_S = 320, 180, 5, 30.
FRAME_BYTES = WIDTH * HEIGHT * 3
MAX_MEDIA_BYTES = 134217728


class MediaBudget:
    def __init__(self, limit=MAX_MEDIA_BYTES):
        integer(limit, 'media budget', 1)
        self.limit = min(limit, MAX_MEDIA_BYTES)
        self._used = 0
        self._lock = Lock()

    @property
    def used_bytes(self):
        with self._lock:
            return self._used

    def reserve(self, size):
        integer(size, 'allocation size')
        with self._lock:
            if self._used + size > self.limit:
                raise MemoryError('Optional media buffer limit exceeded')
            self._used += size

    def release(self, size):
        with self._lock:
            self._used -= size
            assert self._used >= 0


class _Pixels(bytes):
    """The allocation, not an array view, owns the reservation."""
    def __new__(cls, array, budget):
        obj = super().__new__(cls, memoryview(array))
        obj.budget = budget
        return obj

    def __del__(self):
        self.budget.release(len(self))


@dataclass(frozen=True)
class ClipFrame:
    source_t_s: float
    sequence: int
    frame_bgr: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self):
        number(self.source_t_s, 'source_t_s')
        integer(self.sequence, 'sequence')
        array = self.frame_bgr
        if (not isinstance(array, np.ndarray) or array.shape != (HEIGHT, WIDTH, 3)
                or array.dtype != np.uint8 or array.flags.writeable):
            raise ValueError('Clip frame must be immutable 320x180 uint8 BGR')
        while isinstance(array, np.ndarray):
            array = array.base
        if not isinstance(array, _Pixels):
            raise ValueError('Clip pixels must own a media budget reservation')


@dataclass(frozen=True)
class ClipPayload:
    incident_id: str
    session_id: str
    requested_start_s: float
    requested_end_s: float
    frames: tuple[ClipFrame, ...]
    truncated: bool
    reasons: tuple[str, ...] = ()
    source_time_offset_s: float = 0.

    def __post_init__(self):
        text(self.incident_id, 'incident_id')
        text(self.session_id, 'session_id')
        number(self.requested_start_s, 'requested_start_s', -3.)
        number(self.requested_end_s, 'requested_end_s', max(0., self.requested_start_s))
        number(self.source_time_offset_s, 'source_time_offset_s')
        if type(self.truncated) is not bool or type(self.frames) is not tuple or type(self.reasons) is not tuple:
            raise ValueError('Clip payload must use immutable tuples and boolean truncation')
        for reason in self.reasons:
            text(reason, 'truncation reason')
        if self.reasons and not self.truncated:
            raise ValueError('Truncation reasons require truncated=true')
        for frame in self.frames:
            if not isinstance(frame, ClipFrame):
                raise ValueError('Expected immutable clip frames')
            if not self.requested_start_s <= frame.source_t_s <= self.requested_end_s + 1e-9:
                raise ValueError('Frame outside requested window')
        if any(b.source_t_s <= a.source_t_s or b.sequence <= a.sequence
               for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError('Clip frames must follow source order')

    def metadata(self):
        return {'schema_version': '1.0', 'session_id': self.session_id,
                'requested_start_s': self.requested_start_s,
                'requested_end_s': self.requested_end_s,
                'time_basis': 'session_source_time',
                'source_time_offset_s': self.source_time_offset_s,
                'playback_fps': FPS, 'truncated': self.truncated,
                'reasons': list(self.reasons),
                'frames': [{'sequence': f.sequence, 'source_t_s': f.source_t_s}
                           for f in self.frames]}


class ClipRecorder:
    def __init__(self, budget=None, *, max_pending=8):
        self.budget = budget if budget is not None else MediaBudget()
        integer(max_pending, 'max_pending', 1)
        self.max_pending = max_pending
        self._ring = deque()
        self._capacity_gaps = deque(maxlen=150)
        self._pending = {}
        self._session = None
        self._last_input = self._last_sample = None
        self._offset = 0.

    @property
    def buffered_frames(self):
        return tuple(self._ring)

    def add(self, packet: FramePacket):
        """Return completed requests; caller must flush before a session reset."""
        if self._session not in (None, packet.session_id):
            raise ValueError('Flush/reset recorder before changing session')
        t = packet.source_t_s
        if self._last_input is not None and t <= self._last_input:
            raise ValueError('Recorder requires strictly increasing source timestamps')
        self._session = packet.session_id
        self._offset = packet.source_time_offset_s
        self._last_input = t
        while self._ring and (t - self._ring[0].source_t_s >= RETENTION_S - 1e-9):
            self._ring.popleft()
        while self._capacity_gaps and t - self._capacity_gaps[0] >= RETENTION_S:
            self._capacity_gaps.popleft()
        if self._last_sample is None or t - self._last_sample >= 1 / FPS - 1e-9:
            # Reserve both resize scratch and retained immutable allocation before
            # allocating. The source packet belongs to detection, not this ring.
            try:
                self.budget.reserve(2 * FRAME_BYTES)
            except MemoryError:
                self._last_sample = t  # Keep failed sampling attempts bounded too.
                self._capacity_gaps.append(t)
                for request in self._pending.values():
                    if request['start'] <= t <= request['end']:
                        request['capacity_gap'] = True
            else:
                try:
                    small = cv2.resize(packet.frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    pixels = _Pixels(small, self.budget)
                except Exception:
                    self.budget.release(FRAME_BYTES)
                    raise
                finally:
                    self.budget.release(FRAME_BYTES)
                sample = ClipFrame(t, packet.sequence,
                    np.frombuffer(pixels, np.uint8).reshape(HEIGHT, WIDTH, 3))
                self._ring.append(sample)
                self._last_sample = t
                for request in self._pending.values():
                    if request['start'] <= t <= request['end'] + 1e-9:
                        request['frames'].append(sample)
        ready = []
        for key, request in tuple(self._pending.items()):
            if t >= request['end'] - 1e-9:
                ready.append(self._finish(key))
        return tuple(ready)

    def request(self, incident_id, onset_s, confirmed_s):
        text(incident_id, 'incident_id')
        number(onset_s, 'onset_s')
        number(confirmed_s, 'confirmed_s', onset_s)
        if self._session is None or self._last_input != confirmed_s:
            raise ValueError('Request must follow the confirmation frame')
        if incident_id in self._pending:
            raise ValueError('Clip already requested')
        if len(self._pending) >= self.max_pending:
            raise MemoryError('Optional clip request capacity exceeded')
        start, end = onset_s - 3., confirmed_s + 3.
        self._pending[incident_id] = {'start': start, 'end': end,
            'capacity_gap': any(start <= t <= end for t in self._capacity_gaps),
            'frames': [f for f in self._ring if start <= f.source_t_s <= end]}

    def _finish(self, key, reason=None):
        request = self._pending.pop(key)
        frames = tuple(request['frames'])
        reasons = [] if reason is None else [reason]
        if request['capacity_gap']:
            reasons.append('Media capacity limit omitted sampled frames')
        if not frames:
            reasons.append('No media frames available (capacity or source unavailable)')
        else:
            if request['start'] < 0 or frames[0].source_t_s > request['start'] + 1/FPS + 1e-9:
                reasons.append('Pre-onset history incomplete')
            if frames[-1].source_t_s < request['end'] - 1/FPS - 1e-9:
                reasons.append('Post-confirmation history incomplete')
            if any(b.source_t_s - a.source_t_s > 2/FPS + 1e-9 for a, b in zip(frames, frames[1:])):
                reasons.append('Source/sampling gap; no detection evidence interpolated')
        return ClipPayload(key, self._session, request['start'], request['end'], frames,
                           bool(reasons), tuple(reasons), self._offset)

    def flush(self, reason='Source ended before post-roll completed'):
        ready = tuple(self._finish(key, reason) for key in tuple(self._pending))
        self._ring.clear()
        self._capacity_gaps.clear()
        self._session = self._last_input = self._last_sample = None
        return ready
