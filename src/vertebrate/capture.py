"""Bounded capture with a single owner for every reader and lifecycle change.

CaptureSessionManager is a synchronous pump for deterministic integrations.
CaptureWorker serializes its public commands on a dedicated owner thread.
"""
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread, get_ident
import time
from typing import Protocol
import uuid

from .clocks import LiveSourceClock, VideoFileClock
from .contracts import (CaptureError, EndOfStreamEvent, FramePacket, QueueItem,
                        SessionResetEvent, SourceKind, SourceTimingError, TimingMode)
from .validation import integer, number, utc_string


class BoundedFrameQueue:
    """Capacity counts both frames and control events; controls are never evicted.

    A LIVE queue containing only controls uses finite backpressure (Full on
    timeout). It is impossible to guarantee bounded storage, no control loss,
    and unconditional nonblocking insertion at the same time.
    """
    def __init__(self, source_kind: SourceKind, capacity: int = 2):
        integer(capacity, 'capacity', 1)
        if not isinstance(source_kind, SourceKind):
            raise ValueError('source_kind must be SourceKind')
        self.capacity = capacity
        self.source_kind = source_kind
        self.dropped_frames_count = 0
        self.reset_discarded_frames_count = 0
        self._items = deque()
        self._condition = Condition()
        self._closed = False
        self._session_id = None

    def __len__(self):
        with self._condition:
            return len(self._items)

    def put(self, item: QueueItem, timeout_s: float = 1.0, cancel: Event | None = None) -> bool:
        if not isinstance(item, (FramePacket, SessionResetEvent, EndOfStreamEvent)):
            raise TypeError('Expected FramePacket or lifecycle event')
        number(timeout_s, 'timeout_s')
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._closed or (cancel is not None and cancel.is_set()):
                    return False
                if isinstance(item, FramePacket):
                    if item.source_kind != self.source_kind:
                        raise CaptureError('Frame source kind differs from queue policy')
                    if self._session_id is not None and item.session_id != self._session_id:
                        return False  # producer was cancelled by an atomic reset
                if len(self._items) < self.capacity:
                    self._items.append(item)
                    self._condition.notify_all()
                    return True
                if self.source_kind == SourceKind.LIVE:
                    victim = next((i for i, x in enumerate(self._items) if isinstance(x, FramePacket)), None)
                    if victim is not None:
                        del self._items[victim]
                        self.dropped_frames_count += 1
                        continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Full('Capture queue full; consume lifecycle events/frames or cancel')
                self._condition.wait(min(remaining, .02))

    def get(self, timeout_s: float = 1.0) -> QueueItem:
        number(timeout_s, 'timeout_s')
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if self._closed or remaining <= 0:
                    raise Empty('Capture queue empty/closed')
                self._condition.wait(remaining)
            result = self._items.popleft()
            self._condition.notify_all()
            return result

    def reset(self, event: SessionResetEvent):
        """Atomically discard stale frames and append the reset after old controls.

        If controls alone fill capacity, fail explicitly without changing state.
        Consumers can drain them and retry. Blocked old producers cannot leak.
        """
        with self._condition:
            controls = deque(x for x in self._items if not isinstance(x, FramePacket))
            if len(controls) >= self.capacity:
                raise Full('Drain lifecycle events before another reset')
            self.reset_discarded_frames_count += len(self._items) - len(controls)
            controls.append(event)
            self._items = controls
            self._session_id = event.new_session_id
            self.source_kind = event.source_kind
            self.dropped_frames_count = 0  # per-session metric, including after source switch
            self._closed = False
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FrameSource(Protocol):
    """Methods run only in the owning manager/worker context.

    Video fps is verified by the adapter, not a guessed FPS. read returns
    (None, None) only at normal EOF and raises CaptureError on other failures.
    """
    fps: float
    frame_count: int

    def open(self) -> None: ...
    def read(self) -> tuple[object | None, float | None]: ...
    def seek(self, frame_index: int) -> None: ...
    def release(self) -> None: ...


class OpenCVFrameSource:
    """Local files/camera only. Verify all decoded file PTS before first emission.

    Missing PTS are accepted for fixed-rate AVI, or when the caller supplies an
    independently verified fps. Other opaque containers with no PTS are rejected.
    Full validation is a bounded-memory read pass, not a cached frame collection.
    """
    def __init__(self, source: int | str | Path, verified_fps: float | None = None):
        self.source = source
        self.verified_fps = verified_fps
        self._cap = None
        self.fps = 0.0
        self.frame_count = 0
        self.frames_read = 0
        self.timing_modes = set()

    def open(self):
        import cv2
        if type(self.source) is not int:
            path = Path(self.source)
            if '://' in str(path) or str(path).startswith(('\\\\', '//')) or not path.is_file():
                raise CaptureError(f'Expected an existing local video file: {self.source}')
        elif self.source < 0:
            raise CaptureError('Camera index must be nonnegative')
        self._cap = cv2.VideoCapture(self.source if type(self.source) is int else str(self.source))
        try:
            if not self._cap.isOpened():
                raise CaptureError(f'Cannot open source: {self.source}')
            if type(self.source) is int:
                return
            self.fps = self._cap.get(cv2.CAP_PROP_FPS)
            clock = VideoFileClock(self.fps)
            count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            try:
                number(count, 'frame count')
            except ValueError as exc:
                raise SourceTimingError(str(exc)) from exc
            if count <= 0 or not float(count).is_integer():
                raise SourceTimingError('Missing/invalid video frame count')
            self.frame_count = int(count)
            with Path(self.source).open('rb') as stream:
                header = stream.read(12)
            fixed_avi = header[:4] == b'RIFF' and header[8:12] == b'AVI '
            if self.verified_fps is not None:
                VideoFileClock(self.verified_fps)
                if abs(self.verified_fps - self.fps) > 1e-6:
                    raise SourceTimingError('Supplied verified FPS differs from container FPS')
            for index in range(self.frame_count):
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    raise CaptureError(f'Truncated/unreadable video at frame {index} of {self.frame_count}')
                pts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                _, mode = clock.compute_frame_time(index, pts)
                self.timing_modes.add(mode)
                if mode == TimingMode.FRAME_INDEX_FPS_FALLBACK and not fixed_avi and self.verified_fps is None:
                    raise SourceTimingError('PTS unavailable; CFR cannot be verified for this container. '
                                            'Supply independently verified_fps or convert')
            if self._cap.read()[0]:
                raise SourceTimingError('Decoded frame count exceeds container frame count')
            self.seek(0)
        except Exception:
            self.release()
            raise

    def read(self):
        import cv2
        if self._cap is None:
            raise CaptureError('Source is not open')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if type(self.source) is int:
                raise CaptureError(f'Live camera read failed: {self.source}')
            if self.frames_read != self.frame_count:
                raise CaptureError(f'Unexpected decode failure at frame {self.frames_read}')
            return None, None
        pts = None if type(self.source) is int else self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        self.frames_read += 1
        return frame, pts

    def seek(self, frame_index):
        import cv2
        integer(frame_index, 'frame_index')
        if type(self.source) is int or self._cap is None or frame_index >= self.frame_count:
            raise CaptureError('Seek requires an open file and an in-range frame index')
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise CaptureError(f'Backend refused seek to frame {frame_index}')
        if abs(self._cap.get(cv2.CAP_PROP_POS_FRAMES) - frame_index) > .1:
            raise CaptureError('Backend seek position differs from requested frame')
        self.frames_read = frame_index

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CaptureSessionManager:
    """Synchronous, single-owner capture. Call pump repeatedly and drain queue.

    On seek, source_t_s is rebased to zero; source_time_offset_s and
    source_frame_index preserve alignment with original clip annotations/UTC.
    """
    def __init__(self, source: int | str | Path,
                 source_factory: Callable = OpenCVFrameSource, capacity: int = 2,
                 monotonic_ns_fn=time.monotonic_ns,
                 utc_now_fn=lambda: datetime.now(timezone.utc),
                 session_id_fn=lambda: str(uuid.uuid4()), recorded_start_utc: str | None = None):
        self.source = source
        self.source_factory = source_factory
        self.queue = BoundedFrameQueue(self._kind(source), capacity)
        self.monotonic_ns_fn = monotonic_ns_fn
        self.utc_now_fn = utc_now_fn
        self.session_id_fn = session_id_fn
        self.recorded_start_utc = recorded_start_utc
        self.session_id = None
        self.state = 'stopped'
        self._reader = None
        self._owner = None
        self._pending = None
        self.last_source_t_s = None
        self.sequence = 0

    @staticmethod
    def _kind(source):
        if type(source) is int:
            if source < 0:
                raise CaptureError('Camera index must be nonnegative')
            return SourceKind.LIVE
        if not isinstance(source, (str, Path)) or not str(source).strip():
            raise CaptureError('Source must be camera index or local file path')
        return SourceKind.VIDEO

    def _assert_owner(self):
        if self._owner is None:
            self._owner = get_ident()
        if self._owner != get_ident():
            raise CaptureError('Reader belongs to another thread; use CaptureWorker commands')

    def _release(self):
        if self._reader is not None:
            self._reader.release()
            self._reader = None

    def _reset(self, reason, frame_index=0):
        new_id = self.session_id_fn()
        kind = self._kind(self.source)
        event = SessionResetEvent(self.session_id, new_id, reason, kind, str(self.source))
        if kind == SourceKind.LIVE:
            clock = LiveSourceClock(self.monotonic_ns_fn, self.utc_now_fn)
            clock.start_session(new_id)
            started_utc = clock.session_start_utc
            started_ns = clock.session_start_monotonic_ns
            offset = 0.0
        else:
            clock = VideoFileClock(self._reader.fps, self.recorded_start_utc)
            clock.reset(frame_index)
            offset = frame_index / clock.verified_fps
            started_ns = self.monotonic_ns_fn()
            started_utc = utc_string(self.utc_now_fn())
        self.queue.reset(event)
        self.session_id = new_id
        self.clock = clock
        self.session_start_utc = started_utc
        self.session_start_monotonic_ns = started_ns
        self.source_time_offset_s = offset
        self._frame_index = frame_index
        self.sequence = 0
        self.last_source_t_s = None
        self._pending = None
        self.state = 'running'
        return new_id

    def _open(self, reason, frame_index=0):
        self._assert_owner()
        self._release()
        self._reader = self.source_factory(self.source)
        try:
            self._reader.open()
            if frame_index:
                self._reader.seek(frame_index)
            return self._reset(reason, frame_index)
        except Exception as exc:
            self._release()
            self.state = 'error'
            if isinstance(exc, (CaptureError, SourceTimingError, Full)):
                raise
            raise CaptureError(f'Cannot open/reset source {self.source}: {exc}') from exc

    def start(self):
        if self.state in ('running', 'paused'):
            raise CaptureError('Capture is already started')
        return self._open('start' if self.session_id is None else 'restart')

    def stop(self):
        self._assert_owner()
        self._release()
        self._pending = None
        self.state = 'stopped'
        self.queue.close()

    def pause(self):
        self._assert_owner()
        if self._kind(self.source) == SourceKind.LIVE or self.state != 'running':
            raise CaptureError('Pause requires a running video file')
        self.clock.pause()
        self.state = 'paused'

    def resume(self):
        self._assert_owner()
        if self.state != 'paused':
            raise CaptureError('Resume requires a paused video file')
        self.clock.resume()
        self.state = 'running'

    def seek(self, frame_index):
        self._assert_owner()
        integer(frame_index, 'frame_index')
        if self._kind(self.source) != SourceKind.VIDEO:
            raise CaptureError('Live sources cannot seek')
        # Reopening verifies timing, handles seek after EOF, and keeps one owner.
        return self._open('seek', frame_index)

    def replay(self):
        self._assert_owner()
        if self._kind(self.source) != SourceKind.VIDEO:
            raise CaptureError('Replay requires a video file')
        return self._open('replay')

    def restart(self):
        return self._open('restart')

    def switch_source(self, source, recorded_start_utc=None):
        self._assert_owner()
        self._kind(source)
        self.source = source
        self.recorded_start_utc = recorded_start_utc
        return self._open('source_switch')

    def estimate_event_utc(self, source_t_s):
        return self.clock.estimate_event_utc(source_t_s + self.source_time_offset_s)

    def pump(self, timeout_s=.02) -> QueueItem | None:
        self._assert_owner()
        if self.state != 'running':
            return None
        try:
            if self._pending is None:
                frame, pts = self._reader.read()
                if frame is None:
                    if self._kind(self.source) == SourceKind.LIVE:
                        raise CaptureError('Live source ended unexpectedly')
                    if self._frame_index != self._reader.frame_count:
                        raise CaptureError('Video ended before its declared frame count')
                    self._pending = EndOfStreamEvent(self.session_id, str(self.source), self.sequence,
                                                     self.sequence - 1 if self.sequence else None,
                                                     self.last_source_t_s)
                else:
                    if self._kind(self.source) == SourceKind.LIVE:
                        seq, source_t, stamp = self.clock.tick()
                        mode = TimingMode.LIVE_MONOTONIC
                    else:
                        if self._frame_index >= self._reader.frame_count:
                            raise CaptureError('Video emitted more than declared frame count')
                        absolute_t, mode = self.clock.compute_frame_time(self._frame_index, pts)
                        source_t = max(0.0, absolute_t - self.source_time_offset_s)
                        seq, stamp = self.sequence, self.monotonic_ns_fn()
                    self._pending = FramePacket(
                        self.session_id, seq, self._kind(self.source), str(self.source),
                        frame.shape[1], frame.shape[0], frame, source_t, stamp, mode,
                        self.session_start_utc, self.session_start_monotonic_ns,
                        self.recorded_start_utc, self._frame_index, self.source_time_offset_s)
            try:
                if not self.queue.put(self._pending, timeout_s):
                    return None
            except Full:
                return None  # retained pending item: retry without another read/tick
            emitted = self._pending
            self._pending = None
            if isinstance(emitted, EndOfStreamEvent):
                self.state = 'eof'
                self._release()
            else:
                self.sequence += 1
                self._frame_index += 1
                self.last_source_t_s = emitted.source_t_s
            return emitted
        except Exception as exc:
            self.state = 'error'
            self._release()
            if isinstance(exc, (CaptureError, SourceTimingError)):
                raise
            raise CaptureError(f'{self.source}: {exc}') from exc


VideoCaptureController = CaptureSessionManager


class CaptureWorker:
    """Threaded adapter; synchronous command acknowledgements define reset fences.

    stop is cooperative; a native camera backend stuck in read cannot safely be
    released from another thread. A timeout reports that condition explicitly.
    """
    def __init__(self, source, **kwargs):
        self.controller = CaptureSessionManager(source, **kwargs)
        self.queue = self.controller.queue
        self._commands = Queue()
        self._thread = None
        self.error = None

    @property
    def session_id(self):
        return self.controller.session_id

    @property
    def state(self):
        return self.controller.state

    def _run(self):
        while True:
            try:
                name, args, done, result = self._commands.get(timeout=.01)
            except Empty:
                if self.controller.state == 'running':
                    try:
                        self.controller.pump()
                    except Exception as exc:
                        self.error = exc
                continue
            try:
                result.append(getattr(self.controller, name)(*args))
            except Exception as exc:
                result.append(exc)
            finally:
                done.set()
            if name == 'stop':
                return

    def _call(self, name, *args, timeout_s=10):
        if self._thread is None or not self._thread.is_alive():
            raise CaptureError('Capture worker is not running')
        done, result = Event(), []
        self._commands.put((name, args, done, result))
        if not done.wait(timeout_s):
            raise CaptureError(f'Capture {name} did not finish in {timeout_s}s; backend may be blocked')
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def start(self):
        if self._thread is not None:
            raise CaptureError('Worker can be started once; use restart for a new source session')
        self._thread = Thread(target=self._run, name='vertebrate-capture', daemon=True)
        self._thread.start()
        try:
            return self._call('start')
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self._thread is not None and self._thread.is_alive():
            self._call('stop')
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise CaptureError('Capture worker did not stop cooperatively')

    def pause(self): return self._call('pause')
    def resume(self): return self._call('resume')
    def seek(self, frame_index): return self._call('seek', frame_index)
    def replay(self): return self._call('replay')
    def restart(self): return self._call('restart')
    def switch_source(self, source, recorded_start_utc=None):
        return self._call('switch_source', source, recorded_start_utc)
