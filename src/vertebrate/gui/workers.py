"""Four contexts, bounded mailboxes, and a timer-polled immutable display slot.

The QObject owners run cooperative loops on three QThreads. Commands use bounded
Python queues because a blocking reader cannot service queued Qt slots. No frame
signals are emitted. The coordinator is main-thread-only; its futures must be
polled (or observed with callbacks), never waited on from the GUI thread.
"""
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, get_ident
import time

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Slot

from ..capture import CaptureSessionManager
from ..config import AppConfig
from ..contracts import EndOfStreamEvent, FramePacket, PoseObservation, SessionResetEvent
from ..incidents import IncidentFactory
from ..pipeline import HeadlessPipeline
from ..storage import StorageError, StorageWorker


@dataclass(frozen=True)
class PipelineStatus:
    state: str = 'stopped'
    message: str = ''
    frames: int = 0
    dropped_frames: int = 0
    network_attempts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DisplayResult:
    frame: FramePacket
    decisions: tuple
    observation: PoseObservation | None = None
    processing_fps: float = 0.
    source_fps: float = 0.
    stillness: tuple = ()

    def __post_init__(self):
        # FramePacket takes its own immutable bytes-backed copy. A painter can
        # call copy_frame() when it needs a writable/QImage-owned buffer.
        object.__setattr__(self, 'frame', replace(self.frame))
        object.__setattr__(self, 'decisions', tuple(self.decisions))
        if self.observation is not None and (
                self.observation.session_id, self.observation.sequence, self.observation.source_t_s,
                self.observation.width, self.observation.height) != (
                self.frame.session_id, self.frame.sequence, self.frame.source_t_s,
                self.frame.width, self.frame.height):
            raise ValueError('Display overlay must match the frame sequence and source')


class LatestResultSlot:
    """One retained result, replaced atomically; never a queue of GUI frames."""
    def __init__(self):
        self._lock = Lock()
        self._result = None
        self._status = PipelineStatus()

    def set(self, result: DisplayResult | None):
        if result is not None and not isinstance(result, DisplayResult):
            raise TypeError('Expected immutable DisplayResult')
        with self._lock:
            self._result = result

    def get(self):
        with self._lock:
            return self._result

    @property
    def status(self):
        with self._lock:
            return self._status

    def update_status(self, **changes):
        with self._lock:
            self._status = replace(self._status, **changes)


class _LoopOwner(QObject):
    def __init__(self, target):
        super().__init__()
        self.target = target

    @Slot()
    def run(self):
        try:
            self.target()
        finally:
            self.target = None
            QThread.currentThread().quit()


class QtWorkerThread:
    """Thread-compatible adapter: lets the existing writer keep its sole owner."""
    def __init__(self, *, target, name, daemon=True):
        self.qt_thread = QThread()
        self.qt_thread.setObjectName(name)
        self.owner = _LoopOwner(target)
        self.owner.moveToThread(self.qt_thread)
        self.qt_thread.started.connect(self.owner.run)
        self.qt_thread.finished.connect(self.owner.deleteLater)

    def start(self):
        self.qt_thread.start()

    def is_alive(self):
        # isRunning can become false before native thread-local cleanup ends.
        # A zero-time wait is a nonblocking completion fence before deleteLater.
        return self.qt_thread is not None and (self.qt_thread.isRunning() or not self.qt_thread.wait(0))

    def join(self, timeout=None):
        if self.qt_thread is not None:
            self.qt_thread.wait(round(timeout * 1000) if timeout is not None else 0xFFFFFFFF)

    def dispose(self):
        if self.is_alive():
            raise RuntimeError('Cannot dispose a running worker')
        self.qt_thread.deleteLater()
        self.qt_thread = self.owner = None


# Ultralytics IDs and offline patches are process-global. Permit only one active
# threaded vision pipeline, and keep timed-out owners alive until they unwind.
_VISION_LEASE = Lock()
_ACTIVE = set()


def _resolve(future, value=None, error=None):
    if not future.done():
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)


class ThreadedPipeline(QObject):
    """Nonblocking main-thread Start/Stop/Pause/Resume/Reset coordinator.

start/stop/commands return Futures; run the Qt event loop to observe completion.
stop drains accepted storage commands, aborts unprocessed frames, and reports a
timeout without terminating a native reader or releasing it from the GUI.
Factories are called in their owning workers, permitting hardware-free tests.
"""
    def __init__(self, config: AppConfig | None = None, root_dir=None, *,
                 pipeline_factory=None, storage_factory=StorageWorker,
                 environment=None, location=None):
        super().__init__()
        self._assert_main()
        self.config = config or AppConfig()
        self.root = Path(root_dir or Path.cwd()).resolve()
        self.pipeline_factory = pipeline_factory
        self.storage_factory = storage_factory
        self.environment, self.location = environment, location
        self.latest = LatestResultSlot()
        self._threads = []
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._poll)
        self._start_future = self._stop_future = None
        self.storage = None
        self.owner_ids = {}

    @staticmethod
    def _assert_main():
        app = QCoreApplication.instance()
        if app is None or QThread.currentThread() != app.thread():
            raise RuntimeError('ThreadedPipeline requires the Qt main thread and application')

    @property
    def status(self):
        return self.latest.status

    @property
    def running_threads(self):
        return sum(thread.is_alive() for thread in self._threads)

    def start(self, source, *, timeout_s=30, **capture_options):
        self._assert_main()
        if self._threads:
            raise RuntimeError('Stop all owners before starting another run')
        retained = self.storage is not None and bool(self.storage.unsaved_incident_ids)
        if retained and self.storage.outbox != (self.root / self.config.runtime.outbox_dir).resolve():
            raise StorageError('Retry retained incidents before changing the outbox')
        self._validate_timeout(timeout_s)
        if not _VISION_LEASE.acquire(blocking=False):
            raise RuntimeError('Another threaded vision pipeline is active')
        try:
            # Python 3.12 Windows monotonic_ns can have a coarse clock tick.
            # perf_counter_ns is also monotonic, with sufficient resolution for
            # successive camera frames; the entire session uses the same clock.
            capture_options.setdefault('monotonic_ns_fn', time.perf_counter_ns)
            self.capture = CaptureSessionManager(source, capacity=self.config.runtime.capture_queue_capacity,
                                                 **capture_options)
            self._cancel, self._hold, self._faulted = Event(), Event(), Event()
            self._vision_ready, self._capture_ready = Event(), Event()
            self._commands, self._capture_commands = Queue(8), Queue(8)
            self._start_future, self._stop_future = Future(), None
            self._start_deadline = time.monotonic() + timeout_s
            self._stop_deadline = None
            self.owner_ids = {'gui': get_ident()}
            self.latest.set(None)
            self.latest.update_status(state='starting', message='', frames=0, dropped_frames=0, network_attempts=())
            if retained:
                self._storage_fault()
                self.storage.reopen(thread_factory=QtWorkerThread, wait_ready=False)
            else:
                self.storage = self.storage_factory(
                    self.root / self.config.runtime.outbox_dir, pause_acquisition=self._storage_fault,
                    capacity=self.config.runtime.incident_queue_capacity,
                    max_media_buffer_bytes=self.config.runtime.max_media_buffer_bytes)
                self.storage.start(thread_factory=QtWorkerThread, wait_ready=False)
            self._threads = [self.storage._thread]
            for name, target in [('capture', self._capture_loop), ('vision', self._vision_loop)]:
                thread = QtWorkerThread(target=target, name=f'vertebrate-{name}')
                self._threads.append(thread)
                thread.start()
            _ACTIVE.add(self)
            self._timer.start()
            return self._start_future
        except Exception:
            if self._threads:
                _ACTIVE.add(self)
                self.stop()
            else:
                _VISION_LEASE.release()
            raise

    @staticmethod
    def _validate_timeout(value):
        import math
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('timeout_s must be finite and positive')

    def _storage_fault(self):
        self._hold.set()
        self._faulted.set()
        self.latest.update_status(state='storage_fault', message='Storage fault: acquisition and processing paused; alerts retained for retry')

    def _failure(self, exc):
        message = f'{type(exc).__name__}: {exc}'
        self.latest.update_status(state='error', message=message)
        self._cancel.set()
        self.capture.queue.close()
        # Tracebacks can retain a model/native reader in worker stack frames.
        # Send only an error value across the ownership boundary.
        error = TimeoutError(message) if isinstance(exc, TimeoutError) else RuntimeError(message)
        _resolve(self._start_future, error=error)

    def _capture_loop(self):
        self.owner_ids['capture'] = get_ident()
        try:
            while not self._vision_ready.wait(.02):
                if self._cancel.is_set():
                    return
            if self._cancel.is_set():
                return
            self.capture.start()
            self._source_fps = self.capture._reader.fps
            self._capture_ready.set()
            while not self._cancel.is_set():
                try:
                    name, args, future = self._capture_commands.get_nowait()
                except Empty:
                    if not self._hold.is_set() and not self._faulted.is_set() and self.capture.state == 'running':
                        self.capture.pump(timeout_s=.02)
                    else:
                        self._cancel.wait(.01)
                    continue
                try:
                    _resolve(future, getattr(self.capture, name)(*args))
                    if self.capture._reader is not None:
                        self._source_fps = self.capture._reader.fps
                except Exception as exc:
                    _resolve(future, error=exc)
                finally:
                    self._capture_commands.task_done()
        except Exception as exc:
            self._failure(exc)
        finally:
            try:
                self.capture.stop()  # only the reader owner may release it
            except Exception as exc:
                self._failure(exc)
            self._drain_commands(self._capture_commands)

    def _vision_loop(self):
        self.owner_ids['vision'] = get_ident()
        attempts = []
        pipe = None
        try:
            from ..pose import PoseAdapter, guarded_runtime
            with guarded_runtime(self.root, attempts):
                while not self.storage.ready:
                    if self._cancel.wait(.01):
                        return
                if self.storage.startup_error:
                    raise StorageError(str(self.storage.startup_error))
                if self._cancel.is_set():
                    return
                self.owner_ids['storage'] = self.storage._owner
                if self.pipeline_factory:
                    pipe = self.pipeline_factory(self.config, self.root)
                else:
                    pipe = HeadlessPipeline(self.config, self.root,
                                            pose=PoseAdapter(self.config, self.root, headless=True))
                pipe.retain_history = False
                factory = IncidentFactory()
                current = None

                def alert_ready(alert, features):
                    incident = factory.create(alert, features, current, self.config,
                        confirmed_at_utc=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                        environment=self.environment, location=self.location)
                    self.storage.submit(incident, self.config, current.frame_bgr)

                pipe.on_alert = alert_ready
                self._vision_ready.set()
                while not self._cancel.is_set():
                    try:
                        command = self._commands.get_nowait()
                    except Empty:
                        command = None
                    if command is not None:
                        self._execute_command(command)
                        self._commands.task_done()
                        continue
                    if self._hold.is_set() or self._faulted.is_set():
                        self._cancel.wait(.01)
                        continue
                    try:
                        current = self.capture.queue.get(timeout_s=.02)
                    except Empty:
                        continue
                    if isinstance(current, SessionResetEvent):
                        # Fresh per-session state bounds retired identities and
                        # candidate caches; the queue is the stale-frame fence.
                        pipe.source_error()
                        pipe._retired_sessions.clear()
                        factory = IncidentFactory()
                        self.latest.set(None)
                    started = time.perf_counter()
                    decisions = pipe.process(current)
                    if isinstance(current, FramePacket):
                        elapsed = time.perf_counter() - started
                        stillness = tuple((key, max(0., current.source_t_s - fsm._still_start))
                            for key, (_, fsm) in pipe._people.items() if fsm._still_start is not None)
                        self.latest.set(DisplayResult(current, decisions, pipe.last_observation,
                            1 / elapsed if elapsed > 0 else 0., getattr(self, '_source_fps', 0.), stillness))
                        self.latest.update_status(frames=pipe.frames,
                            dropped_frames=self.capture.queue.dropped_frames_count)
                    elif isinstance(current, EndOfStreamEvent) and not self._faulted.is_set() and not self._cancel.is_set():
                        self.latest.update_status(state='eof', message='End of source; incomplete observations closed')
                pipe.source_error()
        except Exception as exc:
            if pipe is not None:
                pipe.source_error()
            self._failure(exc)
        finally:
            # Drop model/tracker/frame references in their owner, before exit.
            pipe = None
            self.latest.update_status(network_attempts=tuple(attempts))
            self._drain_commands(self._commands)

    @staticmethod
    def _drain_commands(queue):
        while True:
            try:
                _, _, future = queue.get_nowait()
            except Empty:
                return
            _resolve(future, error=RuntimeError('Pipeline stopped before command completion'))
            queue.task_done()

    def _execute_command(self, command):
        name, args, future = command
        try:
            if self._faulted.is_set():
                if name != 'resume' or self.storage.unsaved_incident_ids or self.storage.startup_error:
                    raise StorageError('Storage fault is latched; retry retained records before resuming')
                self._faulted.clear()
                self.storage.pause_requested.clear()
                if self.capture.state == 'running':
                    self._hold.clear()
                    self.latest.update_status(state='running', message='')
                    _resolve(future)
                    return
            reply = Future()
            self._capture_commands.put_nowait((name, args, reply))
            while not reply.done():
                if self._cancel.wait(.01):
                    raise RuntimeError('Command interrupted by stop')
            result = reply.result()
            if self._faulted.is_set():
                raise StorageError('Storage fault interrupted control command')
            if name == 'pause':
                self._hold.set()
                self.latest.update_status(state='paused', message='Source timeline paused')
            else:
                if name != 'resume':
                    self.latest.set(None)
                self._hold.clear()
                self.latest.update_status(state='running', message='')
            _resolve(future, result)
        except Exception as exc:
            _resolve(future, error=exc)

    def _command(self, name, *args):
        self._assert_main()
        future = Future()
        if not self._threads or self._cancel.is_set():
            _resolve(future, error=RuntimeError('Pipeline is not running'))
        else:
            try:
                self._commands.put_nowait((name, args, future))
            except Full:
                _resolve(future, error=RuntimeError('Control queue is full; retry after acknowledgement'))
        return future

    def pause(self): return self._command('pause')
    def resume(self): return self._command('resume')
    def reset(self): return self._command('restart')
    def replay(self): return self._command('replay')
    def seek(self, frame_index): return self._command('seek', frame_index)
    def switch_source(self, source, recorded_start_utc=None):
        return self._command('switch_source', source, recorded_start_utc)

    def stop(self, *, timeout_s=10):
        self._assert_main()
        self._validate_timeout(timeout_s)
        if self._stop_future is not None and not self._stop_future.done():
            return self._stop_future
        self._stop_future = Future()
        if not self._threads:
            _resolve(self._stop_future)
            return self._stop_future
        self._cancel.set()
        self.capture.queue.close()
        self._stop_deadline = time.monotonic() + timeout_s
        self.latest.update_status(state='stopping', message='Finishing critical writes')
        _resolve(self._start_future, error=RuntimeError('Start interrupted by stop'))
        self._timer.start()
        return self._stop_future

    @Slot()
    def _poll(self):
        if not self._threads:
            return
        now = time.monotonic()
        if self._faulted.is_set() and not self._cancel.is_set():
            self.latest.update_status(state='storage_fault', message='Storage fault: acquisition and processing paused; alerts retained for retry')
        if not self._start_future.done():
            if self._capture_ready.is_set() and self._vision_ready.is_set():
                if self.status.state == 'starting':
                    self.latest.update_status(state='running')
                _resolve(self._start_future, self.capture.session_id)
            elif now >= self._start_deadline:
                self._failure(TimeoutError('Pipeline start timed out; source/model may be blocked'))
        if self._cancel.is_set():
            if self._stop_deadline is None:
                self._stop_deadline = now + 10
            # No more submitters once vision has exited. Drain the writer then.
            if len(self._threads) < 3 or not self._threads[2].is_alive():
                self.storage.request_stop()
            if not self.running_threads:
                for thread in self._threads:
                    thread.dispose()
                self._threads.clear()
                self.latest.set(None)
                if self.status.state not in ('error', 'stop_timeout', 'storage_fault'):
                    self.latest.update_status(state='stopped', message='')
                if self._stop_future is not None:
                    _resolve(self._stop_future)
                self._timer.stop()
                _ACTIVE.discard(self)
                _VISION_LEASE.release()
            elif now >= self._stop_deadline:
                message = 'Stop timed out: blocked capture/vision/storage owner; resources retained until cooperative exit'
                self.latest.update_status(state='stop_timeout', message=message)
                if self._stop_future is not None:
                    _resolve(self._stop_future, error=TimeoutError(message))
