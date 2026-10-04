"""Single local writer, commit-marker JSON, retained failures and explicit pause.

StorageWorker requires a pause_acquisition callback (e.g. CaptureWorker.pause
for files, or a cooperative stop/request-pause control for cameras). The fault
Event and immutable records are available immediately to the future GUI. Only
the writer thread mutates incident files; acknowledgement is also queued.
"""
from concurrent.futures import Future
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, get_ident
from uuid import uuid4

import numpy as np

from .config import AppConfig, config_sha256, config_to_dict, load_config
from .incidents import (Acknowledgement, Delivery, Incident, Media, incident_from_json,
                        incident_to_dict, incident_to_json, relative_path)
from .validation import integer, utc_datetime
from .recorder import ClipPayload, MediaBudget, FPS, WIDTH, HEIGHT


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorageRecord:
    incident: Incident
    status: str
    error: str | None = None


@dataclass(frozen=True)
class InboxScan:
    incidents: tuple[Incident, ...]
    issues: tuple[str, ...]


def _contained(folder: Path, name: str) -> Path:
    relative_path(name)
    target = folder / name
    if not target.resolve().is_relative_to(folder.resolve()):
        raise StorageError(f'Path escapes local incident directory: {name}')
    return target


def _read_complete(folder: Path) -> Incident:
    incident = incident_from_json(_contained(folder, 'incident.json').read_text(encoding='utf-8'))
    if incident.incident_id != folder.name or incident.delivery.status != 'saved':
        raise StorageError('Invalid incident ID/delivery completeness marker')
    cfg = load_config(_contained(folder, incident.configuration.snapshot_path))
    if (cfg.profile, cfg.revision) != (incident.configuration.profile, incident.configuration.revision):
        raise StorageError('Configuration snapshot does not match incident')
    if incident.media.snapshot_path:
        import cv2
        data = _contained(folder, incident.media.snapshot_path).read_bytes()
        if not data:
            raise StorageError('Snapshot is empty')
        try:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise StorageError(f'Invalid snapshot: {exc}') from exc
        if image is None or not image.size:
            raise StorageError('Snapshot missing or not decodable')
    if incident.media.clip_path and not _contained(folder, incident.media.clip_path).is_file():
        raise StorageError('Declared clip is missing')
    if incident.media.clip_timestamps_path:
        json.loads(_contained(folder, incident.media.clip_timestamps_path).read_text(encoding='utf-8'))
    return incident


def scan_outbox(outbox: Path | str) -> InboxScan:
    root = Path(outbox).resolve()
    incidents, issues = [], []
    if not root.exists():
        return InboxScan((), ())
    for folder in sorted(root.iterdir()):
        if folder.name == '.writer.lock':
            continue
        try:
            if not folder.is_dir() or not folder.resolve().is_relative_to(root):
                raise StorageError('Temporary/unrecognized outbox entry')
            if not (folder / 'incident.json').is_file():
                raise StorageError('Incomplete record without incident.json')
            incidents.append(_read_complete(folder))
            if any(folder.rglob('*.tmp')):
                issues.append(f'{folder.name}: ignored leftover temporary files')
        except (OSError, ValueError, StorageError) as exc:
            issues.append(f'{folder.name}: {exc}')
    return InboxScan(tuple(incidents), tuple(issues))


def _identity_payload(incident):
    data = incident_to_dict(incident)
    for name in ('delivery', 'acknowledgement', 'media'):
        data.pop(name)
    return data


class StorageWorker:
    def __init__(self, outbox='runtime/outbox', *, pause_acquisition, capacity=8,
                 max_media_buffer_bytes=134217728):
        if not callable(pause_acquisition):
            raise ValueError('A callable acquisition pause control is required')
        integer(capacity, 'capacity', 1)
        integer(max_media_buffer_bytes, 'max_media_buffer_bytes', 1)
        self.outbox = Path(outbox).resolve()
        self.pause_acquisition = pause_acquisition
        self.queue = Queue(capacity)
        self.max_media_buffer_bytes = max_media_buffer_bytes
        self.media_budget = MediaBudget(max_media_buffer_bytes)
        self._clip_requests = {}
        self._clip_futures = {}
        self._clip_active = set()
        self.pause_requested = Event()
        self.faults = []
        self.startup_issues = ()
        self._records = {}
        self._requests = {}
        self._futures = {}
        self._config_hashes = {}
        self._media_bytes = 0
        self._lock = Lock()
        self._ready, self._stop = Event(), Event()
        self._thread = None
        self._startup_error = None
        self._owner = None

    def start(self, *, thread_factory=Thread, wait_ready=True):
        if self._thread is not None:
            raise StorageError('Writer can only be started once')
        self._thread = thread_factory(target=self._run, name='vertebrate-storage', daemon=True)
        self._thread.start()
        if not wait_ready:
            return self
        if not self._ready.wait(10):
            raise StorageError('Storage startup timed out')
        if self._startup_error:
            raise StorageError(str(self._startup_error)) from self._startup_error
        return self

    @property
    def ready(self):
        return self._ready.is_set()

    @property
    def startup_error(self):
        return self._startup_error

    @property
    def unsaved_incident_ids(self):
        with self._lock:
            return tuple(dict.fromkeys((*self._requests, *self._clip_requests)))

    def request_stop(self):
        """Nonblocking cancellation; the owner drains accepted writes before exit."""
        with self._lock:
            self._stop.set()

    def reopen(self, *, thread_factory=Thread, wait_ready=True):
        """Restart a fully stopped owner without losing retained failed requests."""
        if self._thread is None or self._thread.is_alive() or not self._stop.is_set():
            raise StorageError('Reopen requires a fully stopped writer')
        self._thread = None
        self._stop.clear()
        self._ready.clear()
        self._startup_error = None
        return self.start(thread_factory=thread_factory, wait_ready=wait_ready)

    def record(self, incident_id) -> StorageRecord:
        with self._lock:
            return self._records[incident_id]

    @property
    def records(self):
        with self._lock:
            return tuple(self._records.values())

    def read_incident(self, incident_id) -> Future:
        """Read complete metadata/media on the single owner, never on the GUI."""
        self._assert_running()
        future = Future()
        with self._lock:
            self._assert_running()
            if incident_id not in self._records:
                raise StorageError('Unknown incident ID')
            try:
                self.queue.put_nowait(('read', incident_id, None, future))
            except Full:
                future.set_exception(StorageError('Storage busy; retry opening the incident'))
        return future

    def _fault(self, message):
        with self._lock:
            self.faults.append(message)
        self.pause_requested.set()
        try:
            self.pause_acquisition()
        except Exception as exc:
            with self._lock:
                self.faults.append(f'Acquisition pause failed: {exc}')

    def _assert_running(self):
        if self._thread is None or not self._thread.is_alive() or self._stop.is_set() or not self._ready.is_set():
            raise StorageError('Storage worker is not running')

    def submit(self, incident: Incident, config: AppConfig, snapshot=None) -> Future:
        self._assert_running()
        if not isinstance(incident, Incident) or not isinstance(config, AppConfig):
            raise ValueError('submit requires immutable Incident and AppConfig')
        if (incident.configuration.profile, incident.configuration.revision) != (config.profile, config.revision):
            raise ValueError('Incident configuration does not match supplied snapshot')
        if incident.configuration.snapshot_path != 'config.json' or incident.media.clip_path is not None:
            raise ValueError('Submit config.json and snapshot first; attach optional clip through the writer')
        if incident.media.snapshot_path not in (None, 'snapshot.jpg'):
            raise ValueError('P3 snapshot path must be snapshot.jpg')
        snapshot_error = incident.media.snapshot_error
        if snapshot is not None and (not isinstance(snapshot, np.ndarray) or snapshot.dtype != np.uint8
                                     or snapshot.ndim != 3 or snapshot.shape[2] != 3 or snapshot.size == 0):
            raise ValueError('Snapshot must be nonempty uint8 BGR')
        fault = None
        with self._lock:
            self._assert_running()
            old = self._records.get(incident.incident_id)
            if old is not None:
                if _identity_payload(old.incident) != _identity_payload(incident):
                    raise StorageError('Conflicting payload for existing incident ID')
                if incident.incident_id in self._requests and config_sha256(config) != config_sha256(self._requests[incident.incident_id][1]):
                    raise StorageError('Conflicting configuration for existing incident ID')
                if old.status == 'saved':
                    if self._config_hashes[incident.incident_id] != config_sha256(config):
                        raise StorageError('Conflicting configuration for existing incident ID')
                    future = Future()
                    future.set_result(old.incident)
                    return future
                if old.status in ('pending', 'saving'):
                    return self._futures[incident.incident_id]
                raise StorageError('Retained failed incident: use retry(incident_id)')
            if snapshot is not None:
                try:
                    self.media_budget.reserve(snapshot.nbytes)
                except MemoryError:
                    snapshot_error = 'Snapshot not retained: media memory limit exceeded'
                    snapshot = None
                    fault = snapshot_error
                else:
                    try:
                        snapshot = np.frombuffer(snapshot.tobytes(), dtype=np.uint8).reshape(snapshot.shape)
                    except Exception:
                        self.media_budget.release(snapshot.nbytes)
                        raise
                    self._media_bytes += snapshot.nbytes
            pending = replace(incident, delivery=Delivery())
            self._records[incident.incident_id] = StorageRecord(pending, 'pending')
            self._requests[incident.incident_id] = (pending, config, snapshot, snapshot_error)
            future = Future()
            self._futures[incident.incident_id] = future
            if fault is None:
                try:
                    self.queue.put_nowait(('save', incident.incident_id, None, future))
                except Full:
                    fault = 'Storage queue is full; alert retained for retry'
            if fault:
                failed = replace(pending, delivery=Delivery(status='save_failed'))
                self._records[incident.incident_id] = StorageRecord(failed, 'save_failed', fault)
                future.set_exception(StorageError(fault))
        if fault:
            self._fault(fault)
        return future

    def retry(self, incident_id) -> Future:
        self._assert_running()
        if incident_id in self._clip_requests and incident_id not in self._requests:
            return self.attach_clip(self._clip_requests[incident_id])
        fault = None
        with self._lock:
            self._assert_running()
            record = self._records[incident_id]
            if record.status in ('pending', 'saving'):
                return self._futures[incident_id]
            if record.status == 'saved':
                future = Future()
                future.set_result(record.incident)
                return future
            if incident_id not in self._requests:
                raise StorageError('No retained save request; retry acknowledgement explicitly')
            future = Future()
            self._futures[incident_id] = future
            try:
                self.queue.put_nowait(('save', incident_id, None, future))
                self._records[incident_id] = StorageRecord(replace(record.incident, delivery=Delivery()), 'pending')
            except Full:
                fault = 'Storage queue is full; alert retained for retry'
                future.set_exception(StorageError(fault))
        if fault:
            self._fault(fault)
        return future

    def attach_clip(self, payload: ClipPayload) -> Future:
        """Queue post-roll without delaying the original snapshot/JSON receipt.

        Payloads share this writer's budget. On queue/disk failure they remain
        retained and retryable, with the same visible acquisition pause policy.
        """
        self._assert_running()
        if not isinstance(payload, ClipPayload):
            raise ValueError('Expected immutable ClipPayload')
        for frame in payload.frames:
            base = frame.frame_bgr
            while isinstance(base, np.ndarray):
                base = base.base
            if getattr(base, 'budget', None) is not self.media_budget:
                raise ValueError('Clip must use the writer shared media budget')
        future, fault = Future(), None
        with self._lock:
            self._assert_running()
            key = payload.incident_id
            if key not in self._records:
                raise StorageError('Save incident before attaching clip')
            if self._records[key].incident.session_id != payload.session_id:
                raise ValueError('Clip session does not match incident')
            previous = self._clip_futures.get(key)
            if key in self._clip_active:
                return previous
            self._clip_requests[key] = payload
            self._clip_futures[key] = future
            try:
                self.queue.put_nowait(('clip', key, None, future))
                self._clip_active.add(key)
            except Full:
                fault = 'Storage queue full; optional clip retained for retry'
                future.set_exception(StorageError(fault))
        if fault:
            self._fault(fault)
        return future

    def acknowledge(self, incident_id, at_utc) -> Future:
        self._assert_running()
        utc_datetime(at_utc)
        fault = None
        future = Future()
        with self._lock:
            self._assert_running()
            if incident_id not in self._records:
                raise StorageError('Unknown incident ID')
            try:
                self.queue.put_nowait(('ack', incident_id, at_utc, future))
            except Full:
                fault = 'Storage queue full; acknowledgement was not saved; retry explicitly'
                future.set_exception(StorageError(fault))
        if fault:
            self._fault(fault)
        return future

    def stop(self, timeout_s=10):
        self.request_stop()
        if self._thread:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise StorageError('Storage stop timed out; pending incidents retained in memory')

    def _atomic_write(self, destination: Path, data: bytes):
        if get_ident() != self._owner:
            raise StorageError('Only the storage owner may write incident files')
        temporary = destination.with_name(f'.{destination.name}.{uuid4()}.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _save(self, incident, config, snapshot, snapshot_error):
        folder = self.outbox / incident.incident_id
        if not folder.resolve().is_relative_to(self.outbox):
            raise StorageError('Incident directory escapes outbox')
        folder.mkdir(exist_ok=True)
        final = _contained(folder, 'incident.json')
        if final.exists():
            existing = _read_complete(folder)
            if (_identity_payload(existing) != _identity_payload(incident)
                    or config_sha256(load_config(folder / 'config.json')) != config_sha256(config)):
                raise StorageError('Conflicting saved incident/configuration')
            return existing
        media = incident.media
        try:
            if snapshot is None or snapshot_error:
                raise StorageError(snapshot_error or 'Snapshot unavailable at confirmation')
            import cv2
            ok, encoded = cv2.imencode('.jpg', snapshot)
            if not ok:
                raise StorageError('JPEG encoder failed')
            self._atomic_write(_contained(folder, 'snapshot.jpg'), encoded.tobytes())
            media = replace(media, snapshot_path='snapshot.jpg', snapshot_error=None)
        except Exception as exc:
            media = replace(media, snapshot_path=None, snapshot_error=f'{type(exc).__name__}: {exc}')
        self._atomic_write(_contained(folder, 'config.json'),
                           (json.dumps(config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        saved = replace(incident, media=media, delivery=Delivery(status='saved'))
        # JSON is the last rename and the only completeness marker.
        self._atomic_write(final, (incident_to_json(saved) + '\n').encode())
        return saved

    def _save_clip(self, key):
        import cv2
        payload = self._clip_requests[key]
        folder = self.outbox / key
        incident = _read_complete(folder)
        if incident.media.clip_status in ('saved', 'truncated', 'unavailable'):
            return incident  # Idempotent receipt; never re-encode or undo ack.
        temporary = folder / f'.clip.{uuid4()}.tmp.avi'
        writer = None
        try:
            if not payload.frames:
                raise StorageError('; '.join(payload.reasons) or 'No clip frames available')
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (WIDTH, HEIGHT))
            if not writer.isOpened():
                raise StorageError('MJPG/AVI codec unavailable')
            for frame in payload.frames:
                writer.write(frame.frame_bgr)
            writer.release()
            writer = None
            # Some backends silently fail write(). Verify every frame locally.
            reader = cv2.VideoCapture(str(temporary))
            try:
                count = 0
                while True:
                    ok, frame = reader.read()
                    if not ok:
                        break
                    if frame.shape != (HEIGHT, WIDTH, 3):
                        raise StorageError('Clip encoder returned incorrect dimensions')
                    count += 1
                if count != len(payload.frames):
                    raise StorageError('Clip encoder produced incomplete or empty output')
            finally:
                reader.release()
            with temporary.open('rb+') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, _contained(folder, 'clip.avi'))
            self._atomic_write(_contained(folder, 'clip-timestamps.json'),
                (json.dumps(payload.metadata(), indent=2, allow_nan=False) + '\n').encode())
            media = replace(incident.media, clip_path='clip.avi', clip_timestamps_path='clip-timestamps.json',
                            clip_status='truncated' if payload.truncated else 'saved', clip_error=None)
        except Exception as exc:
            media = replace(incident.media, clip_path=None, clip_timestamps_path=None,
                            clip_status='unavailable', clip_error=f'{type(exc).__name__}: {exc}')
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass  # Optional backend teardown cannot prevent JSON fallback.
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass  # Startup ignores unreferenced temporary media.
        saved = replace(incident, media=media)
        self._atomic_write(_contained(folder, 'incident.json'), (incident_to_json(saved) + '\n').encode())
        return saved

    def _run(self):
        self._owner = get_ident()
        lease = None
        locked = False
        try:
            self.outbox.mkdir(parents=True, exist_ok=True)
            lease = (self.outbox / '.writer.lock').open('a+b')
            lease.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            probe = self.outbox / f'.probe.{uuid4()}.tmp'
            self._atomic_write(probe, b'writable')
            probe.unlink()
            scan = scan_outbox(self.outbox)
            # A prior process may have exited during optional post-roll. The
            # snapshot is complete; make the lost clip explicit on restart.
            recovered = []
            for incident in scan.incidents:
                if incident.media.clip_status == 'pending' and incident.incident_id not in self._clip_requests:
                    incident = replace(incident, media=replace(incident.media,
                        clip_status='unavailable', clip_error='Post-roll interrupted by application exit'))
                    self._atomic_write(self.outbox / incident.incident_id / 'incident.json',
                                       (incident_to_json(incident) + '\n').encode())
                recovered.append(incident)
            scan = InboxScan(tuple(recovered), scan.issues)
            self.startup_issues = scan.issues
            with self._lock:
                self._records.update({i.incident_id: StorageRecord(i, 'saved') for i in scan.incidents})
                self._config_hashes.update({i.incident_id: config_sha256(load_config(
                    self.outbox / i.incident_id / i.configuration.snapshot_path)) for i in scan.incidents})
            self._ready.set()
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    operation, key, at_utc, future = self.queue.get(timeout=.05)
                except Empty:
                    continue
                deliver = future.set_running_or_notify_cancel()
                if operation == 'read':
                    try:
                        folder = self.outbox / key
                        incident = _read_complete(folder)
                        media = (_contained(folder, incident.media.snapshot_path).read_bytes()
                                 if incident.media.snapshot_path else None)
                        if deliver:
                            future.set_result((incident, media))
                    except Exception as exc:
                        if deliver:
                            future.set_exception(StorageError(f'Read failed: {exc}'))
                    finally:
                        self.queue.task_done()
                    continue
                try:
                    if operation == 'save':
                        with self._lock:
                            request = self._requests[key]
                            self._records[key] = StorageRecord(request[0], 'saving')
                        saved = self._save(*request)
                    elif operation == 'clip':
                        saved = self._save_clip(key)
                    else:
                        folder = self.outbox / key
                        saved = _read_complete(folder)
                        if saved.acknowledgement.status != 'acknowledged':
                            saved = replace(saved, acknowledgement=Acknowledgement('acknowledged', at_utc))
                            self._atomic_write(_contained(folder, 'incident.json'), (incident_to_json(saved) + '\n').encode())
                    with self._lock:
                        self._records[key] = StorageRecord(saved, 'saved')
                        if operation == 'save':
                            request = self._requests.pop(key)
                            self._config_hashes[key] = config_sha256(request[1])
                            if request[2] is not None:
                                self._media_bytes -= request[2].nbytes
                                self.media_budget.release(request[2].nbytes)
                            request = None  # Do not retain the last snapshot on idle.
                        elif operation == 'clip':
                            self._clip_requests.pop(key, None)
                            self._clip_active.discard(key)
                    if deliver:
                        future.set_result(saved)
                except Exception as exc:
                    message = f'{operation} failed: {type(exc).__name__}: {exc}'
                    with self._lock:
                        current = self._records[key].incident
                        if operation == 'save':
                            current = replace(current, delivery=Delivery(status='save_failed'))
                        self._records[key] = StorageRecord(current, 'save_failed' if operation == 'save' else
                                                         'clip_failed' if operation == 'clip' else 'ack_failed', message)
                        if operation == 'clip':
                            self._clip_active.discard(key)
                    if deliver:
                        future.set_exception(StorageError(message))
                    self._fault(message)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self._startup_error = exc
            self._fault(f'Storage unavailable: {exc}')
        finally:
            self._ready.set()
            if lease:
                if locked and os.name == 'nt':
                    import msvcrt
                    lease.seek(0)
                    msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
                lease.close()
