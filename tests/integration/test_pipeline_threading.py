"""Real Qt owners/disk/fixture inference; injected readers model driver faults.

Controlled pose trajectories test orchestration, not detector accuracy. The
separate fixture test uses the actual local YOLO and ByteTrack implementations.
"""
from dataclasses import replace
import gc
from pathlib import Path
from threading import Event, get_ident
import time
import weakref

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from vertebrate.config import AppConfig
from vertebrate.contracts import FramePacket, PersonTrackKey, PoseObservation, SourceKind, TrackSample
from vertebrate.pipeline import HeadlessPipeline, ThreadedPipeline, DisplayResult, LatestResultSlot
from vertebrate.storage import StorageWorker, scan_outbox

ROOT = Path(__file__).resolve().parents[2]
APP = QCoreApplication.instance() or QCoreApplication([])


def until(predicate, timeout=10):
    end = time.monotonic() + timeout
    while not predicate():
        APP.processEvents()
        if time.monotonic() > end:
            pytest.fail('Qt operation did not finish within timeout')
        time.sleep(.002)
    APP.processEvents()


def completed(future, timeout=10):
    until(future.done, timeout)
    return future.result()


class Reader:
    fps = 10.
    frame_count = 70

    def __init__(self, source, *, delay=.001):
        self.index = 0
        self.delay = delay
        self.calls = []
        self.buffer = np.zeros((360, 640, 3), np.uint8)

    def open(self): self.calls.append(('open', get_ident()))
    def seek(self, index): self.index = index
    def read(self):
        self.calls.append(('read', get_ident()))
        time.sleep(self.delay)
        if self.index == self.frame_count:
            return None, None
        self.buffer.fill(self.index % 256)
        pts = self.index / self.fps
        self.index += 1
        return self.buffer, pts
    def release(self): self.calls.append(('release', get_ident()))


class EmptyPose:
    def infer(self, frame):
        return PoseObservation(frame.session_id, frame.sequence, frame.source_t_s, frame.width, frame.height,
                               np.empty((0, 4)), np.empty(0), np.empty((0, 17, 2)),
                               np.empty((0, 17)), np.empty((0, 17), bool))


class DirectTracker:
    def handle_event(self, event): pass
    def update(self, observation):
        return tuple(TrackSample(PersonTrackKey(observation.session_id, i, 1), observation.sequence,
            observation.source_t_s, box, float(observation.box_confidences[i]), observation.keypoints_xy[i],
            observation.keypoint_confidences[i], observation.keypoint_valid_mask[i], True)
            for i, box in enumerate(observation.boxes_xyxy))


def fast_pipeline(cfg, root):
    return HeadlessPipeline(cfg, root, pose=EmptyPose(), tracker=DirectTracker())


def configuration(tmp_path):
    cfg = AppConfig()
    return replace(cfg, runtime=replace(cfg.runtime, outbox_dir=str(tmp_path / 'outbox')))


@pytest.fixture
def coordinator(tmp_path):
    pipe = ThreadedPipeline(configuration(tmp_path), ROOT, pipeline_factory=fast_pipeline)
    yield pipe
    completed(pipe.stop())
    assert pipe.running_threads == 0


def test_real_fixture_four_qt_contexts_and_offline_local_inference(tmp_path):
    pipe = ThreadedPipeline(configuration(tmp_path), ROOT)
    try:
        completed(pipe.start(ROOT / 'tests/fixtures/videos/dev_empty_scene.avi'), 60)
        until(lambda: pipe.status.state in ('eof', 'error'), 60)
        assert pipe.status.state == 'eof', pipe.status
        assert pipe.status.frames == 100 and pipe.status.dropped_frames == 0
        assert len(set(pipe.owner_ids.values())) == 4
        assert pipe.owner_ids['gui'] == get_ident()
        assert pipe.running_threads == 3
        result = pipe.latest.get()
        assert result.frame.sequence == 99
        assert result.frame.source_t_s == pytest.approx(99 / 25)
        assert not result.frame.frame_bgr.flags.writeable
        assert all(t.owner.thread() == t.qt_thread for t in pipe._threads)
    finally:
        completed(pipe.stop(), 15)
    assert pipe.status.network_attempts == () and pipe.running_threads == 0


def test_twenty_start_stop_cycles_release_owner_handles_and_bounded_display(coordinator):
    readers, old_frames, old_owners = [], [], []
    def reader(source):
        value = Reader(source)
        value.frame_count = 5
        readers.append(value)
        return value
    for _ in range(20):
        completed(coordinator.start('injected', source_factory=reader))
        until(lambda: coordinator.status.state == 'eof')
        old_frames.append(weakref.ref(coordinator.latest.get().frame))
        old_owners.extend(weakref.ref(t.owner) for t in coordinator._threads)
        completed(coordinator.stop())
        assert coordinator.running_threads == 0
        assert coordinator.latest.get() is None
        assert len(coordinator.capture.queue) <= 2
    APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    assert all(ref() is None for ref in old_frames + old_owners)
    assert all([name for name, _ in r.calls].count('release') == 1 for r in readers)
    assert all(len({owner for _, owner in r.calls}) == 1 for r in readers)
    assert all(r.calls[0][1] != get_ident() for r in readers)


def test_display_result_owns_immutable_copy_and_single_slot():
    producer = np.zeros((2, 3, 3), np.uint8)
    packet = FramePacket('s', 0, SourceKind.VIDEO, 'local', 3, 2, producer, 0., 100)
    result = DisplayResult(packet, ())
    assert not np.shares_memory(result.frame.frame_bgr, packet.frame_bgr)
    producer.fill(200)
    copy = result.frame.copy_frame()
    copy.fill(100)
    assert not result.frame.frame_bgr.any()
    with pytest.raises(ValueError):
        result.frame.frame_bgr.setflags(write=True)
    slot = LatestResultSlot()
    old = weakref.ref(result)
    slot.set(result)
    del result
    for _ in range(100):
        slot.set(DisplayResult(packet, ()))
    gc.collect()
    assert old() is None


def test_pause_resume_freezes_frame_processing_and_source_timeline(coordinator):
    completed(coordinator.start('injected', source_factory=lambda s: Reader(s, delay=.005)))
    until(lambda: coordinator.latest.get() is not None)
    completed(coordinator.pause())
    result = coordinator.latest.get()
    source_index = coordinator.capture._reader.index
    for _ in range(30):
        APP.processEvents()
        time.sleep(.002)
    assert coordinator.latest.get() is result
    assert coordinator.capture._reader.index == source_index
    completed(coordinator.resume())
    until(lambda: coordinator.latest.get().frame.sequence > result.frame.sequence)
    resumed = coordinator.latest.get().frame
    assert resumed.session_id == result.frame.session_id
    assert resumed.source_t_s == pytest.approx(resumed.sequence / 10)


@pytest.mark.parametrize('operation,args', [('reset', ()), ('replay', ()), ('seek', (10,)),
                                          ('switch_source', ('other-injected',))])
def test_reset_fences_old_display_and_source_frames(coordinator, operation, args):
    completed(coordinator.start('injected', source_factory=lambda s: Reader(s, delay=.004)))
    until(lambda: coordinator.latest.get() is not None)
    completed(coordinator.pause())
    old_session = coordinator.latest.get().frame.session_id
    session = completed(getattr(coordinator, operation)(*args))
    assert session != old_session
    until(lambda: coordinator.latest.get() is not None)
    for _ in range(10):
        result = coordinator.latest.get()
        assert result.frame.session_id == session
        assert result.frame.source_t_s == pytest.approx(result.frame.sequence / 10)
        APP.processEvents()
        time.sleep(.002)


def test_live_pause_rejected_and_release_stays_on_capture_owner(coordinator):
    reader = Reader(0, delay=.003)
    completed(coordinator.start(0, source_factory=lambda s: reader))
    with pytest.raises(Exception, match='Pause requires'):
        completed(coordinator.pause())
    completed(coordinator.stop())
    assert {i for _, i in reader.calls} == {coordinator.owner_ids['capture']}


def test_slow_vision_file_backpressure_is_lossless(coordinator):
    seen = []
    class SlowPose(EmptyPose):
        def infer(self, frame):
            time.sleep(.003)
            seen.append(frame.sequence)
            return super().infer(frame)
    coordinator.pipeline_factory = lambda cfg, root: HeadlessPipeline(cfg, root, pose=SlowPose(), tracker=DirectTracker())
    completed(coordinator.start('injected', source_factory=lambda s: Reader(s, delay=0)))
    until(lambda: coordinator.status.state == 'eof')
    assert seen == list(range(70))
    assert coordinator.capture.queue.dropped_frames_count == 0


def test_live_slow_vision_drops_frames_but_keeps_queue_bounded(coordinator):
    class SlowPose(EmptyPose):
        def infer(self, frame):
            time.sleep(.02)
            return super().infer(frame)
    coordinator.pipeline_factory = lambda cfg, root: HeadlessPipeline(cfg, root, pose=SlowPose(), tracker=DirectTracker())
    completed(coordinator.start(0, source_factory=lambda s: Reader(s, delay=.001)))
    until(lambda: coordinator.capture.queue.dropped_frames_count > 2)
    assert len(coordinator.capture.queue) <= 2
    completed(coordinator.stop())


def test_blocked_reader_stop_times_out_without_blocking_gui(coordinator):
    entered, release = Event(), Event()
    class BlockedReader(Reader):
        def read(self):
            entered.set()
            release.wait(10)
            return super().read()
    reader = BlockedReader(0)
    try:
        completed(coordinator.start(0, source_factory=lambda s: reader))
        until(entered.is_set)
        before = time.monotonic()
        future = coordinator.stop(timeout_s=.1)
        assert time.monotonic() - before < .1
        with pytest.raises(TimeoutError, match='blocked'):
            completed(future)
        assert coordinator.status.state == 'stop_timeout'
        assert coordinator.running_threads == 1
        assert not any(name == 'release' for name, _ in reader.calls)
    finally:
        release.set()
        until(lambda: coordinator.running_threads == 0)
    assert reader.calls[-1] == ('release', coordinator.owner_ids['capture'])


def test_stop_interrupts_blocked_start_and_control_queue_is_bounded(coordinator):
    entered, release = Event(), Event()
    class SlowOpen(Reader):
        def open(self):
            entered.set()
            release.wait(10)
            super().open()
    start = coordinator.start('injected', source_factory=SlowOpen)
    until(entered.is_set)
    futures = [coordinator.pause() for _ in range(20)]
    assert coordinator._commands.qsize() <= 8
    assert any(f.done() and f.exception() for f in futures)
    stop = coordinator.stop(timeout_s=2)
    release.set()
    completed(stop)
    assert start.done() and start.exception()
    assert all(f.done() for f in futures)


class FallPose:
    """Deterministic injected pose, exercising actual feature/FSM/incident code."""
    def __init__(self, people=1): self.people = people
    def infer(self, frame):
        p = min(1., max(0., (frame.source_t_s - 1) / .6))
        xy = np.full((17, 2), 100.)
        for a, b, center in ((5, 6, (100 - 80*p, 150 + 200*p)), (11, 12, (100, 250 + 100*p)),
                             (13, 14, (100, 275 + 75*p)), (15, 16, (100, 300 + 50*p))):
            xy[a], xy[b] = (center[0] - 10, center[1]), (center[0] + 10, center[1])
        n = self.people
        return PoseObservation(frame.session_id, frame.sequence, frame.source_t_s, frame.width, frame.height,
            np.tile([0., 100., 80 + 140*p, 300 - 100*p], (n, 1)), np.full(n, .9),
            np.tile(xy, (n, 1, 1)), np.ones((n, 17)), np.ones((n, 17), bool))


def test_confirmed_incident_crosses_to_single_writer_and_stop_drains(tmp_path):
    cfg = configuration(tmp_path)
    pipe = ThreadedPipeline(cfg, ROOT, pipeline_factory=lambda c, r: HeadlessPipeline(
        c, r, pose=FallPose(), tracker=DirectTracker()))
    try:
        completed(pipe.start('injected', source_factory=Reader))
        until(lambda: pipe.status.state == 'eof')
        completed(pipe.stop())
        inbox = scan_outbox(tmp_path / 'outbox')
        assert len(inbox.incidents) == 1 and inbox.issues == ()
        incident = inbox.incidents[0]
        assert incident.delivery.status == 'saved'
        assert incident.timing.confirmed_source_s == 5.6
        assert incident.media.snapshot_path == 'snapshot.jpg'
        assert pipe.status.network_attempts == ()
    finally:
        completed(pipe.stop())


def test_slow_disk_queue_overflow_preserves_alerts_and_pauses_both_workers(tmp_path):
    entered, release = Event(), Event()
    class SlowStorage(StorageWorker):
        def _save(self, *args):
            entered.set()
            release.wait(10)
            return super()._save(*args)
    pipe = ThreadedPipeline(configuration(tmp_path), ROOT, storage_factory=SlowStorage,
        pipeline_factory=lambda c, r: HeadlessPipeline(c, r, pose=FallPose(11), tracker=DirectTracker()))
    try:
        completed(pipe.start('injected', source_factory=Reader))
        until(lambda: pipe.status.state == 'storage_fault')
        until(entered.is_set)
        assert pipe.storage.queue.maxsize == 8 and pipe.storage.queue.qsize() <= 8
        until(lambda: len(pipe.storage.unsaved_incident_ids) == 11)
        assert 'retained' in pipe.status.message
        with pytest.raises(Exception, match='retry retained'):
            completed(pipe.resume())
        # Allow a read/processing boundary already in flight to complete.
        time.sleep(.03)
        count = pipe.capture._reader.index
        result = pipe.latest.get()
        time.sleep(.05)
        assert pipe.capture._reader.index == count and pipe.latest.get() is result
        failed = [i for i in pipe.storage.unsaved_incident_ids if pipe.storage.record(i).status == 'save_failed']
        assert failed
        release.set()
        until(lambda: all(pipe.storage.record(i).status == 'saved' for i in
                         pipe.storage.unsaved_incident_ids if i not in failed))
        for key in failed:
            completed(pipe.storage.retry(key))
        completed(pipe.resume())
        until(lambda: pipe.status.state == 'eof')
        completed(pipe.stop())
        assert len(scan_outbox(tmp_path / 'outbox').incidents) == 11
    finally:
        release.set()
        completed(pipe.stop())


def test_capture_failure_and_storage_startup_failure_are_visible_and_cleanup(coordinator, tmp_path):
    class BadReader(Reader):
        def open(self): raise OSError('injected unavailable camera')
    with pytest.raises(Exception, match='unavailable camera'):
        completed(coordinator.start('injected', source_factory=BadReader))
    until(lambda: coordinator.running_threads == 0)
    assert 'unavailable camera' in coordinator.status.message
    obstruction = tmp_path / 'file-not-directory'
    obstruction.write_bytes(b'x')
    coordinator.config = replace(coordinator.config, runtime=replace(coordinator.config.runtime, outbox_dir=str(obstruction)))
    with pytest.raises(Exception):
        completed(coordinator.start('injected', source_factory=Reader))
    until(lambda: coordinator.running_threads == 0)
    assert coordinator.status.state == 'error'


def test_second_pipeline_cannot_overlap_global_offline_guards(coordinator, tmp_path):
    completed(coordinator.start('injected', source_factory=Reader))
    second = ThreadedPipeline(configuration(tmp_path / 'second'), ROOT, pipeline_factory=fast_pipeline)
    with pytest.raises(RuntimeError, match='Another'):
        second.start('injected', source_factory=Reader)


def test_storage_failure_retained_across_stop_restart_and_retry(tmp_path):
    fail = Event()
    fail.set()
    class FailingStorage(StorageWorker):
        def _atomic_write(self, destination, data):
            if destination.name == 'incident.json' and fail.is_set():
                raise OSError('injected metadata disk failure')
            return super()._atomic_write(destination, data)
    pipe = ThreadedPipeline(configuration(tmp_path), ROOT, storage_factory=FailingStorage,
        pipeline_factory=lambda c, r: HeadlessPipeline(c, r, pose=FallPose(), tracker=DirectTracker()))
    try:
        completed(pipe.start('injected', source_factory=Reader))
        until(lambda: pipe.status.state == 'storage_fault')
        ids = pipe.storage.unsaved_incident_ids
        assert len(ids) == 1
        writer = pipe.storage
        completed(pipe.stop())
        assert writer.record(ids[0]).status == 'save_failed'
        fail.clear()
        completed(pipe.start('injected', source_factory=Reader))
        assert pipe.storage is writer
        assert pipe.status.state == 'storage_fault' and pipe.capture.sequence == 0
        completed(writer.retry(ids[0]))
        assert writer.record(ids[0]).status == 'saved'
        completed(pipe.resume())
        until(lambda: pipe.status.frames > 1)
        completed(pipe.stop())
        assert len(scan_outbox(tmp_path / 'outbox').incidents) == 1
    finally:
        completed(pipe.stop())


def test_stop_timeout_on_slow_disk_preserves_pending_commit(tmp_path):
    entered, release = Event(), Event()
    class BlockedStorage(StorageWorker):
        def _save(self, *args):
            entered.set()
            release.wait(10)
            return super()._save(*args)
    pipe = ThreadedPipeline(configuration(tmp_path), ROOT, storage_factory=BlockedStorage,
        pipeline_factory=lambda c, r: HeadlessPipeline(c, r, pose=FallPose(), tracker=DirectTracker()))
    try:
        completed(pipe.start('injected', source_factory=Reader))
        until(entered.is_set)
        before = time.monotonic()
        future = pipe.stop(timeout_s=.1)
        assert time.monotonic() - before < .1
        with pytest.raises(TimeoutError):
            completed(future)
        assert pipe.running_threads == 1
        assert pipe.storage.unsaved_incident_ids
        release.set()
        until(lambda: pipe.running_threads == 0)
        assert len(scan_outbox(tmp_path / 'outbox').incidents) == 1
    finally:
        release.set()
        completed(pipe.stop())


def test_vision_failure_closes_source_and_returns_visible_error(coordinator):
    class BadPose:
        def infer(self, frame): raise RuntimeError('injected inference failure')
    coordinator.pipeline_factory = lambda c, r: HeadlessPipeline(c, r, pose=BadPose(), tracker=DirectTracker())
    readers = []
    def reader(source):
        readers.append(Reader(source))
        return readers[-1]
    future = coordinator.start('injected', source_factory=reader)
    until(lambda: coordinator.status.state == 'error')
    until(lambda: coordinator.running_threads == 0)
    assert 'injected inference failure' in coordinator.status.message
    assert future.done()
    assert readers[0].calls[-1] == ('release', coordinator.owner_ids['capture'])


def test_stop_during_inflight_vision_completes_and_releases_model_in_owner(coordinator):
    entered, release = Event(), Event()
    deleted = []
    class BlockingPose(EmptyPose):
        def infer(self, frame):
            entered.set()
            release.wait(10)
            return super().infer(frame)
        def __del__(self): deleted.append(get_ident())
    coordinator.pipeline_factory = lambda c, r: HeadlessPipeline(c, r, pose=BlockingPose(), tracker=DirectTracker())
    try:
        completed(coordinator.start('injected', source_factory=Reader))
        until(entered.is_set)
        future = coordinator.stop()
        assert not future.done()
        release.set()
        completed(future)
        assert deleted == [coordinator.owner_ids['vision']]
        assert coordinator.status.state == 'stopped'
    finally:
        release.set()


def test_commands_require_qt_main_thread(coordinator):
    from threading import Thread
    errors = []
    def call():
        try:
            coordinator.pause()
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = Thread(target=call)
    thread.start()
    thread.join(1)
    assert errors == ['ThreadedPipeline requires the Qt main thread and application']
