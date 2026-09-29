"""P1A acceptance: deterministic clocks, queues, lifecycle, labels, and real clips.

Injected readers test camera/control behavior without claiming hardware tests.
Real AVI decode, CFR verification, fixture regeneration and worker integration
are tested separately, without substituting mocks for the integration paths.
"""
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Empty, Full
from threading import Event, Thread, get_ident
import time

import numpy as np
import pytest

from vertebrate.capture import (BoundedFrameQueue, CaptureSessionManager, CaptureWorker,
                                OpenCVFrameSource)
from vertebrate.clocks import LiveSourceClock, VideoFileClock
from vertebrate.contracts import (CaptureError, EndOfStreamEvent, FramePacket,
    ManifestValidationError, PersonTrackKey, PoseObservation, SessionResetEvent,
    SourceKind, SourceTimingError, TimingMode, TrackSample)
from vertebrate.evaluation.labels import (ClipManifestRow, DatasetManifest, FrameTimingSpec,
    PersonAnnotation, TimeInterval, load_dataset_manifest, save_dataset_manifest)

ROOT = Path(__file__).resolve().parents[2]
UTC = '2020-01-02T03:04:05Z'


class FakeTime:
    ns = 1_000_000_000
    utc = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    def monotonic_ns(self): return self.ns
    def utc_now(self): return self.utc
    def advance(self, seconds): self.ns += int(seconds * 1e9)


def packet(sequence=0, kind=SourceKind.LIVE, session='s1', **overrides):
    values = dict(session_id=session, sequence=sequence, source_kind=kind,
                  source_name='fixture', width=8, height=6,
                  frame_bgr=np.zeros((6, 8, 3), dtype=np.uint8), source_t_s=sequence / 25,
                  acquired_monotonic_ns=1_000_000_000 + sequence * 40_000_000,
                  timing_mode=TimingMode.LIVE_MONOTONIC if kind == SourceKind.LIVE else TimingMode.PTS_VERIFIED)
    return FramePacket(**(values | overrides))


def test_live_progression_and_utc():
    now = FakeTime()
    clock = LiveSourceClock(now.monotonic_ns, now.utc_now)
    assert clock.start_session('live-1') == 'live-1'
    assert clock.tick() == (0, 0.0, now.ns)
    now.advance(.5)
    assert clock.tick() == (1, .5, now.ns)
    now.utc = datetime(2030, 1, 1, tzinfo=timezone.utc)  # wall clock adjustment is irrelevant
    assert clock.estimate_event_utc(.5) == '2020-01-02T03:04:05.500000Z'
    assert clock.tick()[1] == .5  # equal monotonic timestamps are allowed live
    now.advance(-.1)
    with pytest.raises(SourceTimingError, match='ffmpeg'):
        clock.tick()
    clock.start_session('live-2')
    assert clock.tick()[:2] == (0, 0)


def test_live_requires_start_and_aware_utc():
    with pytest.raises(CaptureError): LiveSourceClock().tick()
    with pytest.raises(CaptureError): LiveSourceClock().estimate_event_utc(0)
    with pytest.raises(ValueError):
        LiveSourceClock(utc_now_fn=lambda: datetime(2020, 1, 1)).start_session()


@pytest.mark.parametrize('fps', [None, 0, -1, float('nan'), float('inf'), -float('inf'), 241, True, '25'])
def test_bad_fps_has_actionable_conversion(fps):
    with pytest.raises(SourceTimingError, match='ffmpeg -i "input.mp4" -vf fps=30'):
        VideoFileClock(fps)


@pytest.mark.parametrize('fps', [20, 25., 30., 240.])
def test_cfr_pts(fps):
    clock = VideoFileClock(fps)
    for index in range(20):
        value, mode = clock.compute_frame_time(index, index / fps)
        assert value == pytest.approx(index / fps)
        assert mode == TimingMode.PTS_VERIFIED
    assert clock.estimate_event_utc(1) is None


@pytest.mark.parametrize('pts', [None, 0.0])
def test_zero_pts_fallback_latches(pts):
    clock = VideoFileClock(25)
    assert clock.compute_frame_time(0, 0) == (0, TimingMode.PTS_VERIFIED)
    for i in range(1, 5):
        assert clock.compute_frame_time(i, pts) == (i / 25, TimingMode.FRAME_INDEX_FPS_FALLBACK)
    assert clock.compute_frame_time(5, .2) == (.2, TimingMode.FRAME_INDEX_FPS_FALLBACK)
    # Even after fallback, nonzero bad PTS must not be silently ignored.
    with pytest.raises(SourceTimingError, match='ffmpeg'):
        clock.compute_frame_time(6, .8)


def test_missing_startup_pts_and_offset():
    assert VideoFileClock(25).compute_frame_time(0, None) == (0, TimingMode.FRAME_INDEX_FPS_FALLBACK)
    assert VideoFileClock(25).compute_frame_time(0, .001) == (0, TimingMode.PTS_VERIFIED)
    with pytest.raises(SourceTimingError): VideoFileClock(25).compute_frame_time(0, .2)


@pytest.mark.parametrize('bad', [-.001, -1, float('nan'), float('inf')])
@pytest.mark.parametrize('index', [0, 1])
def test_invalid_pts(index, bad):
    clock = VideoFileClock(25)
    if index: clock.compute_frame_time(0, 0)
    with pytest.raises(SourceTimingError, match='ffmpeg'):
        clock.compute_frame_time(index, bad)


@pytest.mark.parametrize('pts', [.04, .03, .2])
def test_backward_frozen_and_vfr_pts(pts):
    clock = VideoFileClock(25)
    clock.compute_frame_time(0, 0)
    clock.compute_frame_time(1, .04)
    with pytest.raises(SourceTimingError, match='ffmpeg'):
        clock.compute_frame_time(2, pts)


def test_vfr_jitter_and_cumulative_drift():
    clock = VideoFileClock(25)
    clock.compute_frame_time(0, 0)
    clock.compute_frame_time(1, .025)
    with pytest.raises(SourceTimingError, match='jitter'):
        clock.compute_frame_time(2, .095)  # individually near CFR, delta is not
    clock = VideoFileClock(25)
    for index in range(3): clock.compute_frame_time(index, index * .05)
    with pytest.raises(SourceTimingError): clock.compute_frame_time(3, .15)


def test_pause_twenty_seconds_and_historical_utc():
    now = FakeTime()
    clock = VideoFileClock(25, UTC)
    clock.compute_frame_time(0, 0)
    clock.pause()
    now.advance(20)
    with pytest.raises(CaptureError, match='paused'):
        clock.compute_frame_time(1, .04)
    assert clock.last_source_t_s == 0
    clock.resume()
    assert clock.compute_frame_time(1, .04)[0] == .04
    assert clock.estimate_event_utc(2) == '2020-01-02T03:04:07Z'
    assert VideoFileClock(25).estimate_event_utc(100) is None
    with pytest.raises(SourceTimingError): VideoFileClock(25, '2020-01-01T00:00:00')
    with pytest.raises(SourceTimingError): clock.compute_frame_time(3, .12)
    clock.reset()
    assert clock.compute_frame_time(0, 0)[0] == 0


def test_frame_immutable_and_copy_isolated():
    source = np.zeros((6, 8, 3), np.uint8)
    frame = packet(frame_bgr=source)
    source[:] = 255
    assert not frame.frame_bgr.any() and not frame.frame_bgr.flags.writeable
    with pytest.raises(ValueError): frame.frame_bgr[0, 0, 0] = 1
    with pytest.raises(ValueError): frame.frame_bgr.setflags(write=True)
    copied = frame.copy_frame()
    assert copied.flags.writeable
    copied[:] = 20
    assert not frame.frame_bgr.any()
    with pytest.raises(FrozenInstanceError): frame.sequence = 4


@pytest.mark.parametrize('overrides', [
    {'sequence': -1}, {'width': 0}, {'height': True}, {'session_id': ''},
    {'source_t_s': float('nan')}, {'acquired_monotonic_ns': 0}, {'source_kind': 'live'},
    {'frame_bgr': np.zeros((6, 8, 3), np.float32)}, {'frame_bgr': np.zeros((8, 6, 3), np.uint8)},
    {'recorded_start_utc': 'yesterday'}, {'session_start_monotonic_ns': 2_000_000_000},
])
def test_frame_validation(overrides):
    with pytest.raises(ValueError): packet(**overrides)


def pose(n=1, **overrides):
    values = dict(session_id='s1', sequence=0, source_t_s=0., width=640, height=360,
        boxes_xyxy=np.zeros((n, 4), np.float32), box_confidences=np.ones(n, np.float32),
        keypoints_xy=np.zeros((n, 17, 2), np.float32), keypoint_confidences=np.ones((n, 17), np.float32),
        keypoint_valid_mask=np.ones((n, 17), bool))
    return PoseObservation(**(values | overrides))


@pytest.mark.parametrize('n', [0, 1, 2])
def test_pose_contract(n):
    observation = pose(n)
    for name in ('boxes_xyxy', 'box_confidences', 'keypoints_xy', 'keypoint_confidences', 'keypoint_valid_mask'):
        assert not getattr(observation, name).flags.writeable
    assert observation.keypoints_xy.shape == (n, 17, 2)


@pytest.mark.parametrize('overrides', [
    {'boxes_xyxy': np.array([[5, 0, 1, 2]], np.float32)},
    {'box_confidences': np.array([1.1])}, {'keypoints_xy': np.zeros((2, 17, 2))},
    {'keypoint_confidences': np.full((1, 17), np.nan)},
    {'keypoint_valid_mask': np.ones((1, 17), np.float32)},
])
def test_invalid_pose(overrides):
    with pytest.raises(ValueError): pose(**overrides)


def sample(**overrides):
    obs = pose()
    values = dict(person_key=PersonTrackKey('s1', 0, 1), sequence=0, source_t_s=0.,
        bbox_xyxy=obs.boxes_xyxy[0], box_confidence=.9, keypoints_xy=obs.keypoints_xy[0],
        keypoint_confidences=obs.keypoint_confidences[0], keypoint_valid_mask=obs.keypoint_valid_mask[0],
        observation_valid=True)
    return TrackSample(**(values | overrides))


def test_track_contract_and_predicted_only():
    assert not sample().bbox_xyxy.flags.writeable
    for overrides in ({'matched_detection': False}, {'predicted_only': True}):
        with pytest.raises(ValueError, match='observation_valid'): sample(**overrides)
        assert not sample(observation_valid=False, **overrides).observation_valid
    for args in (('', 0, 1), ('s', -1, 1), ('s', 1, 0), ('s', True, 1)):
        with pytest.raises(ValueError): PersonTrackKey(*args)


def test_live_drop_oldest():
    queue = BoundedFrameQueue(SourceKind.LIVE)
    for i in range(10): assert queue.put(packet(i), timeout_s=0)
    assert len(queue) == 2
    assert [queue.get().sequence, queue.get().sequence] == [8, 9]
    assert queue.dropped_frames_count == 8


def test_video_backpressure_lossless():
    queue = BoundedFrameQueue(SourceKind.VIDEO)
    waiting, finished = Event(), Event()
    errors = []
    def produce():
        try:
            for i in range(10):
                if i == 2: waiting.set()
                assert queue.put(packet(i, SourceKind.VIDEO), timeout_s=2)
            finished.set()
        except BaseException as exc: errors.append(exc)
    worker = Thread(target=produce)
    worker.start()
    assert waiting.wait(1) and not finished.is_set()
    assert len(queue) == 2
    received = [queue.get(2).sequence for _ in range(10)]
    worker.join(2)
    assert not worker.is_alive() and not errors
    assert received == list(range(10)) and queue.dropped_frames_count == 0


def reset_event(previous=None, new='s1', kind=SourceKind.LIVE):
    return SessionResetEvent(previous, new, 'test', kind, 'fixture')


def test_controls_never_dropped_and_finite_backpressure():
    queue = BoundedFrameQueue(SourceKind.LIVE)
    event = reset_event()
    queue.put(event)
    for i in range(10): queue.put(packet(i), 0)
    assert queue.get() == event
    assert queue.get().sequence == 9
    queue.put(reset_event())
    eof = EndOfStreamEvent('s1', 'fixture', 0, None, None)
    queue.put(eof)
    with pytest.raises(Full): queue.put(packet(), 0)
    with pytest.raises(Full): queue.reset(reset_event('s1', 's2'))
    assert queue.get().new_session_id == 's1'
    assert queue.get() == eof


def test_cancel_stop_and_stale_producer_fence():
    queue = BoundedFrameQueue(SourceKind.VIDEO)
    queue.reset(reset_event(kind=SourceKind.VIDEO))
    queue.get()
    for i in range(2): queue.put(packet(i, SourceKind.VIDEO))
    started, result = Event(), []
    def pending():
        started.set()
        result.append(queue.put(packet(2, SourceKind.VIDEO), 1))
    thread = Thread(target=pending)
    thread.start()
    assert started.wait(1)
    queue.reset(reset_event('s1', 's2', SourceKind.VIDEO))
    thread.join(2)
    assert result == [False] and queue.get().new_session_id == 's2'
    assert queue.reset_discarded_frames_count == 2
    cancel = Event()
    cancel.set()
    assert not queue.put(packet(0, SourceKind.VIDEO, 's2'), cancel=cancel)
    queue.close()
    assert not queue.put(packet(0, SourceKind.VIDEO, 's2'))
    with pytest.raises(Empty): queue.get(0)


class FakeReader:
    fps = 25.
    frame_count = 10
    def __init__(self, source):
        self.source = source
        self.index = 0
        self.calls = []
        self.released = False
    def open(self): self.calls.append(('open', get_ident()))
    def read(self):
        self.calls.append(('read', get_ident()))
        if self.index == self.frame_count: return None, None
        index = self.index
        self.index += 1
        return np.full((6, 8, 3), index, np.uint8), index / self.fps
    def seek(self, index):
        self.calls.append(('seek', get_ident()))
        if index >= self.frame_count: raise CaptureError('Seek out of bounds')
        self.index = index
    def release(self):
        self.calls.append(('release', get_ident()))
        self.released = True


def manager(source='fixture.avi', **kwargs):
    return CaptureSessionManager(source, source_factory=FakeReader, **kwargs)


def test_capture_pause_no_reads_twenty_seconds():
    now = FakeTime()
    controller = manager(monotonic_ns_fn=now.monotonic_ns, utc_now_fn=now.utc_now)
    sid = controller.start()
    assert isinstance(controller.queue.get(), SessionResetEvent)
    controller.pump()
    first = controller.queue.get()
    controller.pause()
    calls = list(controller._reader.calls)
    now.advance(20)
    for _ in range(5): assert controller.pump() is None
    assert controller._reader.calls == calls
    assert controller.last_source_t_s == first.source_t_s == 0
    controller.resume()
    controller.pump()
    second = controller.queue.get()
    assert (second.session_id, second.sequence, second.source_t_s) == (sid, 1, .04)
    assert second.acquired_monotonic_ns - first.acquired_monotonic_ns == 20_000_000_000
    assert controller.estimate_event_utc(.04) is None
    controller.stop()


@pytest.mark.parametrize('operation,args', [('seek', (4,)), ('replay', ()), ('restart', ()),
                                         ('switch_source', ('another.avi',)), ('switch_source', (1,))])
def test_reset_fences_sequence_and_clock(operation, args):
    now = FakeTime()
    controller = manager(monotonic_ns_fn=now.monotonic_ns, utc_now_fn=now.utc_now, recorded_start_utc=UTC)
    old_id = controller.start()
    controller.queue.get()
    controller.pump()
    controller.pump()  # old unprocessed frames intentionally remain queued
    old_reader = controller._reader
    new_id = getattr(controller, operation)(*args)
    assert new_id != old_id and controller.sequence == 0
    assert old_reader.released
    event = controller.queue.get()
    assert event.previous_session_id == old_id and event.new_session_id == new_id
    controller.pump()
    frame = controller.queue.get()
    assert (frame.session_id, frame.sequence, frame.source_t_s) == (new_id, 0, 0.)
    if operation == 'seek':
        assert frame.source_frame_index == 4 and frame.source_time_offset_s == .16
        assert controller.estimate_event_utc(0) == '2020-01-02T03:04:05.160000Z'
    controller.stop()


def test_camera_switch_and_live_pause_rejected():
    controller = manager(0)
    first = controller.start()
    controller.queue.get()
    with pytest.raises(CaptureError): controller.pause()
    with pytest.raises(CaptureError): controller.seek(1)
    second = controller.switch_source(1)
    assert first != second and controller.sequence == 0
    assert controller.queue.get().source_kind == SourceKind.LIVE
    controller.stop()


def test_eof_exactly_once_no_synthetic_time():
    controller = manager()
    controller.start()
    controller.queue.get()
    reader = controller._reader
    frames = []
    while controller.state == 'running':
        controller.pump()
        item = controller.queue.get()
        if isinstance(item, FramePacket): frames.append(item)
        else: eof = item
    assert len(frames) == 10
    assert eof == EndOfStreamEvent(controller.session_id, 'fixture.avi', 10, 9, .36)
    assert reader.released and controller.state == 'eof'
    for _ in range(10): assert controller.pump() is None
    assert controller.last_source_t_s == .36
    with pytest.raises(Empty): controller.queue.get(0)
    old = controller.session_id
    controller.replay()
    assert controller.queue.get().previous_session_id == old
    controller.stop()


def test_pending_frame_not_lost_or_read_twice_under_backpressure():
    controller = manager()
    controller.start()
    controller.queue.get()
    controller.pump()
    controller.pump()
    assert controller.pump(0) is None
    count = len(controller._reader.calls)
    for _ in range(3): assert controller.pump(0) is None
    assert len(controller._reader.calls) == count
    assert controller.queue.get().sequence == 0
    assert controller.pump(0).sequence == 2
    assert [controller.queue.get().sequence, controller.queue.get().sequence] == [1, 2]
    controller.stop()


@pytest.mark.parametrize('failure', ['bad_fps', 'bad_pts', 'early_eof', 'read_error', 'open_error'])
def test_capture_failure_and_release(failure):
    readers = []
    class BrokenReader(FakeReader):
        def open(self):
            super().open()
            if failure == 'open_error': raise OSError('injected open failure')
            if failure == 'bad_fps': self.fps = 0
        def read(self):
            if failure == 'early_eof': return None, None
            if failure == 'read_error': raise OSError('injected read failure')
            frame, pts = super().read()
            return frame, float('nan') if failure == 'bad_pts' else pts
    def factory(source):
        reader = BrokenReader(source)
        readers.append(reader)
        return reader
    controller = CaptureSessionManager('test.avi', source_factory=factory)
    with pytest.raises((CaptureError, SourceTimingError)):
        controller.start()
        controller.queue.get()
        controller.pump()
    assert readers[0].released and controller.state == 'error'


def test_worker_owner_thread_pause_seek_and_stop():
    readers = []
    def factory(source):
        reader = FakeReader(source)
        readers.append(reader)
        return reader
    worker = CaptureWorker('test.avi', source_factory=factory)
    try:
        old = worker.start()
        assert isinstance(worker.queue.get(2), SessionResetEvent)
        worker.pause()
        count = len(readers[-1].calls)
        assert worker.state == 'paused'
        # Acknowledged pause is the synchronization boundary, not a sleep guess.
        assert len(readers[-1].calls) == count
        new = worker.seek(3)
        assert old != new
        event = worker.queue.get(2)
        assert isinstance(event, SessionResetEvent)
        first = worker.queue.get(2)
        assert (first.session_id, first.sequence, first.source_t_s) == (new, 0, 0)
    finally:
        worker.stop()
    assert all(r.released for r in readers)
    owners = {tid for reader in readers for _, tid in reader.calls}
    assert len(owners) == 1 and get_ident() not in owners
    assert not worker._thread.is_alive()


def manifest_dict():
    return asdict(load_dataset_manifest(ROOT / 'data/manifests/dev.json'))


def write_manifest(tmp_path, data):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_dev_manifest_hashes_and_roundtrip(tmp_path):
    manifest = load_dataset_manifest(ROOT / 'data/manifests/dev.json', verify_files=True, root_dir=ROOT)
    assert len(manifest.clips) == 3
    assert [c.people_count for c in manifest.clips] == [0, 1, 2]
    assert all(c.recorded_start_utc is None for c in manifest.clips)
    target = tmp_path / 'saved.json'
    save_dataset_manifest(manifest, target)
    assert load_dataset_manifest(target, True, ROOT) == manifest
    with pytest.raises(FrozenInstanceError): manifest.partition = 'holdout'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_invalid_intervals(bad):
    with pytest.raises(ManifestValidationError): TimeInterval(bad, 1)
    with pytest.raises(ManifestValidationError): TimeInterval(0, bad)


def test_interval_order_and_manifest_types():
    with pytest.raises(ManifestValidationError): TimeInterval(2, 1)
    with pytest.raises(ManifestValidationError): FrameTimingSpec(0, 10, 1, True)
    with pytest.raises(ManifestValidationError): FrameTimingSpec(25, 0, 1, True)
    with pytest.raises(ManifestValidationError): FrameTimingSpec(25, 25, 10, True)
    with pytest.raises(ManifestValidationError): FrameTimingSpec(25, 25, 1, 'true')
    with pytest.raises(ManifestValidationError): PersonAnnotation('p', 'yes')
    with pytest.raises(ManifestValidationError): DatasetManifest('2.0', 'dev', ())
    with pytest.raises(ManifestValidationError): DatasetManifest('1.0', 'train', ())


@pytest.mark.parametrize('field,value', [
    ('sha256', 'xyz'), ('sha256', 'A' * 64), ('people_count', -1),
    ('relative_path', '../escape.avi'), ('relative_path', 'https://example.org/a.mp4'),
    ('eligibility', 'unknown'), ('recorded_start_utc', '2020-01-01'),
])
def test_bad_manifest_row(tmp_path, field, value):
    data = manifest_dict()
    data['clips'][0][field] = value
    with pytest.raises(ManifestValidationError): load_dataset_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize('field', ['clip_id', 'sha256', 'subject_id', 'frame_timing', 'recorded_start_utc', 'exclusion_reason'])
def test_missing_required_manifest_keys(tmp_path, field):
    data = manifest_dict()
    del data['clips'][0][field]
    with pytest.raises(ManifestValidationError): load_dataset_manifest(write_manifest(tmp_path, data))


def test_supported_fall_order_and_duration(tmp_path):
    data = manifest_dict()
    row = data['clips'][1]
    row['eligibility'] = 'supported'
    row['exclusion_reason'] = None
    assert load_dataset_manifest(write_manifest(tmp_path, data)).clips[1].eligibility == 'supported'
    person = row['person_annotations'][0]
    for change in (
        {'fall_onset_interval_s': None},
        {'down_posture_interval_s': {'start_s': .1, 'end_s': .2}},
        {'stillness_eligible_interval_s': {'start_s': 1.6, 'end_s': 5}},
        {'visibility_gaps_s': [{'start_s': 3, 'end_s': 2}]},
    ):
        original = dict(person)
        person.update(change)
        with pytest.raises(ManifestValidationError): load_dataset_manifest(write_manifest(tmp_path, data))
        person.clear()
        person.update(original)


def test_manifest_missing_and_corrupt_file(tmp_path):
    data = manifest_dict()
    data['clips'][0]['relative_path'] = 'missing.avi'
    with pytest.raises(ManifestValidationError, match='missing'):
        load_dataset_manifest(write_manifest(tmp_path, data), True, tmp_path)
    (tmp_path / 'missing.avi').write_bytes(b'corrupted')
    with pytest.raises(ManifestValidationError, match='SHA-256'):
        load_dataset_manifest(write_manifest(tmp_path, data), True, tmp_path)


@pytest.mark.parametrize('payload', ['{}', '[]', '{broken', '{"clips":[],"clips":[]}',
    '{"schema_version":"1.0","partition":"dev","clips":[],"unknown":1}',
    '{"schema_version":"1.0","partition":"dev","clips":[NaN]}'])
def test_strict_manifest_json(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(payload)
    with pytest.raises(ManifestValidationError): load_dataset_manifest(path)


@pytest.mark.parametrize('clip_index', [0, 1, 2])
def test_real_cfr_file_capture_to_eof(clip_index):
    manifest = load_dataset_manifest(ROOT / 'data/manifests/dev.json', True, ROOT)
    row = manifest.clips[clip_index]
    worker = CaptureWorker(ROOT / row.relative_path)
    received = []
    try:
        worker.start()
        assert isinstance(worker.queue.get(2), SessionResetEvent)
        while True:
            item = worker.queue.get(2)
            if isinstance(item, EndOfStreamEvent):
                eof = item
                break
            received.append((item.sequence, item.source_t_s, item.recorded_start_utc))
            assert item.frame_bgr.shape == (360, 640, 3)
        assert worker.error is None
        assert len(received) == row.frame_timing.frame_count == 100
        assert [r[0] for r in received] == list(range(100))
        assert [r[1] for r in received] == pytest.approx([i / 25 for i in range(100)])
        assert all(r[2] is None for r in received)
        assert eof.last_source_t_s == pytest.approx(3.96)
        assert worker.queue.dropped_frames_count == 0
    finally:
        worker.stop()


def test_fixture_regeneration_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('generate_dev', ROOT / 'tools/generate_dev_fixtures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.generate(tmp_path)
    expected = load_dataset_manifest(ROOT / 'data/manifests/dev.json')
    assert [c.sha256 for c in generated.clips] == [c.sha256 for c in expected.clips]
    assert load_dataset_manifest(tmp_path / 'data/manifests/dev.json', True, tmp_path) == generated


def test_local_sources_only(tmp_path):
    with pytest.raises(CaptureError): OpenCVFrameSource(tmp_path / 'absent.avi').open()
    with pytest.raises(CaptureError): OpenCVFrameSource('https://example.com/movie.mp4').open()


def test_live_drop_counter_resets_when_switching_to_video():
    queue = BoundedFrameQueue(SourceKind.LIVE)
    for i in range(10): queue.put(packet(i))
    assert queue.dropped_frames_count == 8
    queue.reset(reset_event('s1', 's2', SourceKind.VIDEO))
    assert queue.source_kind == SourceKind.VIDEO and queue.dropped_frames_count == 0
    assert queue.get().new_session_id == 's2'


def test_capture_zero_pts_fallback():
    class ZeroPTS(FakeReader):
        def read(self):
            frame, pts = super().read()
            return frame, 0.0 if frame is not None else None
    controller = CaptureSessionManager('fixture.avi', source_factory=ZeroPTS)
    controller.start()
    controller.queue.get()
    frames = []
    for _ in range(10):
        controller.pump()
        frames.append(controller.queue.get())
    assert [f.source_t_s for f in frames] == [i / 25 for i in range(10)]
    assert all(f.timing_mode == TimingMode.FRAME_INDEX_FPS_FALLBACK for f in frames[1:])
    controller.stop()


def test_real_file_seek_replay_and_utc_offset():
    row = load_dataset_manifest(ROOT / 'data/manifests/dev.json').clips[1]
    controller = CaptureSessionManager(ROOT / row.relative_path, recorded_start_utc=UTC)
    first_id = controller.start()
    controller.queue.get()
    controller.pump()
    first_pixels = controller.queue.get().copy_frame()
    second_id = controller.seek(25)
    assert first_id != second_id
    assert controller.queue.get().reason == 'seek'
    controller.pump()
    frame = controller.queue.get()
    assert frame.sequence == 0 and frame.source_t_s == 0
    assert frame.source_frame_index == 25 and frame.source_time_offset_s == 1
    assert controller.estimate_event_utc(0) == '2020-01-02T03:04:06Z'
    third_id = controller.replay()
    controller.queue.get()
    controller.pump()
    assert third_id != second_id
    assert np.array_equal(controller.queue.get().frame_bgr, first_pixels)
    controller.stop()


def test_live_reader_eof_is_failure_not_file_eof():
    class DeadCamera(FakeReader):
        def read(self): return None, None
    controller = CaptureSessionManager(0, source_factory=DeadCamera)
    controller.start()
    controller.queue.get()
    with pytest.raises(CaptureError, match='unexpectedly'): controller.pump()
    assert controller.state == 'error'
    with pytest.raises(Empty): controller.queue.get(0)


def test_cancel_wakes_blocked_file_producer():
    queue = BoundedFrameQueue(SourceKind.VIDEO)
    for i in range(2): queue.put(packet(i, SourceKind.VIDEO))
    cancel, started, result = Event(), Event(), []
    def produce():
        started.set()
        result.append(queue.put(packet(2, SourceKind.VIDEO), timeout_s=2, cancel=cancel))
    thread = Thread(target=produce)
    thread.start()
    assert started.wait(1)
    cancel.set()
    thread.join(1)
    assert not thread.is_alive() and result == [False]
    assert queue.dropped_frames_count == 0


def test_zero_frame_eof_and_event_validation():
    class EmptyReader(FakeReader):
        frame_count = 0
    controller = CaptureSessionManager('empty.avi', source_factory=EmptyReader)
    controller.start()
    controller.queue.get()
    controller.pump()
    assert controller.queue.get() == EndOfStreamEvent(controller.session_id, 'empty.avi', 0, None, None)
    assert controller.last_source_t_s is None
    with pytest.raises(ValueError): EndOfStreamEvent('s', 'file', 0, 0, 0.)
    with pytest.raises(ValueError): EndOfStreamEvent('s', 'file', 10, 3, .12)
    with pytest.raises(ValueError): SessionResetEvent('s', 's', 'restart', SourceKind.VIDEO, 'file')
