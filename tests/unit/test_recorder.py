"""Deterministic media tests plus real local disk/AVI encode/decode checks.

Injected codec and disk failures are explicitly labelled; no accuracy claims.
"""
from dataclasses import replace
import gc
import json
from pathlib import Path
import re
from threading import Event
from uuid import uuid4

import cv2
import numpy as np
import pytest

from vertebrate.config import AppConfig
from vertebrate.contracts import FramePacket, SourceKind
from vertebrate.incidents import incident_from_dict, incident_from_json
from vertebrate.recorder import ClipRecorder, MediaBudget, FRAME_BYTES, MAX_MEDIA_BYTES
from vertebrate.storage import StorageWorker, StorageError, scan_outbox

ROOT = Path(__file__).resolve().parents[2]


def packet(t, session='s', offset=0.):
    return FramePacket(session, round(t * 25), SourceKind.VIDEO, 'synthetic', 640, 360,
        np.full((360, 640, 3), round(t * 5) % 256, np.uint8), t, round(t*1e9)+1,
        source_time_offset_s=offset)


def fill(recorder, start, end):
    ready = []
    for i in range(round(start*25), round(end*25)+1):
        ready.extend(recorder.add(packet(i/25)))
    return ready


def test_dimensions_sampling_and_detection_buffer_isolation():
    r = ClipRecorder()
    original = packet(0.)
    before = original.frame_bgr.copy()
    r.add(original)
    fill(r, .04, 1.)
    assert [f.source_t_s for f in r.buffered_frames] == pytest.approx([0,.2,.4,.6,.8,1.])
    frame = r.buffered_frames[0].frame_bgr
    assert frame.shape == (180,320,3) and frame.dtype == np.uint8
    assert not frame.flags.writeable and not np.shares_memory(frame, original.frame_bgr)
    with pytest.raises(ValueError):
        frame.setflags(write=True)
    assert np.array_equal(original.frame_bgr, before)


def test_thirty_second_ring_has_at_most_150_frames_and_releases():
    r = ClipRecorder()
    fill(r, 0, 36)
    assert len(r.buffered_frames) == 150
    assert r.buffered_frames[0].source_t_s == pytest.approx(6.2)
    assert r.budget.used_bytes == 150 * FRAME_BYTES
    r.flush()
    gc.collect()
    assert r.budget.used_bytes == 0


def test_full_requested_window_and_timestamp_metadata():
    r = ClipRecorder()
    fill(r,0,8)
    r.request('one', 5., 8.)
    assert fill(r,8.04,10.96) == []
    payload, = r.add(packet(11.))
    assert not payload.truncated
    assert (payload.requested_start_s, payload.requested_end_s) == (2.,11.)
    assert [f.source_t_s for f in payload.frames] == pytest.approx(np.arange(2,11.01,.2))
    assert payload.metadata()['frames'][0] == {'sequence':50,'source_t_s':2.}


def test_simultaneous_incidents_share_immutable_pixels_and_combined_budget():
    budget = MediaBudget(20 * FRAME_BYTES)
    r = ClipRecorder(budget)
    fill(r,0,2)
    r.request('one',1.,2.)
    r.request('two',1.,2.)
    ready = fill(r,2.04,5)
    assert len(ready) == 2 and all(p.truncated for p in ready)
    assert ready[0].frames[0] is ready[1].frames[0]
    assert budget.used_bytes <= budget.limit
    r.flush()
    assert budget.used_bytes > 0  # Writer payload still owns its pixels.
    ready.clear()
    gc.collect()
    assert budget.used_bytes == 0


def test_hard_128_mib_cap_and_atomic_reservations():
    budget = MediaBudget(MAX_MEDIA_BYTES*2)
    assert budget.limit == MAX_MEDIA_BYTES
    budget.reserve(MAX_MEDIA_BYTES)
    with pytest.raises(MemoryError): budget.reserve(1)
    assert budget.used_bytes == MAX_MEDIA_BYTES
    budget.release(MAX_MEDIA_BYTES)
    with pytest.raises(ValueError): budget.reserve(-1)


@pytest.mark.parametrize('limit', [1, FRAME_BYTES, 2*FRAME_BYTES-1])
def test_no_room_for_resize_has_explicit_unavailable_clip(limit):
    r = ClipRecorder(MediaBudget(limit))
    r.add(packet(0.))
    r.request('one',0.,0.)
    payload, = r.flush()
    assert payload.frames == () and payload.truncated and payload.reasons
    assert r.budget.used_bytes == 0


@pytest.mark.parametrize('reason', ['EOF','Stop','Replay','Seek','Source switch'])
def test_flush_truncates_postroll_and_resets_session(reason):
    r = ClipRecorder()
    fill(r,0,2)
    r.request('one',1.,2.)
    payload, = r.flush(reason)
    assert payload.truncated and reason in payload.reasons
    assert payload.frames[-1].source_t_s == 2.
    assert r.flush() == ()
    r.add(packet(0.,'new'))
    assert r.buffered_frames[0].source_t_s == 0.


def test_old_pre_roll_is_truncated_and_occlusion_gap_is_not_filled():
    r = ClipRecorder()
    fill(r,0,35)
    r.request('old',1.,35.)
    p, = r.add(packet(38.))
    assert p.truncated
    assert 'Pre-onset history incomplete' in p.reasons
    assert any('gap' in reason for reason in p.reasons)
    assert p.frames[-2].source_t_s == 35.


def test_source_offset_preserved_without_changing_session_times():
    r = ClipRecorder()
    r.add(packet(0.,offset=12.))
    r.request('one',0.,0.)
    p, = r.flush()
    assert p.metadata()['source_time_offset_s'] == 12.
    assert p.frames[0].source_t_s == 0.


def test_lifecycle_and_request_validation():
    r = ClipRecorder(max_pending=1)
    with pytest.raises(ValueError): r.request('one',0,0)
    r.add(packet(0.))
    with pytest.raises(ValueError): r.add(packet(0.))
    with pytest.raises(ValueError): r.add(packet(1.,'other'))
    with pytest.raises(ValueError): r.request('one',2.,1.)
    r.request('one',0.,0.)
    with pytest.raises(ValueError): r.request('one',0.,0.)
    with pytest.raises(MemoryError): r.request('two',0.,0.)


@pytest.fixture
def incident():
    source = (ROOT/'docs/03-validated-blueprint-v2.md').read_text(encoding='utf-8')
    data = json.loads(re.search(r'```json\s*(.*?)\s*```', source.split('## 10. Incident schema')[1], re.S).group(1))
    value = incident_from_dict(data)
    return replace(value, session_id='s', configuration=replace(value.configuration, revision=1),
                   media=replace(value.media, clip_status='pending'))


@pytest.fixture
def writer(tmp_path):
    w = StorageWorker(tmp_path/'outbox',pause_acquisition=lambda:None).start()
    yield w
    w.stop()


def make_clip(writer, incident):
    r = ClipRecorder(writer.media_budget)
    fill(r,0,1)
    r.request(incident.incident_id,0.,1.)
    return r.flush()[0]


def save_snapshot(writer,incident):
    return writer.submit(incident,AppConfig(),np.zeros((180,320,3),np.uint8)).result(10)


def test_real_clip_atomic_commit_timestamps_ack_and_idempotency(writer,incident,monkeypatch):
    assert save_snapshot(writer,incident).media.clip_status == 'pending'
    p = make_clip(writer,incident)
    renames = []
    import vertebrate.storage as module
    original = module.os.replace
    def watch(src,dest):
        assert Path(src).exists()
        renames.append(Path(dest).name)
        return original(src,dest)
    monkeypatch.setattr(module.os,'replace',watch)
    saved = writer.attach_clip(p).result(10)
    assert saved.delivery.status == 'saved' and saved.media.clip_status == 'truncated'
    assert renames == ['clip.avi','clip-timestamps.json','incident.json']
    folder = writer.outbox/incident.incident_id
    metadata = json.loads((folder/'clip-timestamps.json').read_text())
    assert metadata == p.metadata()
    reader = cv2.VideoCapture(str(folder/'clip.avi'))
    try:
        assert reader.get(cv2.CAP_PROP_FPS) == 5
        assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == len(p.frames)
        assert reader.read()[1].shape == (180,320,3)
    finally: reader.release()
    ack = writer.acknowledge(incident.incident_id,'2030-01-01T00:00:00Z').result(10)
    assert writer.attach_clip(p).result(10) == ack
    assert len(scan_outbox(writer.outbox).incidents) == 1
    assert not list(folder.glob('*.tmp*'))


@pytest.mark.parametrize('failure', ['unavailable','silent_write','exception'])
def test_injected_codec_failures_keep_snapshot_and_json(writer,incident,monkeypatch,failure):
    save_snapshot(writer,incident)
    p = make_clip(writer,incident)
    class BrokenWriter:
        def __init__(self,*a):
            if failure == 'exception': raise OSError('injected codec exception')
        def isOpened(self): return failure != 'unavailable'
        def write(self,frame): pass
        def release(self): pass
    monkeypatch.setattr(cv2,'VideoWriter',BrokenWriter)
    saved = writer.attach_clip(p).result(10)
    assert saved.media.clip_path is None and saved.media.clip_status == 'unavailable'
    assert saved.media.clip_error and saved.media.snapshot_path == 'snapshot.jpg'
    assert saved.delivery.status == 'saved' and not writer.pause_requested.is_set()
    assert scan_outbox(writer.outbox).incidents == (saved,)


def test_metadata_failure_retains_clip_for_retry(writer,incident,monkeypatch):
    save_snapshot(writer,incident)
    payload = make_clip(writer,incident)
    original = writer._atomic_write
    def fail(path,data):
        if path.name == 'incident.json': raise OSError('injected metadata failure')
        return original(path,data)
    monkeypatch.setattr(writer,'_atomic_write',fail)
    with pytest.raises(StorageError,match='metadata failure'):
        writer.attach_clip(payload).result(10)
    assert writer.pause_requested.is_set()
    assert incident.incident_id in writer.unsaved_incident_ids
    assert scan_outbox(writer.outbox).incidents[0].media.snapshot_path == 'snapshot.jpg'
    monkeypatch.setattr(writer,'_atomic_write',original)
    assert writer.retry(incident.incident_id).result(10).media.clip_path == 'clip.avi'
    assert writer.unsaved_incident_ids == ()


def test_restart_marks_interrupted_postroll_unavailable(tmp_path,incident):
    with_writer = StorageWorker(tmp_path,pause_acquisition=lambda:None).start()
    save_snapshot(with_writer,incident)
    with_writer.stop()
    new = StorageWorker(tmp_path,pause_acquisition=lambda:None).start()
    try:
        saved = new.record(incident.incident_id).incident
        assert saved.media.clip_status == 'unavailable' and saved.media.clip_error
        assert saved.media.snapshot_path == 'snapshot.jpg'
    finally: new.stop()


def test_clip_queue_overflow_retains_both_incidents_and_shared_budget(tmp_path,incident,monkeypatch):
    writer = StorageWorker(tmp_path,pause_acquisition=lambda:None,capacity=1).start()
    entered, release = Event(), Event()
    original = writer._save_clip
    def blocked(key):
        entered.set()
        assert release.wait(10)
        return original(key)
    try:
        second = replace(incident,incident_id=str(uuid4()))
        save_snapshot(writer,incident)
        save_snapshot(writer,second)
        a, b = make_clip(writer,incident), make_clip(writer,second)
        monkeypatch.setattr(writer,'_save_clip',blocked)
        first = writer.attach_clip(a)
        assert entered.wait(10)
        queued = writer.acknowledge(incident.incident_id,'2030-01-01T00:00:00Z')
        with pytest.raises(StorageError,match='queue full'):
            writer.attach_clip(b).result(10)
        assert writer.pause_requested.is_set()
        assert writer.media_budget.used_bytes <= writer.max_media_buffer_bytes
        assert second.incident_id in writer.unsaved_incident_ids
        release.set()
        first.result(10)
        queued.result(10)
        assert writer.retry(second.incident_id).result(10).media.clip_path == 'clip.avi'
        assert len(scan_outbox(writer.outbox).incidents) == 2
    finally:
        release.set()
        writer.stop()


def test_snapshot_and_recorder_share_cap_and_foreign_payload_rejected(tmp_path,incident):
    writer = StorageWorker(tmp_path,pause_acquisition=lambda:None,
                           max_media_buffer_bytes=2*FRAME_BYTES).start()
    r = ClipRecorder(writer.media_budget)
    try:
        r.add(packet(0.))
        assert writer.media_budget.used_bytes == FRAME_BYTES
        too_large = np.zeros((180,321,3),np.uint8)
        with pytest.raises(StorageError,match='memory limit'):
            writer.submit(incident,AppConfig(),too_large).result(10)
        assert writer.media_budget.used_bytes == FRAME_BYTES
        writer.retry(incident.incident_id).result(10)
        other = ClipRecorder()
        other.add(packet(0.))
        other.request(incident.incident_id,0.,0.)
        with pytest.raises(ValueError,match='shared media budget'):
            writer.attach_clip(other.flush()[0])
    finally:
        r.flush()
        writer.stop()


def test_injected_clip_rename_failure_keeps_durable_snapshot(writer,incident,monkeypatch):
    save_snapshot(writer,incident)
    import vertebrate.storage as module
    original = module.os.replace
    def fail(src,dest):
        if Path(dest).name == 'clip.avi': raise OSError('injected clip disk failure')
        return original(src,dest)
    monkeypatch.setattr(module.os,'replace',fail)
    saved = writer.attach_clip(make_clip(writer,incident)).result(10)
    assert saved.media.clip_status == 'unavailable'
    assert 'injected clip disk failure' in saved.media.clip_error
    assert saved.media.snapshot_path == 'snapshot.jpg'
    assert incident_from_json((writer.outbox/incident.incident_id/'incident.json').read_text()) == saved


def test_one_capacity_miss_is_explicit_even_when_gap_is_only_point_four_seconds():
    r = ClipRecorder()
    fill(r,0,.4)
    pressure = r.budget.limit - r.budget.used_bytes - FRAME_BYTES
    r.budget.reserve(pressure)
    fill(r,.44,.6)
    r.budget.release(pressure)
    fill(r,.64,3)
    r.request('one',3.,3.)
    payload, = fill(r,3.04,6)
    assert payload.truncated
    assert 'Media capacity limit omitted sampled frames' in payload.reasons
    assert all(abs(f.source_t_s-.6)>1e-9 for f in payload.frames)


def test_cancelled_clip_receipt_does_not_duplicate_or_cancel_delivery(tmp_path,incident,monkeypatch):
    writer = StorageWorker(tmp_path,pause_acquisition=lambda:None).start()
    entered, release = Event(), Event()
    original = writer._save_clip
    def blocked(key):
        entered.set()
        assert release.wait(10)
        return original(key)
    try:
        second = replace(incident,incident_id=str(uuid4()))
        save_snapshot(writer,incident)
        save_snapshot(writer,second)
        a,b = make_clip(writer,incident),make_clip(writer,second)
        monkeypatch.setattr(writer,'_save_clip',blocked)
        first = writer.attach_clip(a)
        assert entered.wait(10)
        cancelled = writer.attach_clip(b)
        assert cancelled.cancel()
        assert writer.attach_clip(b) is cancelled
        release.set()
        first.result(10)
        writer.stop()
        assert writer.unsaved_incident_ids == ()
        assert all(i.media.clip_path == 'clip.avi' for i in scan_outbox(tmp_path).incidents)
        assert not writer.faults
    finally:
        release.set()
        writer.stop()
