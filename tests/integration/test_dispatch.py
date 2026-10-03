"""Real local disk/encoder/thread tests; failure cases explicitly inject faults."""
from dataclasses import replace
import json
from pathlib import Path
from threading import Event, get_ident
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest

from vertebrate.capture import CaptureWorker
from vertebrate.config import AppConfig, load_config, config_sha256
from vertebrate.context import build_environment
from vertebrate.contracts import FramePacket, PersonTrackKey, SourceKind, TrackSample, TimingMode
from vertebrate.features import FeatureExtractor
from vertebrate.fsm import PersonFSM
from vertebrate.incidents import IncidentFactory, incident_from_json
from vertebrate.offline import offline_guard
from vertebrate.storage import StorageError, StorageWorker, scan_outbox

ROOT = Path(__file__).resolve().parents[2]
NOW = '2026-10-02T12:00:10Z'


@pytest.fixture(scope='module')
def confirmation():
    """Actual deterministic feature/FSM confirmation, not a mocked alert."""
    key = PersonTrackKey('dispatch-session', 1, 1)
    cfg = AppConfig()
    extractor, fsm = FeatureExtractor(cfg.calibration), PersonFSM(key, cfg.calibration)
    for i in range(71):
        t = i / 10
        p = min(1., max(0., (t - 1) / .6))
        xy = np.full((17, 2), 100.)
        for a, b, center in ((5, 6, (100 - 80*p, 150 + 200*p)), (11, 12, (100, 250 + 100*p)),
                             (13, 14, (100, 275 + 75*p)), (15, 16, (100, 300 + 50*p))):
            xy[a], xy[b] = (center[0] - 10, center[1]), (center[0] + 10, center[1])
        sample = TrackSample(key, i, t, np.array([0., 100., 80 + 140*p, 300 - 100*p]), .9,
                             xy, np.ones(17), np.ones(17, bool), True)
        features = extractor.update(sample, candidate_active=fsm.freeze_baseline)
        decision = fsm.update(features)
        if decision.alert:
            frame = FramePacket(key.session_id, i, SourceKind.VIDEO, 'synthetic-local', 320, 180,
                                np.full((180, 320, 3), 80, np.uint8), t, 1000000 + i)
            return decision.alert, features, frame, cfg
    pytest.fail('Synthetic trajectory did not confirm')


@pytest.fixture
def payload(confirmation):
    alert, features, frame, cfg = confirmation
    incident = IncidentFactory().create(alert, features, frame, cfg, confirmed_at_utc=NOW)
    return incident, cfg, frame.copy_frame()


@pytest.fixture
def worker(tmp_path):
    writer = StorageWorker(tmp_path / 'outbox', pause_acquisition=lambda: None).start()
    yield writer
    writer.stop()


def test_actual_confirmation_persists_one_record_config_and_decodable_snapshot(worker, payload):
    incident, cfg, snapshot = payload
    with offline_guard(ROOT) as attempts:
        future = worker.submit(incident, cfg, snapshot)
        saved = future.result(10)
        folder = worker.outbox / incident.incident_id
        assert sorted(p.name for p in folder.iterdir()) == ['config.json', 'incident.json', 'snapshot.jpg']
        assert incident_from_json((folder / 'incident.json').read_text()) == saved
        assert config_sha256(load_config(folder / 'config.json')) == config_sha256(cfg)
        decoded = cv2.imread(str(folder / 'snapshot.jpg'))
        assert decoded.shape == snapshot.shape and np.max(np.abs(decoded.astype(int) - snapshot)) <= 1
        assert saved.delivery.status == 'saved' and incident.delivery.status == 'pending'
        assert worker.record(incident.incident_id).status == 'saved'
        assert len(scan_outbox(worker.outbox).incidents) == 1
    assert attempts == []


def test_factory_assigns_uuid4_once_and_keeps_observed_stillness_start(confirmation):
    alert, features, frame, cfg = confirmation
    factory = IncidentFactory()
    incident = factory.create(alert, features, frame, cfg, confirmed_at_utc=NOW)
    assert UUID(alert.incident_id).version == 5  # existing deterministic replay token
    assert UUID(incident.incident_id).version == 4
    assert factory.create(alert, features, frame, cfg, confirmed_at_utc=NOW) is incident
    assert incident.timing.stillness_start_source_s == alert.stillness_start_source_t_s
    assert incident.kinematics.common_core_keypoints == 8
    assert incident.timing.fall_onset_utc is None and incident.timing.fall_onset_utc_reason


@pytest.mark.parametrize('kind,anchor,offset', [('video', '2020-01-01T00:00:00Z', 10.),
                                             ('live', '2026-10-02T12:00:00Z', 0.)])
def test_utc_provenance_and_seek_source_offset(confirmation, kind, anchor, offset):
    from datetime import timedelta
    from vertebrate.validation import utc_datetime, utc_string
    alert, features, frame, cfg = confirmation
    frame = replace(frame, source_kind=SourceKind(kind),
                    timing_mode=TimingMode.LIVE_MONOTONIC if kind == 'live' else frame.timing_mode,
                    recorded_start_utc=anchor if kind == 'video' else None,
                    session_start_utc=anchor if kind == 'live' else None, source_time_offset_s=offset)
    incident = IncidentFactory().create(alert, features, frame, cfg, confirmed_at_utc=NOW)
    expected = utc_string(utc_datetime(anchor) + timedelta(seconds=alert.onset_source_t_s + offset))
    assert incident.timing.fall_onset_utc == expected
    assert incident.timing.confirmed_source_s == alert.confirmed_source_t_s + offset


def test_bad_frame_alignment_and_missing_actual_evidence_rejected(confirmation):
    alert, features, frame, cfg = confirmation
    with pytest.raises(ValueError, match='match'):
        IncidentFactory().create(alert, features, replace(frame, sequence=frame.sequence + 1), cfg, confirmed_at_utc=NOW)
    with pytest.raises(ValueError, match='observed'):
        IncidentFactory().create(replace(alert, stillness_start_source_t_s=None), features, frame, cfg, confirmed_at_utc=NOW)


def test_missing_and_stale_environment_do_not_change_confirmation(confirmation):
    alert, features, frame, cfg = confirmation
    stale = build_environment(30, 50, source='manual_measured', observed_at_utc='2026-10-02T10:00:00Z', now_utc=NOW)
    for env in (None, stale, build_environment(float('nan'), 50, source='manual_simulated')):
        incident = IncidentFactory().create(alert, features, frame, cfg, confirmed_at_utc=NOW, environment=env)
        assert incident.timing.confirmed_source_s == alert.confirmed_source_t_s
        assert incident.evidence.sustained_stillness


def test_freshness_recomputed_at_confirmation(confirmation):
    alert, features, frame, cfg = confirmation
    env = build_environment(30, 50, source='manual_measured', observed_at_utc='2026-10-02T11:00:00Z',
                            now_utc='2026-10-02T11:01:00Z')
    assert env.stale is False
    incident = IncidentFactory().create(alert, features, frame, cfg, confirmed_at_utc=NOW, environment=env)
    assert incident.environment.stale is True and env.stale is False


def test_atomic_renames_json_last_and_only_writer_mutates(worker, payload, monkeypatch):
    import vertebrate.storage as storage
    original = storage.os.replace
    calls, owners = [], []
    def observed(src, dst):
        assert Path(src).name.endswith('.tmp') and Path(src).is_file()
        calls.append(Path(dst).name)
        owners.append(get_ident())
        if Path(dst).name == 'incident.json':
            assert (Path(dst).parent / 'config.json').is_file()
            assert (Path(dst).parent / 'snapshot.jpg').is_file()
            assert worker.record(payload[0].incident_id).status != 'saved'
        return original(src, dst)
    monkeypatch.setattr(storage.os, 'replace', observed)
    worker.submit(*payload).result(10)
    assert calls == ['snapshot.jpg', 'config.json', 'incident.json']
    assert len(set(owners)) == 1 and owners[0] != get_ident()


def test_duplicate_save_and_acknowledgement_are_idempotent(worker, payload):
    first = worker.submit(*payload).result(10)
    folder = worker.outbox / first.incident_id
    original = (folder / 'incident.json').read_bytes()
    assert worker.submit(*payload).result(10) == first
    assert (folder / 'incident.json').read_bytes() == original
    ack = worker.acknowledge(first.incident_id, '2026-10-02T12:01:00Z').result(10)
    assert ack.acknowledgement.status == 'acknowledged'
    assert worker.acknowledge(first.incident_id, '2026-10-02T12:02:00Z').result(10) == ack
    assert worker.submit(*payload).result(10) == ack
    assert len(list(worker.outbox.glob('*/incident.json'))) == 1


def test_restart_reads_only_complete_records_and_preserves_ack(tmp_path, payload):
    root = tmp_path / 'outbox'
    writer = StorageWorker(root, pause_acquisition=lambda: None).start()
    writer.submit(*payload).result(10)
    saved = writer.acknowledge(payload[0].incident_id, '2026-10-02T12:01:00Z').result(10)
    writer.stop()
    incomplete = root / str(uuid4())
    incomplete.mkdir()
    (incomplete / 'incident.json.tmp').write_text('{}')
    bad = root / str(uuid4())
    bad.mkdir()
    (bad / 'incident.json').write_text('{}')
    restarted = StorageWorker(root, pause_acquisition=lambda: None).start()
    try:
        assert restarted.record(saved.incident_id).incident == saved
        assert len(restarted.startup_issues) == 2
        assert len(scan_outbox(root).incidents) == 1
        assert restarted.submit(*payload).result(10) == saved
    finally:
        restarted.stop()


def test_snapshot_write_failure_saves_metadata_with_explicit_error(worker, payload, monkeypatch):
    original = worker._atomic_write
    def fail_snapshot(path, data):
        if path.name == 'snapshot.jpg':
            raise OSError('injected snapshot disk failure')
        return original(path, data)
    monkeypatch.setattr(worker, '_atomic_write', fail_snapshot)
    saved = worker.submit(*payload).result(10)
    assert saved.media.snapshot_path is None and 'injected' in saved.media.snapshot_error
    assert saved.delivery.status == 'saved'
    assert scan_outbox(worker.outbox).incidents == (saved,)


@pytest.mark.parametrize('destination', ['config.json', 'incident.json'])
def test_metadata_failure_retains_incident_and_retry_commits_once(worker, payload, monkeypatch, destination):
    original = worker._atomic_write
    def fail(path, data):
        if path.name == destination:
            raise PermissionError('injected metadata failure')
        return original(path, data)
    monkeypatch.setattr(worker, '_atomic_write', fail)
    with pytest.raises(StorageError, match='injected metadata'):
        worker.submit(*payload).result(10)
    incident_id = payload[0].incident_id
    assert worker.pause_requested.wait(1)
    retained = worker.record(incident_id)
    assert retained.status == 'save_failed' and retained.incident.delivery.status == 'save_failed'
    assert not (worker.outbox / incident_id / 'incident.json').exists()
    assert scan_outbox(worker.outbox).incidents == ()
    monkeypatch.setattr(worker, '_atomic_write', original)
    assert worker.retry(incident_id).result(10).delivery.status == 'saved'
    assert len(list(worker.outbox.glob('*/incident.json'))) == 1


def test_failed_ack_does_not_damage_existing_commit(worker, payload, monkeypatch):
    saved = worker.submit(*payload).result(10)
    target = worker.outbox / saved.incident_id / 'incident.json'
    before = target.read_bytes()
    original = worker._atomic_write
    def fail(*args):
        raise OSError('injected acknowledgement failure')
    monkeypatch.setattr(worker, '_atomic_write', fail)
    with pytest.raises(StorageError):
        worker.acknowledge(saved.incident_id, '2026-10-02T12:01:00Z').result(10)
    assert target.read_bytes() == before
    assert worker.record(saved.incident_id).status == 'ack_failed'
    monkeypatch.setattr(worker, '_atomic_write', original)
    assert worker.acknowledge(saved.incident_id, '2026-10-02T12:01:00Z').result(10).acknowledgement.status == 'acknowledged'


def test_capacity_eight_overflow_retains_alert_and_pauses_real_capture(tmp_path, payload, monkeypatch):
    capture = CaptureWorker(ROOT / 'tests/fixtures/videos/dev_empty_scene.avi')
    capture.start()
    writer = StorageWorker(tmp_path / 'outbox', pause_acquisition=capture.pause).start()
    entered, release = Event(), Event()
    original = writer._save
    def blocked(*args):
        entered.set()
        if not release.wait(10):
            raise RuntimeError('Test gate was not released')
        return original(*args)
    monkeypatch.setattr(writer, '_save', blocked)
    try:
        futures = [writer.submit(*payload)]
        assert entered.wait(2)
        for _ in range(8):
            futures.append(writer.submit(replace(payload[0], incident_id=str(uuid4())), payload[1], payload[2]))
        overflow = replace(payload[0], incident_id=str(uuid4()))
        with pytest.raises(StorageError, match='queue is full'):
            writer.submit(overflow, payload[1], payload[2]).result(2)
        assert writer.queue.qsize() == 8
        assert writer.pause_requested.is_set() and capture.state == 'paused'
        assert writer.record(overflow.incident_id).status == 'save_failed'
        release.set()
        for future in futures:
            future.result(10)
        writer.retry(overflow.incident_id).result(10)
        assert len(scan_outbox(writer.outbox).incidents) == 10
        assert capture.state == 'paused'  # no automatic restart after a fault
    finally:
        release.set()
        writer.stop()
        capture.stop()


def test_snapshot_is_copied_before_cross_thread_retention(worker, payload, monkeypatch):
    entered, release = Event(), Event()
    original = worker._save
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(worker, '_save', blocked)
    future = worker.submit(*payload)
    assert entered.wait(2)
    payload[2][:] = 0
    release.set()
    saved = future.result(10)
    decoded = cv2.imread(str(worker.outbox / saved.incident_id / 'snapshot.jpg'))
    assert decoded.mean() == pytest.approx(80, abs=1)


def test_startup_write_failure_and_single_writer_lease(tmp_path, payload):
    obstructed = tmp_path / 'file'
    obstructed.write_text('not a directory')
    paused = Event()
    failed = StorageWorker(obstructed, pause_acquisition=paused.set)
    with pytest.raises(StorageError):
        failed.start()
    failed.stop()
    assert paused.is_set()
    first = StorageWorker(tmp_path / 'outbox', pause_acquisition=lambda: None).start()
    second = StorageWorker(first.outbox, pause_acquisition=lambda: None)
    try:
        with pytest.raises(StorageError):
            second.start()
    finally:
        second.stop()
        first.stop()


def test_conflicting_payload_or_config_cannot_overwrite_saved_id(worker, payload):
    worker.submit(*payload).result(10)
    with pytest.raises(StorageError, match='Conflicting'):
        worker.submit(replace(payload[0], session_id='other'), payload[1], payload[2])
    changed = replace(payload[1], calibration=replace(payload[1].calibration, filter_tau_s=.07))
    with pytest.raises(StorageError, match='Conflicting'):
        worker.submit(payload[0], changed, payload[2])


def test_memory_budget_keeps_metadata_and_explicit_media_failure(tmp_path, payload):
    writer = StorageWorker(tmp_path / 'outbox', pause_acquisition=lambda: None, max_media_buffer_bytes=1).start()
    try:
        with pytest.raises(StorageError, match='memory limit'):
            writer.submit(*payload).result(2)
        saved = writer.retry(payload[0].incident_id).result(10)
        assert saved.media.snapshot_path is None and 'memory limit' in saved.media.snapshot_error
        assert writer.pause_requested.is_set()
    finally:
        writer.stop()


def test_atomic_replace_failure_keeps_old_json_and_removes_temp(worker, payload, monkeypatch):
    import vertebrate.storage as storage
    saved = worker.submit(*payload).result(10)
    folder = worker.outbox / saved.incident_id
    before = (folder / 'incident.json').read_bytes()
    def fail_replace(src, dst):
        assert Path(src).read_bytes()  # failure after actual temp write/fsync
        raise OSError('injected rename failure')
    monkeypatch.setattr(storage.os, 'replace', fail_replace)
    with pytest.raises(StorageError, match='rename failure'):
        worker.acknowledge(saved.incident_id, '2026-10-02T12:01:00Z').result(10)
    assert (folder / 'incident.json').read_bytes() == before
    assert list(folder.glob('*.tmp')) == []


def test_corrupt_snapshot_is_ignored_on_restart(worker, payload):
    saved = worker.submit(*payload).result(10)
    (worker.outbox / saved.incident_id / 'snapshot.jpg').write_bytes(b'')
    scan = scan_outbox(worker.outbox)
    assert scan.incidents == () and 'Snapshot is empty' in scan.issues[0]


def test_cancelled_receipt_does_not_drop_incident_or_stop_writer(worker, payload, monkeypatch):
    entered, release = Event(), Event()
    original = worker._save
    def block(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(worker, '_save', block)
    first = worker.submit(*payload)
    assert entered.wait(2)
    assert worker.submit(*payload) is first
    second = replace(payload[0], incident_id=str(uuid4()))
    receipt = worker.submit(second, payload[1], payload[2])
    assert receipt.cancel()
    release.set()
    first.result(10)
    assert worker.acknowledge(second.incident_id, '2026-10-02T12:01:00Z').result(10).delivery.status == 'saved'
    assert len(scan_outbox(worker.outbox).incidents) == 2
