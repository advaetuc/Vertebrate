"""Real local pose/capture smoke plus controlled real ByteTrack association.

Controlled tensors test mapping/boundaries, not model detection accuracy.
"""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vertebrate.capture import CaptureSessionManager
from vertebrate.config import AppConfig, TrackerConfig
from vertebrate.contracts import (AssetVerificationError, EndOfStreamEvent, FramePacket,
    OfflineNetworkError, PoseObservation, SessionResetEvent, SourceKind)
from vertebrate.pose import PoseAdapter, guarded_runtime, observation_from_result
from vertebrate.tracking import ByteTrackManager

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / 'tests/fixtures/videos/dev_empty_scene.avi'


def packet(width=640, height=360, sequence=0, session='s'):
    return FramePacket(session, sequence, SourceKind.VIDEO, 'local', width, height,
                       np.zeros((height, width, 3), np.uint8), sequence / 25, 1000000 + sequence)


def observation(sequence=0, time=0, scores=(.9, .8), labels=(0, 1), session='s'):
    boxes = np.array([[20 + i * 250, 30, 120 + i * 250, 300] for i in labels], np.float32).reshape(-1, 4)
    kp = np.array([np.tile([60 + i * 250, 100], (17, 1)) for i in labels], np.float32).reshape(-1, 17, 2)
    conf = np.array([np.full(17, .6 + i * .1) for i in labels], np.float32).reshape(-1, 17)
    mask = np.ones((len(labels), 17), bool)
    for row, label in enumerate(labels):
        mask[row, label] = False
    return PoseObservation(session, sequence, time, 640, 360, boxes,
                           np.array(scores, np.float32), kp, conf, mask)


@pytest.fixture
def manager():
    result = ByteTrackManager(TrackerConfig(), ROOT, 25)
    yield result
    assert result.network_attempts == []


@pytest.fixture(scope='module')
def adapter():
    result = PoseAdapter(AppConfig(), ROOT)
    assert result.warmup() > 0
    yield result
    assert result.network_attempts == []


def test_two_stage_alignment_and_confidence_order_flips(manager):
    first = manager.update(observation())
    ids = {int(s.keypoints_xy[0, 0]): s.person_key.tracker_id for s in first}
    # Low-stage row 0 and high-stage row 1 both used to acquire index 0.
    for seq, scores, labels in [(1, (.15, .9), (1, 0)), (2, (.9, .15), (1, 0)),
                                 (3, (.8, .9), (0, 1))]:
        obs = observation(seq, seq * .04, scores, labels)
        samples = manager.update(obs)
        assert len(samples) == 2
        for sample in samples:
            index = next(i for i in range(2) if np.array_equal(obs.keypoints_xy[i], sample.keypoints_xy))
            assert sample.person_key.tracker_id == ids[int(sample.keypoints_xy[0, 0])]
            np.testing.assert_array_equal(sample.keypoint_confidences, obs.keypoint_confidences[index])
            np.testing.assert_array_equal(sample.keypoint_valid_mask, obs.keypoint_valid_mask[index])
            assert sample.box_confidence == float(obs.box_confidences[index])
            np.testing.assert_allclose(sample.bbox_xyxy, obs.boxes_xyxy[index], atol=.01)
            assert sample.observation_valid and not sample.predicted_only


@pytest.mark.parametrize('score,matched', [(.10, False), (.10001, True), (.2499, True), (.25, True)])
def test_strict_low_boundary(manager, score, matched):
    manager.update(observation(scores=(.9,), labels=(0,)))
    result = manager.update(observation(1, .04, (score,), (0,)))
    assert bool(result) is matched
    assert manager.last_matched_source_t_s[1] == (.04 if matched else 0)


@pytest.mark.parametrize('gap,generation', [(.5, 1), (.75, 1), (.750001, 2), (1.2, 2)])
def test_source_time_generation_expiry_and_no_prediction_evidence(manager, gap, generation):
    first = manager.update(observation(scores=(.9,), labels=(0,)))[0]
    assert manager.update(observation(1, .1, (), ())) == ()
    assert manager.last_matched_source_t_s[first.person_key.tracker_id] == 0
    again = manager.update(observation(2, gap, (.9,), (0,)))[0]
    assert again.person_key.tracker_id == first.person_key.tracker_id
    assert again.person_key.generation == generation


def test_empty_and_degenerate_observations(manager):
    assert manager.update(observation(scores=(), labels=())) == ()
    obs = observation(1, .04, (.9, .8), (0, 1))
    boxes = obs.boxes_xyxy.copy()
    boxes[0, 2] = boxes[0, 0] + .001
    result = manager.update(replace(obs, boxes_xyxy=boxes))
    # New tracks after the first empty frame require confirmation.
    result = manager.update(replace(obs, sequence=2, source_t_s=.08, boxes_xyxy=boxes))
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].keypoints_xy, obs.keypoints_xy[1])


@pytest.mark.parametrize('action', ['seek', 'replay', 'restart', 'switch_source'])
def test_capture_resets_fence_tracking_state(manager, action):
    cap = CaptureSessionManager(CLIP)
    try:
        cap.start()
        manager.handle_event(cap.queue.get())
        cap.pump()
        old = cap.queue.get()
        manager.update(observation(scores=(.9,), labels=(0,), session=old.session_id))
        if action == 'seek':
            cap.seek(30)
        elif action == 'switch_source':
            cap.switch_source(CLIP)
        else:
            getattr(cap, action)()
        event = cap.queue.get()
        assert isinstance(event, SessionResetEvent)
        manager.handle_event(event)
        cap.pump()
        new = cap.queue.get()
        assert new.session_id != old.session_id and new.sequence == 0 and new.source_t_s == 0
        sample = manager.update(observation(scores=(.9,), labels=(0,), session=new.session_id))[0]
        assert sample.person_key.generation == 1
        assert manager.last_matched_source_t_s == {sample.person_key.tracker_id: 0}
        with pytest.raises(ValueError, match='closed'):
            manager.update(observation(1, .04, (.9,), (0,), old.session_id))
    finally:
        cap.stop()


def test_session_change_and_eof_fence(manager):
    first = manager.update(observation())[0]
    changed = manager.update(observation(session='new'))[0]
    assert changed.person_key.session_id != first.person_key.session_id
    assert changed.person_key.generation == 1
    manager.handle_event(EndOfStreamEvent('new', 'local', 1, 0, 0))
    assert manager.tracker is None
    with pytest.raises(ValueError, match='closed'):
        manager.update(observation(1, .04, session='new'))


@pytest.mark.parametrize('width,height', [(640, 360), (1280, 720)])
@pytest.mark.parametrize('imgsz', [416, 512, 640])
def test_real_ultralytics_postprocess_unletterboxes_once(width, height, imgsz):
    attempts = []
    with guarded_runtime(ROOT, attempts):
        from vertebrate.cli import _prepare_ultralytics
        _prepare_ultralytics(ROOT)
        import torch
        from ultralytics.models.yolo.pose.predict import PosePredictor
        predictor = PosePredictor(overrides={'device': 'cpu'})
        predictor.model = SimpleNamespace(names={0: 'person'}, kpt_shape=(17, 3))
        frame = packet(width, height)
        gain = imgsz / width
        padding = (imgsz - height * gain) / 2
        raw = torch.zeros((1, 57))
        raw[0, :6] = torch.tensor([100 * gain, 50 * gain + padding,
                                   300 * gain, 300 * gain + padding, .9, 0])
        raw[0, 6:] = torch.tensor([200 * gain, 150 * gain + padding, .8] * 17)
        result = predictor.construct_result(raw, torch.zeros(1, 3, imgsz, imgsz), frame.copy_frame(), 'local')
        obs = observation_from_result(frame, result, .35)
        np.testing.assert_allclose(obs.boxes_xyxy, [[100, 50, 300, 300]], atol=1e-4)
        np.testing.assert_allclose(obs.keypoints_xy[0], np.tile([200, 150], (17, 1)), atol=1e-4)
        assert obs.keypoint_valid_mask.all()
    assert attempts == []


def test_bad_boxes_and_keypoint_mask():
    attempts = []
    with guarded_runtime(ROOT, attempts):
        from vertebrate.cli import _prepare_ultralytics
        _prepare_ultralytics(ROOT)
        import torch
        from ultralytics.engine.results import Results
        frame = packet()
        boxes = torch.tensor([[10, 10, 100, 200, .9, 0], [0, 0, 0, 10, .9, 0],
                              [float('nan'), 0, 10, 10, .9, 0], [0, 0, .001, 10, .9, 0]])
        kp = torch.tensor([[[20, 30, .9]] * 17] * 4)
        kp[0, :8] = torch.tensor([[float('nan'), 30, .9], [640, 20, .9], [10, 360, .9],
                                  [-1, 10, .9], [10, 10, .34], [10, 10, .35],
                                  [10, 10, float('inf')], [10, 10, 1.1]])
        result = Results(frame.copy_frame(), path='local', names={0: 'person'}, boxes=boxes, keypoints=kp)
        obs = observation_from_result(frame, result, .35)
        assert obs.boxes_xyxy.shape == (1, 4)
        assert obs.keypoint_valid_mask[0, :8].tolist() == [False, False, False, False, False, True, False, False]
        assert np.isfinite(obs.keypoints_xy).all()
        assert not obs.keypoints_xy.flags.writeable
    assert attempts == []


@pytest.mark.parametrize('width,height', [(640, 360), (1280, 720)])
def test_real_local_inference_empty_non_square(adapter, width, height):
    obs = adapter.infer(packet(width, height))
    assert obs.boxes_xyxy.shape == (0, 4)
    assert obs.keypoints_xy.shape == (0, 17, 2)
    assert obs.keypoint_valid_mask.shape == (0, 17)
    assert adapter.model_load_ms > 0


def test_real_capture_pose_tracking_full_clip_eof(adapter, manager):
    cap = CaptureSessionManager(CLIP)
    sequences = []
    try:
        cap.start()
        manager.handle_event(cap.queue.get())
        while True:
            cap.pump()
            item = cap.queue.get()
            if isinstance(item, EndOfStreamEvent):
                manager.handle_event(item)
                assert item.total_frames_emitted == 100
                assert item.last_source_t_s == pytest.approx(3.96)
                break
            sequences.append(item.sequence)
            obs = adapter.infer(item)
            assert manager.update(obs) == ()
        assert sequences == list(range(100))
        assert cap.queue.dropped_frames_count == 0
        assert cap.pump() is None
        assert cap.last_source_t_s == pytest.approx(3.96)
    finally:
        cap.stop()


@pytest.mark.parametrize('corrupt', [False, True])
def test_pose_rejects_missing_or_corrupt_asset_before_constructor(tmp_path, monkeypatch, corrupt):
    from vertebrate import cli
    (tmp_path / 'models').mkdir()
    data = json.loads((ROOT / 'models/manifest.json').read_text())
    (tmp_path / 'models/manifest.json').write_text(json.dumps(data))
    if corrupt:
        (tmp_path / 'models/yolo11n-pose.pt').write_bytes(b'corrupt')
    def forbidden(*args):
        pytest.fail('Ultralytics initialization must not precede asset validation')
    monkeypatch.setattr(cli, '_prepare_ultralytics', forbidden)
    with pytest.raises(AssetVerificationError):
        PoseAdapter(AppConfig(), tmp_path)


def test_guard_rejects_even_swallowed_network_attempts():
    import urllib.request
    attempts = []
    with pytest.raises(OfflineNetworkError, match='Unexpected offline attempts'):
        with guarded_runtime(ROOT, attempts):
            try:
                urllib.request.urlopen('https://invalid.example')
            except OfflineNetworkError:
                pass
    assert attempts == ['urllib.request.urlopen']


@pytest.mark.parametrize('sequence,time', [(0, .1), (1, 0)])
def test_tracking_rejects_out_of_order_observations(manager, sequence, time):
    manager.update(observation(time=.04))
    with pytest.raises(ValueError, match='increasing sequence'):
        manager.update(observation(sequence, time))


def test_benchmark_invalid_device_fails_without_output(tmp_path, capsys):
    from vertebrate.cli import main
    out = tmp_path / 'report.json'
    assert main(['benchmark', '--device', 'cuda', '--output', str(out)]) == 1
    assert 'cpu only' in json.loads(capsys.readouterr().out)['error']
    assert not out.exists()
