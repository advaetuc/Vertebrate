"""Offscreen pytest-qt GUI tests; injected poses are not accuracy evidence."""
from dataclasses import replace
import json
from pathlib import Path
import re
from threading import Event, get_ident
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt

from vertebrate.config import AppConfig
from vertebrate.contracts import FramePacket, PersonTrackKey, PoseObservation, SourceKind
from vertebrate.fsm import Decision, State
from vertebrate.gui.window import MainWindow
from vertebrate.gui.workers import DisplayResult, ThreadedPipeline
from vertebrate.incidents import incident_from_dict
from vertebrate.pipeline import HeadlessPipeline

ROOT = Path(__file__).resolve().parents[2]


class Reader:
    fps, frame_count = 25., 1000
    def __init__(self, source): self.index, self.owners, self.released = 0, [], False
    def open(self): self.owners.append(get_ident())
    def read(self):
        self.owners.append(get_ident())
        time.sleep(.003)
        if self.index == self.frame_count: return None, None
        self.index += 1
        return np.full((90, 160, 3), self.index % 255, np.uint8), (self.index-1)/25
    def seek(self, index): self.index = index
    def release(self):
        self.owners.append(get_ident())
        self.released = True


class Pose:
    def infer(self, f):
        return PoseObservation(f.session_id, f.sequence, f.source_t_s, f.width, f.height,
            np.empty((0, 4)), np.empty(0), np.empty((0, 17, 2)), np.empty((0, 17)), np.empty((0, 17), bool))


class Tracker:
    def handle_event(self, event): pass
    def update(self, obs): return ()


@pytest.fixture
def window(qtbot, tmp_path):
    cfg = AppConfig()
    cfg = replace(cfg, runtime=replace(cfg.runtime, outbox_dir=str(tmp_path / 'outbox')))
    pipe = ThreadedPipeline(cfg, ROOT, pipeline_factory=lambda c, r: HeadlessPipeline(c, r, pose=Pose(), tracker=Tracker()))
    win = MainWindow(cfg, ROOT, source='synthetic', coordinator=pipe, capture_options={'source_factory': Reader})
    # Own teardown here: pytest-qt's automatic deleteLater after close would
    # delete a window whose asynchronous cooperative close is still pending.
    win.show()
    yield win
    win.close()
    qtbot.waitUntil(lambda: not win.isVisible(), timeout=15000)
    assert pipe.running_threads == 0
    assert pipe.status.network_attempts == ()


def start(qtbot, win):
    qtbot.mouseClick(win.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: win.coordinator.status.state == 'running' and win._action is None, timeout=10000)


def stop(qtbot, win):
    qtbot.mouseClick(win.stop_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: win.stopped, timeout=10000)


def display(sequence=0, people=1, state=State.MONITORING):
    frame = FramePacket('display', sequence, SourceKind.VIDEO, 'synthetic', 160, 90,
                        np.full((90, 160, 3), (10, 20, 30), np.uint8), sequence/25, 100+sequence)
    obs = PoseObservation('display', sequence, sequence/25, 160, 90,
        np.tile([10., 10., 70., 80.], (people, 1)), np.ones(people),
        np.full((people, 17, 2), 40.), np.ones((people, 17)), np.ones((people, 17), bool))
    decisions = tuple(Decision(PersonTrackKey('display', i, 1), sequence, sequence/25, state, '') for i in range(people))
    return DisplayResult(frame, decisions, obs, 32., 25.)


def test_window_layout_persistent_notice_and_stopped_controls(window):
    assert window.demo_label.text() == 'Local demo — no external emergency dispatch'
    assert window.video.isVisible() and window.inbox.isVisible()
    assert window.start_button.isEnabled() and window.profile.isEnabled()
    assert not window.pause_button.isEnabled()


def test_twenty_start_stop_cycles_in_one_window(qtbot, window):
    readers = []
    def factory(source):
        reader = Reader(source)
        readers.append(reader)
        return reader
    window.capture_options['source_factory'] = factory
    for _ in range(20):
        start(qtbot, window)
        qtbot.waitUntil(lambda: window.video.result is not None)
        stop(qtbot, window)
        assert window.coordinator.running_threads == 0
    assert len(readers) == 20 and all(r.released for r in readers)
    assert all(len(set(r.owners)) == 1 and r.owners[0] != get_ident() for r in readers)


def test_config_and_context_changes_increment_revision_only_while_stopped(qtbot, window):
    revision = window.config.revision
    window.profile.setCurrentText('observation-10s')
    assert window.config.revision == revision+1
    assert window.config.calibration.stillness_duration_s == 10
    window.context_source.setCurrentText('manual_simulated')
    window.temperature.setText('32')
    window.humidity.setText('70')
    qtbot.mouseClick(window.apply_context, Qt.MouseButton.LeftButton)
    assert window.config.revision == revision+2
    assert window.environment.heat_index_c is not None
    start(qtbot, window)
    frozen = window.config
    assert not window.profile.isEnabled() and not window.context_group.isEnabled()
    window.change_context()
    window.change_profile('demo-3s')
    assert window.config is frozen and window.profile.currentText() == 'observation-10s'
    assert window.coordinator.config == frozen
    assert window.coordinator.environment == window.environment


def test_invalid_context_does_not_gate_start(qtbot, window):
    window.context_source.setCurrentText('manual_measured')
    window.temperature.setText('NaN')
    window.humidity.setText('999')
    window.change_context()
    assert window.environment.heat_index_c is None
    start(qtbot, window)
    assert window.coordinator.status.state == 'running'


def test_sequence_aligned_timer_poll_owns_image_and_latest_slot(window, qtbot):
    first = display(1)
    window.coordinator.latest.set(first)
    qtbot.waitUntil(lambda: window.video.result is first)
    assert window.video.image.pixelColor(0, 0).getRgb()[:3] == (30, 20, 10)
    copy = first.frame.copy_frame()
    copy.fill(255)
    assert window.video.image.pixelColor(0, 0).getRgb()[:3] == (30, 20, 10)
    last = display(100)
    for i in range(2, 100): window.coordinator.latest.set(display(i))
    window.coordinator.latest.set(last)
    qtbot.waitUntil(lambda: window.video.result is last)
    assert window.video.result.observation.sequence == window.video.result.frame.sequence == 100
    assert '25.0' in window.metrics.text() and '32.0' in window.metrics.text()
    qtbot.mouseClick(window.skeleton, Qt.MouseButton.LeftButton)
    assert not window.video.skeleton
    assert not window.video.grab().isNull()


def test_mismatched_overlay_rejected():
    value = display(1)
    with pytest.raises(ValueError, match='sequence'):
        DisplayResult(value.frame, value.decisions, display(2).observation)


@pytest.mark.parametrize('state,text', [(State.DESCENT, 'Possible fall — observing'),
    (State.VERIFYING_DOWN, 'Possible fall — observing'), (State.STILLNESS, 'Possible fall — observing'),
    (State.ALERTED, 'confirmed locally')])
def test_candidate_and_confirmed_states_are_distinct(window, qtbot, state, text):
    window.coordinator.latest.set(display(1, 3, state))
    qtbot.waitUntil(lambda: text in window.people_label.text())
    assert 'Outside tested people-count limit' in window.warning_label.text()
    if state != State.ALERTED: assert 'confirmed locally' not in window.people_label.text()


def test_pause_replay_and_source_controls(qtbot, window):
    start(qtbot, window)
    qtbot.mouseClick(window.pause_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.coordinator.status.state == 'paused' and window._action is None)
    frame = window.video.result
    qtbot.wait(70)
    assert window.video.result is frame
    qtbot.mouseClick(window.pause_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.coordinator.status.state == 'running' and window._action is None)
    old = window.coordinator.capture.session_id
    qtbot.mouseClick(window.replay_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.coordinator.capture.session_id != old and window._action is None)
    qtbot.waitUntil(lambda: window.video.result is not None)
    assert window.video.result.frame.session_id != old
    stop(qtbot, window)
    window.source_kind.setCurrentIndex(1)
    start(qtbot, window)
    assert not window.pause_button.isEnabled() and not window.replay_button.isEnabled()


def test_close_waits_for_camera_owner_and_keeps_timeout_visible(qtbot, window):
    entered, release = Event(), Event()
    class Blocked(Reader):
        def read(self):
            entered.set()
            release.wait(8)
            return super().read()
    reader = Blocked('synthetic')
    window.capture_options['source_factory'] = lambda s: reader
    try:
        start(qtbot, window)
        qtbot.waitUntil(entered.is_set)
        window.close()
        qtbot.wait(80)
        assert window.isVisible() and not reader.released
        window.coordinator._stop_deadline = time.monotonic()-.1
        qtbot.waitUntil(lambda: 'timed out' in window.status_label.text())
        assert window.isVisible()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=10000)
    assert reader.released


def test_local_inbox_open_snapshot_acknowledge_and_restart(qtbot, window):
    section = (ROOT / 'docs/03-validated-blueprint-v2.md').read_text(encoding='utf-8').split('## 10. Incident schema')[1]
    incident = incident_from_dict(json.loads(re.search(r'```json\s*(.*?)\s*```', section, re.S).group(1)))
    qtbot.waitUntil(lambda: window.inbox.writer.ready)
    future = window.inbox.writer.submit(incident, window.config, np.zeros((90, 160, 3), np.uint8))
    qtbot.waitUntil(future.done)
    assert future.result().delivery.status == 'saved'
    qtbot.waitUntil(lambda: window.inbox.items.count() == 1)
    qtbot.mouseClick(window.inbox.open_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.inbox.opened_incident is not None)
    assert '"schema_version": "1.0"' in window.inbox.metadata.toPlainText()
    assert not window.inbox.snapshot.pixmap().isNull()
    qtbot.mouseClick(window.inbox.ack_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.inbox.message.text() == 'Acknowledgement saved')
    assert window.inbox.opened_incident.acknowledgement.status == 'acknowledged'
    start(qtbot, window)
    stop(qtbot, window)
    qtbot.waitUntil(lambda: window.inbox.writer and window.inbox.writer.ready)
    assert window.inbox.writer.record(incident.incident_id).incident.acknowledgement.status == 'acknowledged'


def test_source_error_is_visible_without_closing_window(qtbot, window):
    class BadReader(Reader):
        def open(self): raise OSError('injected source unavailable')
    window.capture_options['source_factory'] = BadReader
    qtbot.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: 'source unavailable' in window.status_label.text())
    qtbot.waitUntil(lambda: window.stopped and window.start_button.isEnabled())
    assert window.isVisible() and window.start_button.isEnabled()


def test_cli_default_routes_to_gui_and_keeps_tools_lazy(monkeypatch):
    from vertebrate.cli import main
    calls = []
    monkeypatch.setattr('vertebrate.gui.window.launch_gui', lambda cfg, source, root: calls.append((cfg, source, root)) or 0)
    assert main(['--config', str(ROOT / 'config/demo.json'), '--source', '2']) == 0
    assert calls[0][1] == 2 and calls[0][0].profile == 'demo-3s'


def test_real_local_fixture_runs_in_desktop_and_exits_offline(qtbot, tmp_path):
    cfg = AppConfig()
    cfg = replace(cfg, runtime=replace(cfg.runtime, outbox_dir=str(tmp_path / 'real-outbox')))
    win = MainWindow(cfg, ROOT, source=ROOT / 'tests/fixtures/videos/dev_empty_scene.avi')
    win.show()
    try:
        qtbot.mouseClick(win.start_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: win.coordinator.status.state in ('eof', 'error'), timeout=60000)
        assert win.coordinator.status.state == 'eof', win.coordinator.status
        qtbot.waitUntil(lambda: win.video.result is not None and win.video.result.frame.sequence == 99)
        assert win.video.result.observation.sequence == 99
        assert win.video.result.source_fps == 25.
        assert win.coordinator.status.frames == 100
        assert not win.video.image.isNull()
        assert win.coordinator.status.dropped_frames == 0
    finally:
        win.close()
        qtbot.waitUntil(lambda: not win.isVisible(), timeout=15000)
    assert win.coordinator.running_threads == 0
    assert win.coordinator.status.network_attempts == ()


def test_eof_closes_candidate_status_without_confirming(window, qtbot):
    window.coordinator.latest.set(display(20, 1, State.STILLNESS))
    window.coordinator.latest.update_status(state='eof')
    qtbot.waitUntil(lambda: 'inconclusive' in window.people_label.text())
    assert 'confirmed locally' not in window.people_label.text()


def test_stillness_progress_and_reset_clear_overlay(window, qtbot):
    result = display(20, 1, State.STILLNESS)
    result = replace(result, stillness=((result.decisions[0].person_key, 1.2),))
    window.coordinator.latest.set(result)
    qtbot.waitUntil(lambda: '1.2/3 s' in window.people_label.text())
    window.coordinator.latest.set(None)
    qtbot.waitUntil(lambda: window.video.result is None)
    assert window.video.image.isNull()


def test_main_thread_only_paint_updates(window):
    from threading import Thread
    errors = []
    def other_thread():
        try: window.video.set_result(display())
        except RuntimeError as exc: errors.append(str(exc))
    thread = Thread(target=other_thread)
    thread.start()
    thread.join(1)
    assert errors == ['Widgets require the Qt main thread']


def test_failed_save_is_visible_retained_on_close_and_retryable(window, qtbot, monkeypatch):
    section = (ROOT / 'docs/03-validated-blueprint-v2.md').read_text(encoding='utf-8').split('## 10. Incident schema')[1]
    incident = incident_from_dict(json.loads(re.search(r'```json\s*(.*?)\s*```', section, re.S).group(1)))
    qtbot.waitUntil(lambda: window.inbox.writer.ready)
    writer = window.inbox.writer
    original = writer._atomic_write
    def fail(destination, data):
        if destination.name == 'incident.json': raise OSError('injected disk failure')
        return original(destination, data)
    monkeypatch.setattr(writer, '_atomic_write', fail)
    try:
        future = writer.submit(incident, window.config, np.zeros((90, 160, 3), np.uint8))
        qtbot.waitUntil(future.done)
        assert future.exception() is not None
        qtbot.waitUntil(lambda: window.inbox.items.count() == 1)
        assert 'save_failed' in window.inbox.items.item(0).text()
        window.close()
        qtbot.waitUntil(lambda: 'Retry save before closing' in window.status_label.text())
        assert window.isVisible() and writer.unsaved_incident_ids
    finally:
        monkeypatch.setattr(writer, '_atomic_write', original)
        qtbot.mouseClick(window.inbox.retry_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not writer.unsaved_incident_ids)
    qtbot.waitUntil(lambda: window.inbox.message.text() == 'Incident saved')
