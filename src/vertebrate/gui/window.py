"""Local desktop workflow. All widgets live on the QApplication main thread."""
from datetime import datetime, timezone
import os
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox,
    QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..config import AppConfig, increment_revision
from ..context import Environment, build_environment, environment_at
from ..fsm import State
from .inbox import InboxWidget
from .video import VideoPane, assert_gui_thread
from .workers import ThreadedPipeline


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig | None = None, root_dir=None, *, source=None,
                 coordinator=None, capture_options=None):
        super().__init__()
        assert_gui_thread()
        self.config = config or AppConfig()
        self.root = Path(root_dir or Path.cwd()).resolve()
        self.environment = Environment()
        self.coordinator = coordinator or ThreadedPipeline(self.config, self.root)
        self.capture_options = capture_options or {}
        self._pending_start = False
        self._closing = False
        self._action = None
        self._rendered = None
        self._error = ''
        self.setWindowTitle('VERTEBRATE — Local observation')
        self.resize(1180, 920)
        # Windows' offscreen Qt backend exposes no fonts by default. Register
        # the OS fonts only in that case; no bundled or downloaded fonts.
        if not QFontDatabase.families() and os.name == 'nt':
            fonts = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'Fonts'
            for name in ('segoeui.ttf', 'times.ttf'):
                QFontDatabase.addApplicationFont(str(fonts / name))
        general = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        if QApplication.platformName() == 'offscreen' and 'Segoe UI' in QFontDatabase.families():
            general = QFont('Segoe UI', 10)
        self.setFont(general)
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        title = QLabel('VERTEBRATE')
        font = QFont('Times New Roman' if 'Times New Roman' in QFontDatabase.families() else 'serif', 23)
        font.setStyleHint(QFont.StyleHint.Serif)
        title.setFont(font)
        layout.addWidget(title)
        self.demo_label = QLabel('Local demo — no external emergency dispatch')
        self.demo_label.setStyleSheet('font-weight: bold; padding: 7px; background: #e7f0ed; color: #203c34')
        layout.addWidget(self.demo_label)
        controls = QHBoxLayout()
        self.source_kind = QComboBox()
        self.source_kind.addItems(['Local video', 'Camera'])
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText('Select a local video file')
        self.file_button = QPushButton('Choose video…')
        self.camera = QSpinBox()
        self.camera.setRange(0, 99)
        self.camera.setPrefix('Camera ')
        self.start_button, self.stop_button = QPushButton('Start'), QPushButton('Stop')
        self.pause_button, self.replay_button = QPushButton('Pause'), QPushButton('Replay')
        self.skeleton = QCheckBox('Skeleton')
        self.skeleton.setChecked(True)
        for widget in (self.source_kind, self.source_path, self.file_button, self.camera,
                       self.start_button, self.stop_button, self.pause_button, self.replay_button, self.skeleton):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.video = VideoPane()
        self.status_label = QLabel('Stopped')
        self.status_label.setWordWrap(True)
        self.people_label = QLabel('No observations')
        self.people_label.setWordWrap(True)
        self.warning_label = QLabel('')
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet('color: #924400; font-weight: bold')
        self.metrics = QLabel('Processing: —\nSource FPS: —\nSource elapsed: —')
        self.metrics.setMinimumHeight(65)
        self.revision_label = QLabel()
        self.profile = QComboBox()
        self.profile.addItems(['demo-3s', 'observation-10s'])
        self.profile.setCurrentText(self.config.profile)
        self.context_group = QGroupBox('Environmental context (optional)')
        self.context_group.setMinimumHeight(260)
        form = QFormLayout(self.context_group)
        self.context_source = QComboBox()
        self.context_source.addItems(['missing', 'manual_measured', 'manual_simulated'])
        self.applies_to = QComboBox()
        self.applies_to.addItems(['demo_scenario', 'current_scene', 'recorded_scene'])
        self.temperature, self.humidity = QLineEdit(), QLineEdit()
        self.temperature.setPlaceholderText('Optional °C')
        self.humidity.setPlaceholderText('Optional %')
        self.apply_context = QPushButton('Apply context')
        self.context_summary = QLabel('Context unavailable; observation remains enabled')
        self.context_summary.setWordWrap(True)
        self.context_summary.setMinimumHeight(65)
        for label, widget in [('Source', self.context_source), ('Applies to', self.applies_to),
                              ('Temperature °C', self.temperature), ('Humidity %', self.humidity)]:
            widget.setMinimumHeight(25)
            form.addRow(label, widget)
        form.addRow(self.apply_context)
        form.addRow(self.context_summary)
        side = QWidget()
        side.setMinimumWidth(310)
        side.setMinimumHeight(590)
        column = QVBoxLayout(side)
        for widget in (self.status_label, self.people_label, self.warning_label, self.metrics,
                       QLabel('Profile (editable while stopped)'), self.profile,
                       self.revision_label, self.context_group): column.addWidget(widget)
        self.diagnostics_toggle = QCheckBox('Show diagnostics')
        self.diagnostics = QLabel()
        self.diagnostics.setWordWrap(True)
        self.diagnostics.hide()
        self.diagnostics_toggle.toggled.connect(self.diagnostics.setVisible)
        column.addWidget(self.diagnostics_toggle)
        column.addWidget(self.diagnostics)
        column.addStretch()
        split = QSplitter()
        split.addWidget(self.video)
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(340)
        sidebar.setWidget(side)
        split.addWidget(sidebar)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        self.inbox = InboxWidget(self.root / self.config.runtime.outbox_dir)
        tabs = QTabWidget()
        tabs.addTab(self.inbox, 'Local inbox')
        tabs.setMaximumHeight(200)
        layout.addWidget(split, 3)
        layout.addWidget(tabs, 2)
        self.file_button.clicked.connect(self.choose_file)
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.pause_button.clicked.connect(self.pause_resume)
        self.replay_button.clicked.connect(lambda: self._command('replay'))
        self.skeleton.toggled.connect(self.video.set_skeleton)
        self.profile.currentTextChanged.connect(self.change_profile)
        self.apply_context.clicked.connect(self.change_context)
        self.source_kind.currentIndexChanged.connect(self._update_controls)
        if type(source) is int:
            self.source_kind.setCurrentIndex(1)
            self.camera.setValue(source)
        elif source is not None:
            self.source_path.setText(str(source))
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.poll)
        self.inbox.ensure_writer()
        self.timer.start()
        self._update_controls()

    @property
    def stopped(self):
        return not self._pending_start and not self.coordinator.running_threads and not self._action

    def _update_controls(self, *args):
        editable = self.stopped and not self._closing
        live = self.source_kind.currentIndex() == 1
        for widget in (self.source_kind, self.profile, self.context_group): widget.setEnabled(editable)
        self.source_path.setEnabled(editable and not live)
        self.file_button.setEnabled(editable and not live)
        self.camera.setEnabled(editable and live)
        self.start_button.setEnabled(editable)
        self.stop_button.setEnabled(not self.stopped and not self._closing)
        state = self.coordinator.status.state
        self.pause_button.setEnabled((not live and state in ('running', 'paused') or state == 'storage_fault') and self._action is None and not self._closing)
        self.pause_button.setText('Resume' if state in ('paused', 'storage_fault') else 'Pause')
        self.replay_button.setEnabled(not live and state in ('running', 'paused', 'eof') and self._action is None and not self._closing)
        self.revision_label.setText(f'Configuration revision {self.config.revision}')

    def choose_file(self):
        if not self.stopped: return
        path, _ = QFileDialog.getOpenFileName(self, 'Choose local video', str(self.root),
                                             'Videos (*.avi *.mp4 *.mov *.mkv);;All files (*)')
        if path: self.source_path.setText(path)

    def change_profile(self, profile):
        if not self.stopped or profile == self.config.profile:
            self.profile.blockSignals(True)
            self.profile.setCurrentText(self.config.profile)
            self.profile.blockSignals(False)
            return
        self.config = increment_revision(self.config, profile=profile,
            calibration={'stillness_duration_s': 3. if profile == 'demo-3s' else 10.})
        self._update_controls()

    def change_context(self):
        if not self.stopped: return
        def optional(text):
            try: return float(text) if text.strip() else None
            except ValueError: return None
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        self.environment = build_environment(optional(self.temperature.text()), optional(self.humidity.text()),
            source=self.context_source.currentText(), applies_to=self.applies_to.currentText(),
            observed_at_utc=now, now_utc=now, staleness_minutes=self.config.runtime.context_staleness_minutes)
        self.config = increment_revision(self.config)
        self._update_controls()

    def start_capture(self):
        if not self.stopped or self._closing: return
        if self.source_kind.currentIndex() == 0 and not self.source_path.text().strip():
            self._error = 'Choose a local video first'
            return
        self._error = ''
        self._pending_start = True
        self._rendered = None
        self.video.set_result(None)
        self._update_controls()

    def stop_capture(self):
        self._pending_start = False
        self._action = self.coordinator.stop()
        self._update_controls()

    def _command(self, name):
        if self._action is None and not self.stopped:
            self._action = getattr(self.coordinator, name)()
            self._update_controls()

    def pause_resume(self):
        self._error = ''
        self._command('resume' if self.coordinator.status.state in ('paused', 'storage_fault') else 'pause')

    def poll(self):
        assert_gui_thread()
        self.inbox.poll()
        if self._action is not None and self._action.done():
            try: self._action.result()
            except Exception as exc: self._error = str(exc)
            self._action = None
        if self._pending_start and self.inbox.release():
            self._pending_start = False
            self.coordinator.config = self.config
            self.coordinator.environment = self.environment
            source = self.camera.value() if self.source_kind.currentIndex() else self.source_path.text().strip()
            try:
                self._action = self.coordinator.start(source, **self.capture_options)
                self.inbox.bind(self.coordinator.storage)
            except Exception as exc:
                self._error = str(exc)
        if self._closing:
            writer = self.inbox.writer or self.coordinator.storage
            if not self.coordinator.running_threads and self.inbox.release():
                if writer and writer.unsaved_incident_ids:
                    self._closing = False
                    self._error = 'Unsaved incidents retained. Retry save before closing.'
                    self.inbox.ensure_writer(writer)
                    return
                self.timer.stop()
                self.close()
                return
        elif self.stopped:
            if not self.inbox.owns_writer:
                self.inbox.release()
                retained = self.coordinator.storage if self.coordinator.storage and self.coordinator.storage.unsaved_incident_ids else None
                self.inbox.ensure_writer(retained)
        result = self.coordinator.latest.get()
        if result is not self._rendered:
            self.video.set_result(result)
            self._rendered = result
        state = self.coordinator.status
        self.status_label.setText(self._error or state.message or state.state.replace('_', ' ').capitalize())
        self.diagnostics.setText(f'Worker state: {state.state}\nFrames processed: {state.frames}\nDropped live frames: {state.dropped_frames}')
        if result is not None:
            source_fps = f'{result.source_fps:.1f}' if result.source_fps > 0 else 'unavailable'
            self.metrics.setText(f'Processing: {result.processing_fps:.1f} FPS (vision)\nSource FPS: {source_fps}\nSource elapsed: {result.frame.source_t_s:.2f} s')
            count = len(result.observation.boxes_xyxy) if result.observation is not None else 0
            self.warning_label.setText('Outside tested people-count limit (> 2 detected people)' if count > 2 else '')
            elapsed = dict(result.stillness)
            lines = []
            for decision in result.decisions:
                if decision.state == State.ALERTED:
                    message = 'Suspected fall with stillness — confirmed locally'
                elif decision.state in (State.DESCENT, State.VERIFYING_DOWN, State.STILLNESS):
                    message = ('Observation ended — inconclusive' if state.state == 'eof'
                               else 'Possible fall — observing')
                else:
                    message = 'Monitoring' if decision.state == State.MONITORING else 'Acquiring upright baseline'
                lines.append(f'Person {decision.person_key.tracker_id}: {message}' +
                    (f' · stillness {elapsed[decision.person_key]:.1f}/{self.config.calibration.stillness_duration_s:g} s'
                     if decision.person_key in elapsed else '') +
                    (f' · {decision.reason.replace("_", " ")}' if decision.reason else ''))
            self.people_label.setText('\n'.join(lines) or 'No people detected')
        else:
            self.metrics.setText('Processing: —\nSource FPS: —\nSource elapsed: —')
            self.people_label.setText('No active observations')
            self.warning_label.clear()
        env = environment_at(self.environment, datetime.now(timezone.utc).isoformat(), self.config.runtime.context_staleness_minutes)
        self.context_summary.setText((f'Heat index {env.heat_index_c:.1f} °C' if env.heat_index_c is not None else
            f'Heat index unavailable: {(env.heat_index_reason or "missing context").replace("_", " ")}') + (' · stale (30+ minutes)' if env.stale else '') +
            '\nContext does not gate detection')
        self._update_controls()

    def closeEvent(self, event):
        if self.inbox.owns_writer and any(r.status == 'save_failed' for r in self.inbox.writer.records):
            event.ignore()
            self._error = 'Unsaved incidents retained. Retry save before closing.'
            return
        if self.coordinator.running_threads or self._pending_start or (self.inbox.owns_writer and self.inbox.writer._thread.is_alive()):
            event.ignore()
            if not self._closing:
                self._closing = True
                self.stop_capture()
                if self.inbox.owns_writer: self.inbox.writer.request_stop()
            return
        self.inbox.release()
        self.timer.stop()
        event.accept()


def launch_gui(config, source=None, root_dir=None):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(config, root_dir, source=source)
    window.show()
    return app.exec()
