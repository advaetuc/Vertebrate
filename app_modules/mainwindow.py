"""MainWindow class."""
from .common import *  # shared Qt imports, constants, and helpers
from .alarmsound import AlarmSound
from .camerathread import CameraThread
from .evidencemeter import EvidenceMeter
from .falllog import FallLog
from .falllogdialog import FallLogDialog
from .locationfeed import LocationFeed
from .logomark import LogoMark
from .stattile import StatTile
from .videoview import VideoView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._stopping = False
        self._closing = False
        self._error = None
        self._state = None
        self._state_detail = None
        self._flash = False
        self._people = 0
        self._frame_times = deque(maxlen=30)
        self._falls_today = 0
        self.location = LocationFeed(self)
        self.fall_log = FallLog(device_name())
        self.log_dialog = None
        self.alarm = AlarmSound(self)
        self.settings = QSettings("Vertebrate", "Vertebrate")  # per-user, per-device preferences
        self.setWindowTitle("Vertebrate · Fall Monitor")
        self.resize(1360, 860)
        self.setMinimumSize(1000, 640)

        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 18, 28, 16)
        outer.setSpacing(18)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(20)
        body.addLayout(self._build_main(), 1)
        body.addWidget(self._build_sidebar())
        outer.addLayout(body, 1)
        footer = make_label("Processing stays on this device  ·  Local alarm only — no external emergency dispatch",
                            12, obj="caption")
        outer.addWidget(footer)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLESHEET)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.log_button.clicked.connect(self.open_log)
        self.silence_button.clicked.connect(self.silence_alarm)
        self.alarm_delay.valueChanged.connect(self.alarm_delay_changed)
        self.flash_timer = QTimer(self)  # pulses fall/alarm visuals
        self.flash_timer.setInterval(450)
        self.flash_timer.timeout.connect(self.flash_alert)
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.setInterval(5000)
        self.stop_timer.timeout.connect(self.stop_timeout)
        # Checks the ongoing fall against the alarm delay independently of frame rate.
        self.alarm_timer = QTimer(self)
        self.alarm_timer.setInterval(250)
        self.alarm_timer.timeout.connect(self.check_alarm)
        self.clock_timer = QTimer(self)  # header clock, location card, stats decay
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self.tick)
        self.clock_timer.start()

        self.update_log_button()
        self.alarm_delay_changed(self.alarm_delay.value(), save=False)
        self.set_state("idle")
        self.tick()

    # -- layout --------------------------------------------------------------
    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 16)
        row.setSpacing(14)
        row.addWidget(LogoMark())
        names = QVBoxLayout()
        names.setSpacing(1)
        names.addWidget(make_label("VERTEBRATE", 18, QFont.Weight.Bold, spacing=3.0))
        names.addWidget(make_label("Fall-posture monitoring", 12, obj="muted"))
        row.addLayout(names)
        row.addStretch()
        device = make_label(self.fall_log.device, 12, obj="chip")
        device.setToolTip("This device. Its fall log and alarm setting are kept separately.")
        row.addWidget(device)
        self.clock_label = make_label("", 13, mono=True, obj="muted")
        row.addWidget(self.clock_label)
        row.addSpacing(4)
        self.pill = make_label("", 12, QFont.Weight.Bold, spacing=1.0)
        row.addWidget(self.pill)
        return header

    def _build_main(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(14)

        self.banner = QFrame()
        self.banner.setObjectName("banner")
        self.banner.setVisible(False)
        banner = QHBoxLayout(self.banner)
        banner.setContentsMargins(16, 12, 12, 12)
        banner.setSpacing(14)
        self.banner_icon = make_label("!", 20, QFont.Weight.Black)
        self.banner_icon.setFixedSize(36, 36)
        self.banner_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self.banner_title = make_label("", 17, QFont.Weight.Bold)
        self.banner_detail = make_label("", 13, wrap=True)
        texts.addWidget(self.banner_title)
        texts.addWidget(self.banner_detail)
        self.silence_button = QPushButton("Silence alarm")
        self.silence_button.setObjectName("silence")
        self.silence_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.silence_button.setVisible(False)
        banner.addWidget(self.banner_icon)
        banner.addLayout(texts, 1)
        banner.addWidget(self.silence_button)
        column.addWidget(self.banner)

        card = QFrame()
        card.setObjectName("card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(16, 14, 16, 16)
        inner.setSpacing(12)
        top = QHBoxLayout()
        top.addWidget(caption("Live view"))
        top.addStretch()
        top.addWidget(make_label("Camera 0  ·  YOLO11n pose  ·  on-device CPU", 12, obj="muted"))
        inner.addLayout(top)
        self.video = VideoView()
        inner.addWidget(self.video, 1)
        column.addWidget(card, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.start_button = QPushButton("Start monitoring")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("ghost")
        self.stop_button.setEnabled(False)
        self.log_button = QPushButton("Fall log")
        self.log_button.setObjectName("ghost")
        self.log_button.setToolTip(f"Falls recorded on {self.fall_log.device}")
        for button in (self.start_button, self.stop_button, self.log_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        controls.addWidget(self.log_button)
        column.addLayout(controls)
        return column

    def _card(self, title: str):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(caption(title))
        return card, layout

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet("background: transparent;")
        inner = QWidget()
        inner.setObjectName("sideInner")
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(14)

        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        status = QVBoxLayout(self.status_card)
        status.setContentsMargins(18, 16, 18, 18)
        status.setSpacing(6)
        status.addWidget(caption("Status"))
        self.status_title = make_label("", 26, QFont.Weight.Bold)
        self.status_detail = make_label("", 13, obj="muted", wrap=True)
        status.addWidget(self.status_title)
        status.addWidget(self.status_detail)
        column.addWidget(self.status_card)

        card, layout = self._card("Fall evidence")
        self.meter = EvidenceMeter()
        layout.addWidget(self.meter)
        row = QHBoxLayout()
        self.evidence_label = make_label(f"0 of {CONSECUTIVE_FRAMES} frames", 13, QFont.Weight.DemiBold, mono=True)
        row.addWidget(self.evidence_label)
        row.addStretch()
        row.addWidget(make_label(f"Alert after {CONSECUTIVE_FRAMES} in a row", 12, obj="muted"))
        layout.addLayout(row)
        column.addWidget(card)

        card, layout = self._card("Live stats")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.stat_people = StatTile("People in view")
        self.stat_fps = StatTile("Frames / second")
        self.stat_today = StatTile("Falls today")
        self.stat_alarm = StatTile("Alarm in")
        grid.addWidget(self.stat_people, 0, 0)
        grid.addWidget(self.stat_fps, 0, 1)
        grid.addWidget(self.stat_today, 1, 0)
        grid.addWidget(self.stat_alarm, 1, 1)
        layout.addLayout(grid)
        column.addWidget(card)

        card, layout = self._card("Location")
        self.location_coords = make_label("—", 15, QFont.Weight.DemiBold, mono=True, wrap=True)
        self.location_detail = make_label("", 12, obj="muted", wrap=True)
        layout.addWidget(self.location_coords)
        layout.addWidget(self.location_detail)
        column.addWidget(card)

        card, layout = self._card("Emergency alarm")
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(make_label("Ring if fallen for", 14))
        row.addStretch()
        minus = QPushButton("−")
        plus = QPushButton("+")
        self.alarm_delay = QSpinBox()
        self.alarm_delay.setRange(0, ALARM_MAX_S)
        self.alarm_delay.setSingleStep(5)
        self.alarm_delay.setSuffix(" s")
        self.alarm_delay.setSpecialValueText("Off")  # shown at 0
        self.alarm_delay.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.alarm_delay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alarm_delay.setToolTip("Seconds a fall may last before the siren rings. 0 turns the alarm off.")
        try:
            saved = int(self.settings.value("alarm_after_s", ALARM_DEFAULT_S))
        except (TypeError, ValueError):
            saved = ALARM_DEFAULT_S
        self.alarm_delay.setValue(min(max(saved, 0), ALARM_MAX_S))
        for button, steps in ((minus, -1), (plus, 1)):
            button.setObjectName("step")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAutoRepeat(True)
            button.clicked.connect(lambda _checked=False, s=steps: self.alarm_delay.stepBy(s))
        row.addWidget(minus)
        row.addWidget(self.alarm_delay)
        row.addWidget(plus)
        layout.addLayout(row)
        self.alarm_hint = make_label("", 12, obj="muted", wrap=True)
        layout.addWidget(self.alarm_hint)
        column.addWidget(card)

        column.addStretch()
        scroll.setWidget(inner)
        return scroll

    # -- state & visuals -----------------------------------------------------
    def set_state(self, state: str, detail: str | None = None) -> None:
        self._state, self._state_detail = state, detail
        if state in ("fall", "alarm"):
            if not self.flash_timer.isActive():
                self._flash = True
                self.flash_timer.start()
        else:
            self.flash_timer.stop()
            self._flash = False
        self.apply_state()

    def _fall_detail(self) -> str:
        current = self.fall_log.current
        fallen_for = format_duration(self.fall_log.ongoing_seconds())
        delay = self.alarm_delay.value()
        if current is None:
            return "Fall posture detected."
        if delay == 0:
            return f"Fallen for {fallen_for}. The alarm is off."
        if current["alarm_rang"] is None:
            return f"Fallen for {fallen_for}. Alarm in {self._alarm_remaining()} s."
        return f"Fallen for {fallen_for}. Alarm silenced."

    def _alarm_remaining(self) -> int:
        return max(0, int(self.alarm_delay.value() - self.fall_log.ongoing_seconds() + 0.999))

    def apply_state(self) -> None:
        state = self._state
        pill_text, color = STATE_STYLE[state]
        if state == "alarm":
            color = DANGER if self._flash else DANGER_DEEP
        title, detail = {
            "idle": ("Standby", "Camera is off. Press Start monitoring to begin."),
            "starting": ("Starting…", "Loading the pose model and opening the camera."),
            "monitoring": ("All clear", self._monitoring_detail()),
            "fall": ("Fall detected", self._fall_detail()),
            "alarm": ("Emergency alarm",
                      f"Fallen for {format_duration(self.fall_log.ongoing_seconds())}. Siren ringing on this device."),
            "stopping": ("Stopping…", "Releasing the camera."),
            "error": ("Something went wrong", "Check the message below and try again."),
        }[state]
        if self._state_detail:
            detail = self._state_detail

        self.pill.setText(f"●  {pill_text}")
        self.pill.setStyleSheet(
            f"QLabel {{ color: {color}; background: {rgba(color, .12)}; border: 1px solid {rgba(color, .38)};"
            f" border-radius: 12px; padding: 5px 12px; }}")
        self.status_title.setText(title)
        self.status_title.setStyleSheet(f"color: {TEXT if state in ('idle', 'stopping') else color};")
        self.status_detail.setText(detail)
        self.status_card.setStyleSheet(
            f"QFrame#statusCard {{ border: 1px solid {rgba(color, .40)}; border-radius: 16px;"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {rgba(color, .18)},"
            f" stop:0.65 {SURFACE}, stop:1 {SURFACE}); }}")

        live = state in ("monitoring", "fall", "alarm")
        border = {"monitoring": (QColor(OK), 1.5), "fall": (QColor(WARN), 2.5 if self._flash else 1.5),
                  "alarm": (QColor(color), 3.0), "error": (QColor(DANGER), 1.5)}.get(state, (QColor(BORDER), 1.0))
        if state == "monitoring":
            border[0].setAlpha(110)
        self.video.set_border(border[0], border[1], live)

        if state == "fall":
            self.banner_title.setText("Fall detected")
            self.banner_detail.setText(self._fall_detail())
            self.banner.setStyleSheet(
                f"QFrame#banner {{ background: {rgba(WARN, .13)}; border: 1px solid {rgba(WARN, .55)};"
                f" border-radius: 14px; }} QLabel {{ color: {TEXT}; }}")
            self.banner_icon.setStyleSheet(f"color: #1A0E06; background: {WARN}; border-radius: 18px;")
        elif state == "alarm":
            self.banner_title.setText(f"EMERGENCY ALARM — fallen for {format_duration(self.fall_log.ongoing_seconds())}")
            self.banner_detail.setText("A siren is ringing on this device. Check on the person, then silence it.")
            self.banner.setStyleSheet(
                f"QFrame#banner {{ background: {color}; border: 1px solid {rgba('#FFFFFF', .25)};"
                f" border-radius: 14px; }} QLabel {{ color: #FFFFFF; }}")
            self.banner_icon.setStyleSheet(f"color: {DANGER_DEEP}; background: #FFFFFF; border-radius: 18px;")
        self.banner.setVisible(state in ("fall", "alarm"))
        self.silence_button.setVisible(state == "alarm" and self.alarm.ringing)
        self.refresh_stats()

    def _monitoring_detail(self) -> str:
        people = "No one in view" if self._people == 0 else (
            "1 person in view" if self._people == 1 else f"{self._people} people in view")
        return f"{people}. No fall posture detected."

    def refresh_live_text(self) -> None:
        """Per-frame update: text only. Stylesheets change only when the state does."""
        if self._state == "monitoring" and not self._state_detail:
            self.status_detail.setText(self._monitoring_detail())
        self.refresh_stats()

    def refresh_stats(self) -> None:
        running = self.worker is not None and self._state in ("monitoring", "fall", "alarm")
        self.stat_people.set(str(self._people) if running else "—")
        now = time.monotonic()
        recent = [t for t in self._frame_times if now - t <= 2.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        self.stat_fps.set(f"{fps:.1f}" if running else "—")
        self.stat_today.set(str(self._falls_today), WARN if self._falls_today else None)
        delay = self.alarm_delay.value()
        current = self.fall_log.current
        if delay == 0:
            self.stat_alarm.set("Off", FAINT)
        elif self.alarm.ringing:
            self.stat_alarm.set("Now", DANGER)
        elif current is not None and current["alarm_rang"] is None:
            self.stat_alarm.set(f"{self._alarm_remaining()} s", WARN)
        else:
            self.stat_alarm.set(f"{delay} s")
        self.evidence_label.setText(f"{self.meter.value} of {CONSECUTIVE_FRAMES} frames")

    def refresh_location(self) -> None:
        fix = self.location.current_fix()
        if fix is None:
            running = self.worker is not None
            self.location_coords.setText("—")
            self.location_detail.setText(self.location.problem_text() if running
                                         else "Starts with monitoring. Shown on the video as a watermark.")
            return
        lat, lon, accuracy, source = fix
        self.location_coords.setText(
            f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}\n{abs(lon):.5f}° {'E' if lon >= 0 else 'W'}")
        detail = f"±{accuracy:.0f} m  ·  via {source}" if accuracy else f"via {source}"
        self.location_detail.setText(detail)

    @Slot()
    def tick(self):
        self.clock_label.setText(QDateTime.currentDateTime().toString("ddd d MMM  ·  HH:mm:ss"))
        self.refresh_location()
        if self._state in ("fall", "alarm"):
            return  # flash_alert already refreshes these twice a second
        self.refresh_stats()

    @Slot()
    def flash_alert(self):
        self._flash = not self._flash
        self.apply_state()

    # -- camera lifecycle ----------------------------------------------------
    @Slot()
    def start_camera(self):
        if self.worker is not None:
            return
        self._error = None
        self._stopping = False
        self._frame_times.clear()
        self._people = 0
        self.meter.set_value(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video.show_message("Starting camera…", "Loading the pose model. This takes a few seconds.")
        self.worker = CameraThread(self)
        self.worker.frame_ready.connect(self.show_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.show_error, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.camera_finished, Qt.ConnectionType.QueuedConnection)
        self.set_state("starting")
        self.location.start()
        self.alarm_timer.start()
        self.worker.start()

    @Slot()
    def stop_camera(self):
        if self.worker is None or self._stopping:
            return
        self._stopping = True
        self.stop_alarm()  # Someone is at the device; stopping the camera also stops the alarm.
        self.stop_button.setEnabled(False)
        self.worker.requestInterruption()
        self.set_state("stopping")
        self.stop_timer.start()

    @Slot(QImage, bool, int, int)
    def show_frame(self, image, is_fallen, evidence, people):
        worker = self.sender()
        try:
            if worker is not self.worker or self._stopping:
                return
            paint_watermark(image, self.location.watermark_lines())  # image is our detached copy
            self.video.set_frame(QPixmap.fromImage(image))
            self._frame_times.append(time.monotonic())
            self._people = people
            self.meter.set_value(evidence)
            if self.fall_log.update(is_fallen, self.location.current_fix()):
                self.log_changed()
            self.check_alarm()
            # An ongoing fall event (including its 2 s grace) keeps the fall state,
            # so the banner does not flicker on single missed frames.
            if self.alarm.ringing:
                state = "alarm"
            elif is_fallen or self.fall_log.current is not None:
                state = "fall"
            else:
                state = "monitoring"
            if state != self._state:
                self.set_state(state)
            elif state == "monitoring":
                self.refresh_live_text()  # people count / stats
        finally:
            if worker is not None:
                worker.frame_consumed()

    @Slot(str)
    def show_error(self, message):
        self._error = message
        self.stop_alarm()
        self.set_state("error", message)

    @Slot()
    def camera_finished(self):
        self.stop_timer.stop()
        self.alarm_timer.stop()
        self.stop_alarm()
        # A fall still in progress ends here; record why.
        reason = f"Error: {self._error}" if self._error else ("App closed" if self._closing else "Camera stopped")
        if self.fall_log.close_event(reason):
            self.log_changed()
        self.location.stop()
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._stopping = False
        self._people = 0
        self._frame_times.clear()
        self.meter.set_value(0)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._error:
            self.video.show_message("Camera unavailable", self._error)
            self.set_state("error", self._error)
        else:
            self.video.show_message("Camera is off", "Press Start monitoring to begin.")
            self.set_state("idle")
        if self._closing:
            self.close()

    @Slot()
    def stop_timeout(self):
        if self.worker is not None:
            self.set_state("stopping", "The camera driver has not released yet. This window stays responsive.")

    # -- alarm ---------------------------------------------------------------
    @Slot()
    def check_alarm(self):
        """Ring once per fall when it has lasted the chosen delay; stop when it ends."""
        current = self.fall_log.current
        if current is None:
            if self.alarm.ringing:
                self.stop_alarm()  # The fall ended (posture cleared or camera stopped).
                if self._state == "alarm":
                    self.set_state("monitoring")
            return
        delay = self.alarm_delay.value()
        if (delay > 0 and not self.alarm.ringing and current["alarm_rang"] is None
                and self.fall_log.ongoing_seconds() >= delay):
            self.fall_log.mark_alarm(delay)
            self.alarm.start()
            self.set_state("alarm")
            self.silence_button.setFocus()  # Space/Enter silences it
            self.raise_()
            self.activateWindow()
            self.log_changed()

    @Slot()
    def silence_alarm(self):
        if self.alarm.ringing:
            self.fall_log.mark_silenced()
            self.log_changed()
        self.stop_alarm()
        if self._state == "alarm":
            self.set_state("fall" if self.fall_log.current is not None else "monitoring")

    def stop_alarm(self):
        self.alarm.stop()
        self.silence_button.setVisible(False)

    @Slot(int)
    def alarm_delay_changed(self, value, save=True):
        if save:
            self.settings.setValue("alarm_after_s", value)
        self.alarm_hint.setText(
            "Falls are still logged, but no siren will ring." if value == 0 else
            f"A siren sounds on this device if a fall lasts {format_duration(value)}. "
            "It stops when the person gets up or someone silences it.")
        if value == 0 and self.alarm.ringing:
            self.silence_alarm()  # Turning the alarm off while it rings silences it.
        if self._state is not None:
            self.apply_state()

    # -- fall log ------------------------------------------------------------
    def update_log_button(self):
        rows = self.fall_log.rows()
        today = datetime.now().strftime("%Y-%m-%d")
        self._falls_today = sum(1 for row in rows if row.get("fall_detected_at", "").startswith(today))
        self.log_button.setText(f"Fall log  ·  {len(rows)}" if rows else "Fall log")

    def log_changed(self):
        self.update_log_button()
        if self.log_dialog is not None and self.log_dialog.isVisible():
            self.log_dialog.refresh()

    @Slot()
    def open_log(self):
        if self.log_dialog is None:
            self.log_dialog = FallLogDialog(self.fall_log, self)  # non-modal: video keeps running
        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None:
            self._closing = True
            self.stop_camera()
            event.ignore()  # Never destroy a running QThread or block the GUI.
        else:
            event.accept()

