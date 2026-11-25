"""MainWindow class."""
from .common import *  # shared Qt imports, constants, and helpers
from .alarmsound import AlarmSound
from .camerathread import CameraThread
from .caregivers import (
    CaregiverDialog, DiscordNotifier, caregiver_config_problem, caregiver_embed,
)
from .evidencemeter import EvidenceMeter
from .fallanalysis import (HISTORY_SECONDS, PATTERN_TEXT, PRIORITY_STYLE, TYPE_SETTLE_S, fall_type_text,
                           pattern_detail)
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
        self._leader_id = 0  # person closest to a confirmed fall (0 = none)
        self._aftermath = {}  # {person ID: post-fall report} from the camera thread
        self._start_pending = None  # monotonic time a fall started; its first Discord alert waits
                                    # (up to TYPE_SETTLE_S) until sudden/gradual is measured
        self._frame_times = deque(maxlen=30)
        self._detect_fps = None  # frames the detector processes per second (camera thread)
        self._falls_today = 0
        self.location = LocationFeed(self)
        self.fall_log = FallLog(device_name())
        self.log_dialog = None
        self.alarm = AlarmSound(self)
        self.settings = QSettings("Vertebrate", "Vertebrate")  # per-user, per-device preferences
        self.notifier = DiscordNotifier(self.settings, self)  # caregiver alerts (off until set up)
        self.notifier.result.connect(self.notification_result)
        self._last_notice = None  # (time text, delivered or None while sending, detail) of the latest alert
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
        footer = make_label("Video stays on this device  ·  Caregiver alerts are optional  ·  Not connected to emergency services",
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
        self.refresh_caregiver_card()
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
        row.addWidget(make_label(f"{CONSECUTIVE_FRAMES} in a row, same person", 12, obj="muted"))
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

        card, layout = self._card("Caregiver alerts")
        self.care_status = make_label("", 15, QFont.Weight.DemiBold)
        self.care_detail = make_label("", 12, obj="muted", wrap=True)
        self.care_button = QPushButton("Set up Discord…")
        self.care_button.setObjectName("ghost")
        self.care_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.care_button.clicked.connect(self.open_caregivers)
        layout.addWidget(self.care_status)
        layout.addWidget(self.care_detail)
        layout.addWidget(self.care_button)
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

    def _who(self) -> str:
        current = self.fall_log.current
        return people_text(current["people"]) if current is not None else "Person"

    def _fall_detail(self) -> str:
        current = self.fall_log.current
        fallen_for = format_duration(self.fall_log.ongoing_seconds())
        delay = self.alarm_delay.value()
        if current is None:
            return "Fall posture detected."
        who = self._who()
        movement = self._movement_text()
        movement = f" {movement}" if movement else ""
        if delay == 0:
            return f"{who} down for {fallen_for}.{movement} The alarm is off."
        if current["alarm_rang"] is None:
            return f"{who} down for {fallen_for}.{movement} Alarm in {self._alarm_remaining()} s."
        return f"{who} down for {fallen_for}.{movement} Alarm silenced."

    def _movement_text(self) -> str:
        """'#2 not moving for 18 s.' / '#2 moving.' for the people in the ongoing fall."""
        current = self.fall_log.current
        if current is None:
            return ""
        parts = []
        for pid in sorted(current["people"]):
            report = self._aftermath.get(pid)
            if report is None:
                continue
            if report["status"] == "still":
                parts.append(f"#{pid} not moving for {format_duration(report['still_s'])}")
            elif report["status"] == "moving":
                parts.append(f"#{pid} moving")
            elif report["status"] == "recovered":
                parts.append(f"#{pid} got up")
        text = (", ".join(parts) + ".") if parts else ""
        help_text = self._help_text()
        return f"{text} {help_text}".strip()

    def _help_text(self) -> str:
        """Who is around the people in the ongoing fall: '#1 is with them.',
        '#3 is coming over.', '2 people nearby (#1, #3).', 'Nobody else nearby.'"""
        current = self.fall_log.current
        if current is None:
            return ""
        reports = [self._aftermath[pid] for pid in sorted(current["people"]) if pid in self._aftermath]
        if not reports:
            return ""
        def ids(key):
            found = sorted({i for r in reports for i in r.get(key, ()) if i not in current["people"]})
            return found, ", ".join(f"#{i}" for i in found)
        with_them, with_text = ids("with_them")
        coming, coming_text = ids("approaching")
        nearby, nearby_text = ids("nearby")
        if with_them:
            return f"{with_text} {'is' if len(with_them) == 1 else 'are'} with them."
        if coming:
            return f"{coming_text} {'is' if len(coming) == 1 else 'are'} coming over."
        if nearby:
            if current["no_response"]:
                return f"{nearby_text} nearby but nobody has gone to them."
            return f"{len(nearby)} {'person' if len(nearby) == 1 else 'people'} nearby ({nearby_text})."
        return "Nobody else nearby."

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
                      f"{self._who()} down for {format_duration(self.fall_log.ongoing_seconds())}. "
                      "Siren ringing on this device."),
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

        still = self.fall_log.still_people()
        if state == "fall":
            priority = self.fall_log.priority_now()
            tone = PRIORITY_STYLE[priority][3]
            word = PRIORITY_STYLE[priority][1].capitalize()
            what = self._what()
            if still:
                self.banner_title.setText(f"{word} · No movement — {people_text(still).lower()} lying still")
                self.banner_detail.setText(self._fall_detail() + " Check on them now.")
            else:
                self.banner_title.setText(f"{word} · {what} — {self._who().lower()}")
                self.banner_detail.setText(self._fall_detail())
            self.banner.setStyleSheet(
                f"QFrame#banner {{ background: {rgba(tone, .16)}; border: 1px solid {rgba(tone, .62)};"
                f" border-radius: 14px; }} QLabel {{ color: {TEXT}; }}")
            self.banner_icon.setStyleSheet(
                f"color: {'#FFFFFF' if priority == 'critical' else '#1A0E06'}; background: {tone};"
                f" border-radius: 18px;")
        elif state == "alarm":
            self.banner_title.setText(f"EMERGENCY ALARM — {self._who().lower()} down for "
                                      f"{format_duration(self.fall_log.ongoing_seconds())}")
            movement = self._movement_text()
            self.banner_detail.setText((f"{movement} " if movement else "")
                                       + "A siren is ringing on this device. Check on the person, then silence it.")
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
        if self._detect_fps is not None:
            fps = self._detect_fps  # the rate fall analysis actually gets
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
        if self._leader_id and self.meter.value:
            self.evidence_label.setText(f"#{self._leader_id} · {self.meter.value} of {CONSECUTIVE_FRAMES}")
        else:
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
        self._detect_fps = None
        self._people = 0
        self._leader_id = 0
        self._start_pending = None
        self._aftermath = {}
        self.meter.set_value(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.video.show_message("Starting camera…", "Loading the pose model. This takes a few seconds.")
        self.worker = CameraThread(self)
        self.worker.frame_ready.connect(self.show_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.stats_ready.connect(self.show_stats, Qt.ConnectionType.QueuedConnection)
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

    @Slot(QImage, bool, int, int, int, list, dict)
    def show_frame(self, image, is_fallen, evidence, leader_id, people, confirmed_ids, aftermath):
        worker = self.sender()
        try:
            if worker is not self.worker or self._stopping:
                return
            paint_watermark(image, self.location.watermark_lines())  # image is our detached copy
            self.video.set_frame(QPixmap.fromImage(image))
            self._frame_times.append(time.monotonic())
            self._people = people
            self._leader_id = leader_id
            self._aftermath = aftermath
            self.meter.set_value(evidence)
            was_falling = self.fall_log.current is not None
            if self.fall_log.update(is_fallen, self.location.current_fix(), confirmed_ids):
                self.log_changed()
                if not was_falling and self.fall_log.current is not None:
                    self._start_pending = time.monotonic()
                elif was_falling and self.fall_log.current is None:
                    self.notify_fall_ended()
            changes = self.fall_log.note_aftermath(aftermath)
            if self._start_pending is not None and (
                    self.fall_log.types_final() or time.monotonic() - self._start_pending >= TYPE_SETTLE_S):
                self._flush_start()
            for change in changes:
                if change == "critical":
                    self.notify_no_movement()
                    self.log_changed()
                elif change == "moving_again":
                    self.notify_moving_again()
                elif change == "helper":
                    self.notify_helper()
                    self.log_changed()
                elif change == "no_response":
                    self.notify_no_response()
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

    @Slot(float, float, int)
    def show_stats(self, fps, inference_ms, history_frames):
        if self.sender() is not self.worker or self._stopping:
            return
        self._detect_fps = fps
        self.stat_fps.setToolTip(f"Detection rate {fps:.1f} frames/s · model {inference_ms:.0f} ms per frame\n"
                                 f"A {HISTORY_SECONDS:.0f}-second pose history holds {history_frames} frames per person.")

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
            self.notify_fall_ended()
        self.location.stop()
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._stopping = False
        self._people = 0
        self._leader_id = 0
        self._start_pending = None
        self._aftermath = {}
        self._frame_times.clear()
        self._detect_fps = None
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
            self.notify_alarm()
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

    # -- caregiver alerts ----------------------------------------------------
    def _send_caregiver(self, kind: str, title: str, description: str, location, mention: bool,
                        extra_fields=None, priority: str | None = None) -> None:
        if kind != "test":
            self._flush_start()  # the first fall alert always goes out first
        color = PRIORITY_STYLE[priority][2] if priority in PRIORITY_STYLE else None
        embed = caregiver_embed(kind, self.fall_log.device, title, description, location, extra_fields, color)
        if self.notifier.send(kind, embed, mention):
            self._last_notice = (datetime.now().strftime("%H:%M:%S"), None, "Sending…", "alert")
            self.refresh_caregiver_card()

    def _flush_start(self) -> None:
        """Send the first alert of the ongoing fall if it is still waiting."""
        if self._start_pending is None:
            return
        self._start_pending = None
        if self.fall_log.current is not None:
            self.notify_fall_started()

    def _type_text(self) -> str:
        """'Sudden fall (down in 0.6 s)' for the first person in the ongoing fall."""
        current = self.fall_log.current
        if current is None:
            return ""
        for pid in sorted(current["people"]):
            if pid in self._aftermath:
                return fall_type_text(self._aftermath[pid])
        return ""

    def _first_report(self) -> dict:
        current = self.fall_log.current
        if current is None:
            return {}
        return next((self._aftermath[pid] for pid in sorted(current["people"]) if pid in self._aftermath), {})

    def _what(self) -> str:
        """Headline for the ongoing fall: 'Collapse', 'Possible slip', 'Sudden fall', 'Fall detected'."""
        report = self._first_report()
        if report.get("pattern") in PATTERN_TEXT:
            return PATTERN_TEXT[report["pattern"]][0]
        if report.get("fall_type") in ("sudden", "gradual"):
            return f"{report['fall_type'].capitalize()} fall"
        return "Fall detected"

    def _pattern_fields(self, note: bool = True) -> list:
        """'What it may mean' and 'Signs' fields when a pattern was recognised."""
        report = self._first_report()
        if report.get("pattern") not in PATTERN_TEXT:
            return []
        fields = [{"name": "What it may mean", "value": PATTERN_TEXT[report["pattern"]][1].capitalize(),
                   "inline": False}]
        signs = pattern_detail(report)
        if signs:
            fields.append({"name": "Signs", "value": signs, "inline": False})
        if note:
            fields.append({"name": "Note", "value": "Pattern from camera analysis — not a medical diagnosis.",
                           "inline": False})
        return fields

    def _priority_title(self, priority: str, what: str) -> str:
        emoji, word = PRIORITY_STYLE[priority][:2]
        return f"{emoji} {word} — {what} — {self._who()}"

    def notify_fall_started(self) -> None:
        current = self.fall_log.current
        delay = self.alarm_delay.value()
        when = current["start"].strftime("%H:%M:%S on %d %b %Y")
        follow = (f"The alarm will ring in {format_duration(delay)} if they stay down." if delay
                  else "The alarm on the device is off.")
        help_text = self._help_text()
        priority = self.fall_log.priority_now()
        kind = self._type_text()
        what = self._what()
        fields = [{"name": "Priority", "value": PRIORITY_STYLE[priority][1].capitalize(), "inline": True},
                  {"name": "How they fell", "value": kind or "Fall type unclear", "inline": True}]
        if help_text:
            fields.append({"name": "People around", "value": help_text, "inline": True})
        fields += self._pattern_fields()
        if self._first_report().get("pattern") == "collapse":
            follow = "They did not try to break the fall — please check on them immediately. " + follow
        self._send_caregiver(
            "fall", self._priority_title(priority, what),
            f"A fall was detected on **{self.fall_log.device}** at {when}. " + follow,
            current["location"], mention=True, extra_fields=fields, priority=priority)

    def notify_alarm(self) -> None:
        current = self.fall_log.current
        if current is None:
            return
        self._send_caregiver(
            "alarm", f"🚨 EMERGENCY — {self._who()} still down",
            f"Down for **{format_duration(self.fall_log.ongoing_seconds())}** on **{self.fall_log.device}**. "
            + (f"{self._help_text()} " if self._help_text() else "")
            + "The siren is ringing. Please check on them now.",
            current["location"], mention=True)

    def notify_no_movement(self) -> None:
        current = self.fall_log.current
        still = self.fall_log.still_people()
        if current is None or not still:
            return
        report = self._aftermath.get(still[0], {})
        self._send_caregiver(
            "critical", f"🔴 CRITICAL — {people_text(still)} not moving",
            f"{people_text(still)} fell and has **not moved for {format_duration(report.get('still_s', 0))}** "
            f"on **{self.fall_log.device}** (down for {format_duration(self.fall_log.ongoing_seconds())}). "
            "They may be unconscious. " + (f"**{self._help_text()}** " if self._help_text() else "")
            + "Please check on them immediately or call emergency services.",
            current["location"], mention=True,
            extra_fields=[{"name": "How they fell", "value": self._type_text() or "Fall type unclear",
                           "inline": True}] + self._pattern_fields(note=False) + [
                          {"name": "Note", "value": "Movement judged from camera analysis — not a medical "
                                                     "diagnosis.", "inline": False}],
            priority="critical")

    def notify_moving_again(self) -> None:
        current = self.fall_log.current
        if current is None:
            return
        self._send_caregiver(
            "fall", f"🟠 Update — {self._who()} moving again",
            f"After lying still, {self._who().lower()} is moving again on **{self.fall_log.device}**, "
            f"but is still down ({format_duration(self.fall_log.ongoing_seconds())}). Please still check on them.",
            current["location"], mention=False, priority=self.fall_log.priority_now())

    def notify_helper(self) -> None:
        current = self.fall_log.current
        if current is None or not current["helpers"]:
            return
        helper = f"#{current['helpers'][0]}"
        after = current["help_after"]
        self._send_caregiver(
            "fall", f"🟡 Someone is with them — {self._who()}",
            f"Person {helper} reached {self._who().lower()} on **{self.fall_log.device}**"
            + (f" {format_duration(after)} after the fall" if after is not None else "")
            + ". The alert stays on until they are back on their feet.",
            current["location"], mention=False, priority=self.fall_log.priority_now())

    def notify_no_response(self) -> None:
        current = self.fall_log.current
        if current is None:
            return
        nearby = ", ".join(f"#{i}" for i in sorted(current["nearby"]))
        self._send_caregiver(
            "fall", f"🟠 Nobody has responded — {self._who()}",
            f"{nearby or 'People'} {'has' if len(current['nearby']) == 1 else 'have'} been near "
            f"{self._who().lower()} on **{self.fall_log.device}** but nobody has gone to them "
            f"(down for {format_duration(self.fall_log.ongoing_seconds())}). They may not have noticed — "
            "please check on them.",
            current["location"], mention=True, priority=self.fall_log.priority_now())

    def notify_fall_ended(self) -> None:
        row = self.fall_log.last_finished
        if row is None:
            return
        extra = [{"name": "Ended by", "value": row["ended_by"], "inline": True},
                 {"name": "Duration", "value": row["duration"], "inline": True}]
        if row.get("alarm_rang_at"):
            alarm = f"Rang {row['alarm_rang_at'][-8:]}"
            if row.get("alarm_silenced_at"):
                alarm += f", silenced {row['alarm_silenced_at'][-8:]}"
            extra.append({"name": "Alarm", "value": alarm, "inline": True})
        if row.get("fall_type"):
            extra.append({"name": "How they fell", "value": FallLog.type_text(row), "inline": True})
        if row.get("priority"):
            extra.append({"name": "Highest priority", "value": row["priority"].capitalize(), "inline": True})
        if row.get("after_fall"):
            extra.append({"name": "After the fall", "value": row["after_fall"], "inline": True})
            extra.append({"name": "Help", "value": FallLog.help_text(row), "inline": True})
        if row["ended_by"] == "Got up":
            self._send_caregiver(
                "end", f"🟢 Recovered — {people_text(row.get('people', []))} got up",
                f"{people_text(row.get('people', []))} got up again on **{self.fall_log.device}** after being "
                f"down for {row['duration']} ({row['fall_detected_at'][-8:]}–{row['ended_at'][-8:]}). "
                "No further alert for this fall.",
                row["location"], mention=False, extra_fields=extra, priority="recovered")
            return
        self._send_caregiver(
            "end", f"✅ Fall ended — {people_text(row.get('people', []))}",
            f"{people_text(row.get('people', []))} was down for {row['duration']} on **{self.fall_log.device}** "
            f"({row['fall_detected_at'][-8:]}–{row['ended_at'][-8:]}).",
            row["location"], mention=False, extra_fields=extra)

    @Slot(str, bool, str)
    def notification_result(self, kind, delivered, detail):
        if kind == "test" and not self.notifier.config.get("fixed"):
            return  # the setup window shows its own result
        self._last_notice = (datetime.now().strftime("%H:%M:%S"), delivered, detail,
                             "test" if kind == "test" else "alert")
        self.refresh_caregiver_card()

    def refresh_caregiver_card(self) -> None:
        config = self.notifier.config
        if not self.notifier.active:
            self.care_status.setText("Not set up" if not config["enabled"] else "Needs attention")
            self.care_status.setStyleSheet(f"color: {FAINT if not config['enabled'] else WARN};")
            self.care_detail.setText("Notify caregivers in a Discord channel when someone falls."
                                     if not config["enabled"] else caregiver_config_problem(config) or "")
            self.care_detail.setStyleSheet("")
            self.care_button.setText("Set up Discord…")
            return
        events = [name for key, name in (("on_fall", "falls"), ("on_alarm", "alarms"), ("on_end", "recoveries"))
                  if config[key]]
        fixed = config.get("fixed", False)
        via = "fixed channel" if fixed else ("bot" if config["mode"] == "bot" else "webhook")
        self.care_status.setText(f"Discord · on ({via})")
        self.care_status.setStyleSheet(f"color: {OK};")
        detail = "Sends: " + (", ".join(events) if events else "nothing (all events switched off)")
        failed = False
        if self._last_notice is not None:
            at, delivered, text, what = self._last_notice
            if delivered is None:
                detail += f"\nLast {what} {at}: sending…"
            elif delivered:
                detail += f"\nLast {what} {at}: delivered"
            else:
                detail += f"\nLast {what} {at} failed: {text}"
                failed = True
        self.care_detail.setText(detail)
        self.care_detail.setStyleSheet(f"color: {DANGER};" if failed else "")
        self.care_button.setText("Send test message" if fixed else "Discord settings…")

    @Slot()
    def open_caregivers(self):
        if self.notifier.config.get("fixed"):  # fixed channel: the button just sends a test
            embed = caregiver_embed("test", self.fall_log.device, "✅ Vertebrate test message",
                                    f"Caregiver alerts from **{self.fall_log.device}** are working. "
                                    "Real alerts will look like this, with times and location.")
            if self.notifier.send("test", embed, mention=False, config=self.notifier.config):
                self._last_notice = (datetime.now().strftime("%H:%M:%S"), None, "Sending…", "test")
                self.refresh_caregiver_card()
            return
        dialog = CaregiverDialog(self.notifier, self.fall_log.device, self)
        dialog.exec()  # modal: monitoring keeps running underneath
        self.refresh_caregiver_card()

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None:
            self._closing = True
            self.stop_camera()
            event.ignore()  # Never destroy a running QThread or block the GUI.
        else:
            self.notifier.flush(3.0)  # let a last "fall ended" alert go out (bounded wait)
            event.accept()

