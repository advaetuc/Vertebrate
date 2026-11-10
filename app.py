r"""Standalone webcam pose demo for Windows and macOS.

Run:  Windows  .venv\Scripts\python.exe app.py
      macOS    .venv/bin/python app.py

No VERTEBRATE imports, tracker, baseline, or configuration schema. Geometry is
per-frame; the only detection history is a five-frame, scene-wide debounce.
Uses existing local YOLO weights; does not download weights or upload footage.
The live view carries a location/time watermark: a serial GPS receiver is used
when one is found, otherwise the operating system's location service (its own
GPS chip if the device has one, else Wi-Fi/network positioning).
Confirmed fall alerts are logged per device to runtime/logs/falls/<device>.csv
(times, duration, location; never images) and listed by the Fall log button.
"""
from datetime import datetime
from pathlib import Path
import csv
import os
import platform
import re
import sys
import threading
import wave

# Set before importing Ultralytics. The installed version expects "true".
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / "runtime" / "logs" / "standalone-yolo")

import cv2
import numpy as np
from PySide6.QtCore import (
    QCoreApplication, QDateTime, QLocationPermission, QObject, QRectF, QSettings, QSysInfo, Qt,
    QThread, QTimer, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QCloseEvent, QColor, QDesktopServices, QFont, QFontDatabase, QImage, QPainter, QPixmap,
)
from PySide6.QtPositioning import QGeoPositionInfo, QGeoPositionInfoSource
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
try:  # Qt Multimedia plays the alarm; without it the app falls back to system beeps.
    from PySide6.QtMultimedia import QSoundEffect
except ImportError:
    QSoundEffect = None

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

CONSECUTIVE_FRAMES = 5
GREEN, RED = (80, 220, 100), (60, 60, 255)  # OpenCV BGR
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)

# Location ------------------------------------------------------------------
LOCATION_INTERVAL_MS = 2000  # Requested update interval; the OS may deliver less often.
STALE_AFTER_S = 60           # Older fixes are labelled with their age.
GPS_FRESH_S = 10             # A GPS fix younger than this beats the system fix.
# Override GPS detection: VERTEBRATE_GPS_PORT=COM4 (Windows), =/dev/cu.usbmodem1101
# (macOS), or =off to never use a serial receiver. VERTEBRATE_GPS_BAUD defaults to 9600.
GPS_BAUD = int(os.environ.get("VERTEBRATE_GPS_BAUD", "9600"))
GPS_VENDOR_IDS = {0x1546, 0x091E}  # u-blox, Garmin
GPS_WORDS = ("gps", "gnss", "u-blox", "ublox", "nmea", "sirf", "garmin")
# Qt Positioning plugin that wraps each OS location service. Both fuse a built-in
# GPS chip (when the hardware has one) with Wi-Fi/network positioning.
SYSTEM_SOURCE = "winrt" if IS_WINDOWS else "corelocation" if IS_MACOS else ""

# Fall log ------------------------------------------------------------------
FALL_LOG_DIR = Path(__file__).resolve().parent / "runtime" / "logs" / "falls"  # Git-ignored
# A fall event ends only after the alert has been off this long, so a one-frame
# flicker in the debounce does not split one fall into several log rows.
END_GRACE_S = 2.0
LOG_FIELDS = (
    "device", "fall_detected_at", "ended_at", "utc_offset", "duration", "duration_seconds",
    "latitude", "longitude", "accuracy_m", "location_source", "ended_by",
    "alarm_after_seconds", "alarm_rang_at", "alarm_silenced_at",
)

# Emergency alarm -----------------------------------------------------------
# Rings on this device when a fall lasts longer than the user's chosen delay,
# counted from when the fall alert started. 0 = alarm off.
ALARM_DEFAULT_S = 30
ALARM_MAX_S = 600
ALARM_SOUND = Path(__file__).resolve().parent / "runtime" / "sounds" / "alarm.wav"  # generated

if IS_WINDOWS:
    CAMERA_BACKENDS =(cv2.CAP_DSHOW, cv2.CAP_MSMF)  # DirectShow opens faster; MSMF as fallback.
    CAMERA_HELP = ("Cannot open camera 0. Close other camera apps and allow desktop apps in "
                   "Settings > Privacy & security > Camera.")
elif IS_MACOS:
    CAMERA_BACKENDS = (cv2.CAP_AVFOUNDATION,)
    CAMERA_HELP = ("Cannot open camera 0. Allow your terminal or editor in System Settings > "
                   "Privacy & Security > Camera. If macOS just asked for access, press Start again.")
else:
    CAMERA_BACKENDS = (cv2.CAP_ANY,)
    CAMERA_HELP = "Cannot open camera 0. Close other camera apps and check camera permissions."


def fallen_geometry(box, keypoints):
    """Exact requested OR rule; missing (zeroed) joints cannot define a spine."""
    x1, y1, x2, y2 = map(float, box)
    height = y2 - y1
    wide = height > 0 and (x2 - x1) / height > 1.2
    inverted = False
    if keypoints is not None and len(keypoints) >= 13:
        core = np.asarray(keypoints)[[5, 6, 11, 12]]
        if np.isfinite(core).all() and np.all(core[:, 1] > 0):
            shoulder_y = (float(core[0, 1]) + float(core[1, 1])) / 2
            hip_y = (float(core[2, 1]) + float(core[3, 1])) / 2
            inverted = shoulder_y > hip_y + 20.0
    return bool(wide or inverted)


class Debounce:
    def __init__(self):
        self.count = 0

    def update(self, any_fallen):
        self.count = min(self.count + 1, CONSECUTIVE_FRAMES) if any_fallen else 0
        return self.count >= CONSECUTIVE_FRAMES


def annotate(frame, result):
    """Paint this result onto its own frame; return whether any person matches."""
    if result.boxes is None:
        return False
    boxes = result.boxes.xyxy.cpu().numpy()
    points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    any_fallen = False
    for i, box in enumerate(boxes):
        joints = points[i] if points is not None and i < len(points) else None
        fallen = fallen_geometry(box, joints)
        any_fallen |= fallen
        color = RED if fallen else GREEN
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "Fallen posture" if fallen else "Standing", (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
        if joints is None:
            continue
        visible = np.isfinite(joints).all(axis=1) & (joints[:, 0] > 0) & (joints[:, 1] > 0)
        for a, b in SKELETON:
            if max(a, b) < len(joints) and visible[a] and visible[b]:
                cv2.line(frame, tuple(map(int, joints[a])), tuple(map(int, joints[b])), color, 2, cv2.LINE_AA)
        for point in joints[visible]:
            cv2.circle(frame, tuple(map(int, point)), 3, color, -1, cv2.LINE_AA)
    return bool(any_fallen)


def local_weights():
    root = Path(__file__).resolve().parent
    for path in (root / "yolo11n-pose.pt", root / "models" / "yolo11n-pose.pt"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError("Place yolo11n-pose.pt beside app.py or in models/. No automatic download is attempted.")


def open_camera(index: int = 0):
    """Open the webcam with this platform's preferred capture backend(s).
    Called only from the worker thread, which also releases the capture."""
    for backend in CAMERA_BACKENDS:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    raise RuntimeError(CAMERA_HELP)


class CameraThread(QThread):
    frame_ready = Signal(QImage, bool)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_pending = threading.Event()

    def frame_consumed(self):
        self._frame_pending.clear()

    def run(self):
        capture = None
        try:
            weights = local_weights()  # Fail locally BEFORE constructing YOLO.
            Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS
            # Disable telemetry in this process without changing saved settings.
            dict.__setitem__(SETTINGS, "sync", False)
            dict.__setitem__(SETTINGS, "hub", False)
            model = YOLO(str(weights), task="pose")
            if self.isInterruptionRequested():
                return
            capture = open_camera(0)
            debounce = Debounce()
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("The camera stopped delivering frames. Check its connection and restart.")
                if self.isInterruptionRequested():
                    break
                result = model.predict(frame, device="cpu", verbose=False)[0]
                is_fallen = debounce.update(annotate(frame, result))
                cv2.putText(frame, f"Fall evidence: {debounce.count}/{CONSECUTIVE_FRAMES}", (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, RED if is_fallen else GREEN, 2, cv2.LINE_AA)
                # At most ONE queued frame signal: a slow GUI cannot accumulate
                # large image buffers. Inference/debounce still see every frame.
                if not self._frame_pending.is_set() and not self.isInterruptionRequested():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    self._frame_pending.set()
                    self.frame_ready.emit(image, is_fallen)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()  # Camera is released by its owning thread.


def find_gps_port() -> str | None:
    """Name of a serial/USB/Bluetooth NMEA GPS receiver, or None.
    VERTEBRATE_GPS_PORT overrides detection; only likely GPS devices are opened."""
    override = os.environ.get("VERTEBRATE_GPS_PORT", "").strip()
    if override.lower() in ("off", "none", "0"):
        return None
    if override:
        return override
    try:
        from PySide6.QtSerialPort import QSerialPortInfo
    except ImportError:
        return None
    for port in QSerialPortInfo.availablePorts():
        text = f"{port.description()} {port.manufacturer()}".lower()
        vendor = port.vendorIdentifier() if port.hasVendorIdentifier() else None
        if vendor in GPS_VENDOR_IDS or any(word in text for word in GPS_WORDS):
            return port.portName()
    return None


class LocationFeed(QObject):
    """Live device location for the watermark. GUI thread only.

    Two Qt Positioning sources may run side by side:
      "gps"    - NMEA from a serial GPS receiver, if one is detected;
      "system" - the OS location service (Windows Location / macOS CoreLocation),
                 which uses the device's own GPS chip when it has one and
                 otherwise Wi-Fi/network positioning.
    A fresh GPS-receiver fix wins; otherwise the system fix is shown. Only the
    latest fix per source is kept in memory; nothing is saved or sent by the app.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources = {}
        self._fixes = {}     # kind -> (lat, lon, accuracy m or None, taken QDateTime, received QDateTime)
        self._problems = {}  # kind -> readable reason when that source has no usable fix
        self._running = False
        self._permission_asked = False
        self.gps_port = find_gps_port()
        if self.gps_port:
            source = QGeoPositionInfoSource.createSource(
                "nmea", {"nmea.source": f"serial:{self.gps_port}", "nmea.baudrate": GPS_BAUD}, self)
            if source is not None:
                self._add("gps", source)
        if SYSTEM_SOURCE and SYSTEM_SOURCE in QGeoPositionInfoSource.availableSources():
            source = QGeoPositionInfoSource.createSource(SYSTEM_SOURCE, self)
        else:
            source = QGeoPositionInfoSource.createDefaultSource(self)
        if source is not None:
            # All methods: lets the OS use a built-in GPS chip when present.
            source.setPreferredPositioningMethods(QGeoPositionInfoSource.PositioningMethod.AllPositioningMethods)
            self._add("system", source)
        elif "gps" not in self._sources:
            self._problems["system"] = "Location unavailable on this device"

    def _add(self, kind: str, source: QGeoPositionInfoSource) -> None:
        source.setUpdateInterval(LOCATION_INTERVAL_MS)
        source.positionUpdated.connect(lambda info, k=kind: self._on_position(k, info))
        source.errorOccurred.connect(lambda error, k=kind: self._on_error(k, error))
        self._sources[kind] = source
        self._problems[kind] = "Locating…"

    def start(self) -> None:
        self._running = True
        if "gps" in self._sources:
            self._sources["gps"].startUpdates()
        if "system" in self._sources:
            self._start_system_with_permission()

    def stop(self) -> None:
        self._running = False
        for source in self._sources.values():
            source.stopUpdates()

    def _start_system_with_permission(self) -> None:
        """Ask the OS for location access once, then start the system source.
        The source is started whatever the answer: if access is really blocked,
        it reports AccessError and the watermark shows how to allow it."""
        app = QCoreApplication.instance()
        permission = QLocationPermission()
        permission.setAccuracy(QLocationPermission.Accuracy.Precise)
        permission.setAvailability(QLocationPermission.Availability.WhenInUse)
        try:
            status = app.checkPermission(permission)
        except Exception:
            status = None
        if status == Qt.PermissionStatus.Undetermined and not self._permission_asked:
            self._permission_asked = True
            try:
                app.requestPermission(permission, self, lambda _result: self._start_system())
                return
            except Exception:
                pass
        self._start_system()

    def _start_system(self) -> None:
        if self._running and "system" in self._sources:
            self._sources["system"].startUpdates()

    def _on_position(self, kind: str, info: QGeoPositionInfo) -> None:
        coordinate = info.coordinate()
        if not info.isValid() or not coordinate.isValid():
            return
        accuracy = None
        attribute = QGeoPositionInfo.Attribute.HorizontalAccuracy
        if info.hasAttribute(attribute):
            accuracy = info.attribute(attribute)
        received = QDateTime.currentDateTime()
        taken = info.timestamp() if info.timestamp().isValid() else received
        self._fixes[kind] = (coordinate.latitude(), coordinate.longitude(), accuracy, taken, received)
        self._problems.pop(kind, None)

    def _on_error(self, kind: str, error) -> None:
        errors = QGeoPositionInfoSource.Error
        if kind == "gps":
            if error == errors.UpdateTimeoutError:
                self._problems[kind] = "GPS receiver: no satellite fix yet"
            else:
                self._problems[kind] = f"GPS receiver on {self.gps_port} could not be read"
            return
        if error == errors.AccessError:
            where = ("Settings > Privacy & security > Location" if IS_WINDOWS
                     else "System Settings > Privacy & Security > Location Services")
            self._problems[kind] = f"Location blocked: allow it in {where}"
        elif error == errors.ClosedError:
            self._problems[kind] = "Location services are turned off"
        elif error == errors.UpdateTimeoutError:
            if kind not in self._fixes:
                self._problems[kind] = "Waiting for a location fix…"
        else:
            self._problems[kind] = "Location unavailable"

    def _best(self, now: QDateTime):
        gps = self._fixes.get("gps")
        system = self._fixes.get("system")
        if gps is not None and gps[4].secsTo(now) <= GPS_FRESH_S:
            return "GPS receiver", gps
        if system is not None:
            label = "system location"
            if "gps" in self._sources:
                label += " (GPS receiver has no fix)"
            return label, system
        if gps is not None:
            return "GPS receiver", gps
        return None, None

    def current_fix(self):
        """(latitude, longitude, accuracy m or None, source label) or None."""
        label, fix = self._best(QDateTime.currentDateTime())
        if fix is None:
            return None
        lat, lon, accuracy, _taken, _received = fix
        return lat, lon, accuracy, label

    def watermark_lines(self) -> list[str]:
        now = QDateTime.currentDateTime()
        stamp = now.toString("dd MMM yyyy  HH:mm:ss")
        label, fix = self._best(now)
        if fix is None:
            problem = (self._problems.get("system") or self._problems.get("gps") or "Locating…")
            return [problem, stamp]
        lat, lon, accuracy, taken, _received = fix
        text = f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.5f}° {'E' if lon >= 0 else 'W'}"
        if accuracy is not None and accuracy > 0:
            text += f"  ±{accuracy:.0f} m"
        age = taken.secsTo(now)
        if age > STALE_AFTER_S:
            text += f"  (fix {age // 60} min old)"
        return [text, f"{stamp}  ·  via {label}"]


def paint_watermark(image: QImage, lines: list[str]) -> None:
    """Draw a translucent location/time box in the bottom-left of the frame.
    GUI thread only. Sized from the frame width so it scales with the video."""
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = max(11, image.width() // 55)
        font = QFont()
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad = max(6, size // 2)
        line_height = metrics.height()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 2 * pad
        height = line_height * len(lines) + 2 * pad
        x, y = 2 * pad, image.height() - height - 2 * pad
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 130))
        painter.drawRoundedRect(QRectF(x, y, width, height), pad, pad)
        painter.setPen(QColor(255, 255, 255, 220))
        for i, line in enumerate(lines):
            painter.drawText(x + pad, y + pad + metrics.ascent() + i * line_height, line)
    finally:
        painter.end()


def device_name() -> str:
    """Readable name of this computer, used to keep one fall log per device."""
    name = QSysInfo.machineHostName() or platform.node() or "this-device"
    return name.removesuffix(".local")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


class FallLog:
    """Per-device CSV of confirmed fall alerts. GUI thread only.

    An event starts when the debounced alert turns on and ends when the alert
    has stayed off for END_GRACE_S (end time = when it first went off), or when
    the camera stops. One row is appended per finished event. The file lives in
    runtime/logs/falls/<device>.csv and opens directly in Excel or Numbers.
    """

    def __init__(self, device: str):
        self.device = device
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", device).strip("._") or "this-device"
        self.path = FALL_LOG_DIR / f"{safe}.csv"
        self.current = None       # ongoing event: {"start": datetime, "location": tuple | None}
        self._clear_since = None  # when the alert of the ongoing event last went off
        self.last_error = None

    def update(self, is_fallen: bool, location) -> bool:
        """Feed the alert state of each displayed frame. True if the log changed."""
        now = datetime.now().astimezone()
        if is_fallen:
            self._clear_since = None
            if self.current is None:
                self.current = {"start": now, "location": location,
                                "alarm_after": None, "alarm_rang": None, "alarm_silenced": None}
                return True
            if self.current["location"] is None and location is not None:
                self.current["location"] = location  # first fix arrived mid-event
            return False
        if self.current is None:
            return False
        if self._clear_since is None:
            self._clear_since = now
        elif (now - self._clear_since).total_seconds() >= END_GRACE_S:
            self._finish(self._clear_since, "Posture cleared")
            return True
        return False

    def close_event(self, reason: str) -> bool:
        """End an ongoing event (camera stopped, error, app closing)."""
        if self.current is None:
            return False
        self._finish(self._clear_since or datetime.now().astimezone(), reason)
        return True

    def mark_alarm(self, after_seconds: int) -> None:
        if self.current is not None and self.current["alarm_rang"] is None:
            self.current["alarm_after"] = after_seconds
            self.current["alarm_rang"] = datetime.now().astimezone()

    def mark_silenced(self) -> None:
        if self.current is not None and self.current["alarm_rang"] is not None:
            self.current["alarm_silenced"] = datetime.now().astimezone()

    def ongoing_seconds(self) -> float:
        if self.current is None:
            return 0.0
        return (datetime.now().astimezone() - self.current["start"]).total_seconds()

    def _finish(self, end: datetime, reason: str) -> None:
        event = self.current
        start, location = event["start"], event["location"]
        self.current, self._clear_since = None, None
        stamp = lambda moment: moment.strftime("%Y-%m-%d %H:%M:%S") if moment else ""
        seconds = (end - start).total_seconds()
        offset = start.strftime("%z")
        lat, lon, accuracy, source = location if location is not None else ("", "", None, "unavailable")
        row = {
            "device": self.device,
            "fall_detected_at": start.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end.strftime("%Y-%m-%d %H:%M:%S"),
            "utc_offset": f"{offset[:3]}:{offset[3:]}" if offset else "",
            "duration": format_duration(seconds),
            "duration_seconds": f"{seconds:.1f}",
            "latitude": f"{lat:.6f}" if lat != "" else "",
            "longitude": f"{lon:.6f}" if lon != "" else "",
            "accuracy_m": f"{accuracy:.0f}" if accuracy else "",
            "location_source": source,
            "ended_by": reason,
            "alarm_after_seconds": event["alarm_after"] if event["alarm_after"] is not None else "",
            "alarm_rang_at": stamp(event["alarm_rang"]),
            "alarm_silenced_at": stamp(event["alarm_silenced"]),
        }
        try:
            FALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._upgrade_header()
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            # utf-8-sig so Excel shows the ° and ± correctly; newline="" per csv docs.
            with self.path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self.last_error = None
        except OSError as exc:
            self.last_error = f"Could not write the fall log: {exc}"
            print(self.last_error, file=sys.stderr)

    def _upgrade_header(self) -> None:
        """Rewrite a log made by an older version so its columns match LOG_FIELDS."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) == LOG_FIELDS:
                return
            old_rows = list(reader)
        with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(old_rows)

    def rows(self) -> list[dict]:
        """Finished events for this device, newest first."""
        if not self.path.exists():
            return []
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            self.last_error = f"Could not read the fall log: {exc}"
            return []
        return list(reversed(rows))


class FallLogDialog(QDialog):
    """Window listing this device's fall events; an ongoing fall is shown live on top."""
    COLUMNS = ("Fall detected at", "Ended at", "Duration", "Alarm", "Location", "Location source", "Ended by")
    LOCATION_COLUMN = 4

    def __init__(self, log: FallLog, parent=None):
        super().__init__(parent)
        self.log = log
        self.setWindowTitle(f"Fall log · {log.device}")
        self.resize(980, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel(f"Fall log — {log.device}")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        self.summary = QLabel()
        self.summary.setObjectName("subtitle")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.LOCATION_COLUMN, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self.empty = QLabel("No falls recorded on this device yet.")
        self.empty.setObjectName("subtitle")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty)
        buttons = QHBoxLayout()
        folder = QPushButton("Open log file location")
        folder.clicked.connect(self.open_folder)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(folder)
        buttons.addStretch()
        buttons.addWidget(close)
        layout.addLayout(buttons)
        # Keeps the duration of an ongoing fall ticking while the window is open.
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    @Slot()
    def _tick(self):
        if self.log.current is not None:
            self.refresh()

    def refresh(self):
        rows = self.log.rows()
        entries = []
        if self.log.current is not None:
            current = self.log.current
            stamp = lambda moment: moment.strftime("%Y-%m-%d %H:%M:%S") if moment else ""
            entries.append(({
                "fall_detected_at": stamp(current["start"]),
                "ended_at": "ongoing",
                "duration": format_duration(self.log.ongoing_seconds()),
                **self._location_fields(current["location"]),
                "ended_by": "—",
                "alarm_rang_at": stamp(current["alarm_rang"]),
                "alarm_silenced_at": stamp(current["alarm_silenced"]),
            }, True))
        entries += [(row, False) for row in rows]
        self.table.setRowCount(len(entries))
        alert_bg = QColor("#631e2a")
        for r, (row, ongoing) in enumerate(entries):
            location = "—"
            if row.get("latitude") and row.get("longitude"):
                location = f"{float(row['latitude']):.5f}, {float(row['longitude']):.5f}"
                if row.get("accuracy_m"):
                    location += f"  (±{row['accuracy_m']} m)"
            alarm = "—"
            if row.get("alarm_rang_at"):
                alarm = f"Rang {row['alarm_rang_at'][-8:]}"  # time part; date is in the first column
                if row.get("alarm_silenced_at"):
                    alarm += f", silenced {row['alarm_silenced_at'][-8:]}"
            values = (row.get("fall_detected_at", ""), row.get("ended_at", ""), row.get("duration", ""),
                      alarm, location, row.get("location_source", ""), row.get("ended_by", ""))
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if ongoing:
                    item.setBackground(alert_bg)
                self.table.setItem(r, c, item)
        self.table.setVisible(bool(entries))
        self.empty.setVisible(not entries)
        total = sum(float(row.get("duration_seconds") or 0) for row in rows)
        alarms = sum(1 for row in rows if row.get("alarm_rang_at"))
        text = f"{len(rows)} recorded fall{'s' if len(rows) != 1 else ''}"
        if rows:
            text += f" · alarm rang {alarms}× · total time in alert {format_duration(total)}"
        if self.log.current is not None:
            text += " · a fall is being detected now"
        text += f"\nFile: {self.log.path}"
        if self.log.last_error:
            text += f"\n{self.log.last_error}"
        self.summary.setText(text)

    @staticmethod
    def _location_fields(location):
        if location is None:
            return {"latitude": "", "longitude": "", "accuracy_m": "", "location_source": "unavailable"}
        lat, lon, accuracy, source = location
        return {"latitude": f"{lat}", "longitude": f"{lon}",
                "accuracy_m": f"{accuracy:.0f}" if accuracy else "", "location_source": source}

    @Slot()
    def open_folder(self):
        FALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(FALL_LOG_DIR)))


def ensure_alarm_sound(path: Path = ALARM_SOUND) -> Path:
    """Write a loud two-tone siren (one 1 s loop) as a WAV file if it is missing."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    rate, half = 44100, 22050
    t = np.arange(half) / rate
    ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fades: no clicks at joins
    tones = [np.tanh(3 * np.sin(2 * np.pi * freq * t)) * ramp for freq in (960, 720)]  # bright, square-ish
    samples = (np.concatenate(tones) * 0.9 * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


class AlarmSound(QObject):
    """Loops the siren on this device's speakers until stopped. GUI thread only.
    Falls back to repeated system beeps if Qt Multimedia or the WAV is unavailable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ringing = False
        self._effect = None
        self._beeper = QTimer(self)
        self._beeper.setInterval(600)
        self._beeper.timeout.connect(QApplication.beep)
        if QSoundEffect is None:
            return
        try:
            self._effect = QSoundEffect(self)
            self._effect.setSource(QUrl.fromLocalFile(str(ensure_alarm_sound())))
            self._effect.setLoopCount(QSoundEffect.Loop.Infinite.value)
            self._effect.setVolume(1.0)  # App volume; the device's system volume still applies.
        except Exception as exc:
            print(f"Alarm sound unavailable, using system beeps: {exc}", file=sys.stderr)
            self._effect = None

    def start(self) -> None:
        if self.ringing:
            return
        self.ringing = True
        if self._effect is not None and self._effect.status() != QSoundEffect.Status.Error:
            self._effect.play()
        else:
            QApplication.beep()
            self._beeper.start()

    def stop(self) -> None:
        self.ringing = False
        self._beeper.stop()
        if self._effect is not None:
            self._effect.stop()


#Front end GUI BELOW.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._pixmap = None
        self._alert = False
        self._flash_on = False
        self._stopping = False
        self._closing = False
        self._error = None
        self.location = LocationFeed(self)
        self.fall_log = FallLog(device_name())
        self.log_dialog = None
        self.alarm = AlarmSound(self)
        self.settings = QSettings("Vertebrate", "Vertebrate")  # per-user, per-device preferences
        self.setWindowTitle("Live Pose · Fall Detection")
        self.resize(1120, 800)
        self.setMinimumSize(640, 480)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title = QLabel("Live Pose")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Local webcam · YOLO11 pose · five consecutive processed frames")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        self.video = QLabel("Press Start to open your webcam")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(320, 240)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video, 1)
        self.status = QLabel("Camera stopped")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(56)
        layout.addWidget(self.status)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start camera")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.log_button = QPushButton()
        self.log_button.setObjectName("logButton")
        self.log_button.setToolTip(f"Falls recorded on {self.fall_log.device}")
        self.silence_button = QPushButton("🔕  Silence alarm")
        self.silence_button.setObjectName("silenceButton")
        self.silence_button.setVisible(False)
        alarm_label = QLabel("Alarm if fallen for")
        alarm_label.setObjectName("subtitle")
        self.alarm_delay = QSpinBox()
        self.alarm_delay.setRange(0, ALARM_MAX_S)
        self.alarm_delay.setSingleStep(5)
        self.alarm_delay.setSuffix(" s")
        self.alarm_delay.setSpecialValueText("Off")  # shown at 0
        self.alarm_delay.setToolTip("Ring an alarm on this device when a detected fall lasts this long. "
                                    "0 turns the alarm off.")
        try:
            saved = int(self.settings.value("alarm_after_s", ALARM_DEFAULT_S))
        except (TypeError, ValueError):
            saved = ALARM_DEFAULT_S
        self.alarm_delay.setValue(min(max(saved, 0), ALARM_MAX_S))
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.silence_button)
        controls.addStretch()
        controls.addWidget(alarm_label)
        controls.addWidget(self.alarm_delay)
        controls.addSpacing(12)
        controls.addWidget(self.log_button)
        layout.addLayout(controls)
        footer = QLabel("Local demo — no external emergency dispatch")
        footer.setObjectName("subtitle")
        layout.addWidget(footer)
        self.setCentralWidget(container)
        # No font-family here: the platform's system font is set in main().
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #141820; color: #e7ecf4; font-size: 14px; }
            QLabel#title { font-size: 30px; font-weight: 600; }
            QLabel#subtitle { color: #9ba8bb; }
            QLabel#video { background: #090c11; border: 1px solid #303846; border-radius: 10px; }
            QPushButton { background: #344b68; border: 1px solid #50647d; border-radius: 7px; padding: 12px 24px; }
            QPushButton:hover { background: #425f83; }
            QPushButton:disabled { background: #222832; color: #737e8d; border-color: #303846; }
            QPushButton#logButton { background: #2a3140; }
            QPushButton#logButton:hover { background: #374155; }
            QPushButton#silenceButton { background: #b52e3c; border-color: #e0525f; font-weight: 700; }
            QPushButton#silenceButton:hover { background: #cc3a49; }
            QSpinBox { background: #0f131a; border: 1px solid #50647d; border-radius: 7px;
                       padding: 8px 10px; min-width: 84px; }
            QLabel#dialogTitle { font-size: 22px; font-weight: 600; }
            QTableWidget { background: #0f131a; alternate-background-color: #151b24;
                           gridline-color: #262e3b; border: 1px solid #303846; border-radius: 8px; }
            QTableWidget::item { padding: 6px 10px; }
            QTableWidget::item:selected { background: #344b68; }
            QHeaderView::section { background: #1c2330; color: #9ba8bb; border: none;
                                   border-bottom: 1px solid #303846; padding: 8px 10px; font-weight: 600; }
        """)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.log_button.clicked.connect(self.open_log)
        self.silence_button.clicked.connect(self.silence_alarm)
        self.alarm_delay.valueChanged.connect(self.alarm_delay_changed)
        self.update_log_button()
        # Checks the ongoing fall against the alarm delay independently of frame rate.
        self.alarm_timer = QTimer(self)
        self.alarm_timer.setInterval(250)
        self.alarm_timer.timeout.connect(self.check_alarm)
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(450)
        self.flash_timer.timeout.connect(self.flash_alert)
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.setInterval(5000)
        self.stop_timer.timeout.connect(self.stop_timeout)
        self.set_status("Camera stopped")

    def set_status(self, text, color="#233044"):
        self.status.setText(text)
        self.status.setStyleSheet(f"background: {color}; color: #ffffff; border-radius: 8px; padding: 10px; font-size: 19px; font-weight: 600;")

    @Slot()
    def start_camera(self):
        if self.worker is not None:
            return
        self._error = None
        self._stopping = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_status("Loading local model and opening camera…")
        self.worker = CameraThread(self)
        self.worker.frame_ready.connect(self.show_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.show_error, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.camera_finished, Qt.ConnectionType.QueuedConnection)
        self.location.start()
        self.alarm_timer.start()
        self.worker.start()

    @Slot()
    def stop_camera(self):
        if self.worker is None or self._stopping:
            return
        self._stopping = True
        self.flash_timer.stop()
        self.stop_alarm()  # Someone is at the device; stopping the camera also stops the alarm.
        self._alert = False
        self.stop_button.setEnabled(False)
        self.worker.requestInterruption()
        self.set_status("Stopping camera…")
        self.stop_timer.start()

    @Slot(QImage, bool)
    def show_frame(self, image, is_fallen):
        worker = self.sender()
        try:
            if worker is not self.worker or self._stopping:
                return
            paint_watermark(image, self.location.watermark_lines())  # image is our detached copy
            self._pixmap = QPixmap.fromImage(image)
            self.fit_video()
            if self.fall_log.update(is_fallen, self.location.current_fix()):
                self.log_changed()
            # Keep flashing while the alarm rings, even if one frame misses the person.
            if is_fallen or self.alarm.ringing:
                if not self._alert:
                    self._alert = True
                    self._flash_on = False
                    self.flash_alert()
                    self.flash_timer.start()
            else:
                self._alert = False
                self.flash_timer.stop()
                self.set_status("Monitoring · no confirmed fall", "#1e4336")
            self.check_alarm()
        finally:
            if worker is not None:
                worker.frame_consumed()

    @Slot()
    def flash_alert(self):
        self._flash_on = not self._flash_on
        delay = self.alarm_delay.value()
        current = self.fall_log.current
        fallen_for = self.fall_log.ongoing_seconds()
        if self.alarm.ringing:
            text = f"🚨 EMERGENCY ALARM · fallen for {format_duration(fallen_for)} 🚨"
            colors = ("#e0262f", "#7a0f17")
        elif current is not None and delay > 0 and current["alarm_rang"] is None:
            text = f"🚨 FALL DETECTED · alarm in {max(0, int(delay - fallen_for + 0.999))} s 🚨"
            colors = ("#b52e3c", "#631e2a")
        else:
            text = "🚨 FALL DETECTED 🚨"
            colors = ("#b52e3c", "#631e2a")
        self.set_status(text, colors[0] if self._flash_on else colors[1])

    @Slot()
    def check_alarm(self):
        """Ring once per fall when it has lasted the chosen delay; stop when it ends."""
        current = self.fall_log.current
        if current is None:
            if self.alarm.ringing:
                self.stop_alarm()  # The fall ended (posture cleared or camera stopped).
            return
        delay = self.alarm_delay.value()
        if (delay > 0 and not self.alarm.ringing and current["alarm_rang"] is None
                and self.fall_log.ongoing_seconds() >= delay):
            self.fall_log.mark_alarm(delay)
            self.alarm.start()
            self.silence_button.setVisible(True)
            self.silence_button.setFocus()  # Space/Enter silences it
            self.raise_()
            self.activateWindow()
            self.flash_alert()
            self.log_changed()

    @Slot()
    def silence_alarm(self):
        if self.alarm.ringing:
            self.fall_log.mark_silenced()
            self.log_changed()
        self.stop_alarm()

    def stop_alarm(self):
        self.alarm.stop()
        self.silence_button.setVisible(False)

    @Slot(int)
    def alarm_delay_changed(self, value):
        self.settings.setValue("alarm_after_s", value)
        if value == 0 and self.alarm.ringing:
            self.silence_alarm()  # Turning the alarm off while it rings silences it.

    @Slot(str)
    def show_error(self, message):
        self._error = message
        self.flash_timer.stop()
        self.set_status(message, "#631e2a")

    @Slot()
    def camera_finished(self):
        self.stop_timer.stop()
        self.flash_timer.stop()
        self.alarm_timer.stop()
        self.stop_alarm()
        # A fall still in progress ends here; record why.
        reason = f"Error: {self._error}" if self._error else ("App closed" if self._closing else "Camera stopped")
        if self.fall_log.close_event(reason):
            self.log_changed()
        self.location.stop()
        self._alert = False
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._stopping = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._pixmap = None
        self.video.clear()
        self.video.setText("Press Start to open your webcam")
        self.set_status(self._error or "Camera stopped", "#631e2a" if self._error else "#233044")
        if self._closing:
            self.close()

    def update_log_button(self):
        count = len(self.fall_log.rows())
        self.log_button.setText(f"Fall log ({count})" if count else "Fall log")

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

    @Slot()
    def stop_timeout(self):
        if self.worker is not None:
            self.set_status("Camera/model has not stopped yet. Waiting for the driver; this window remains responsive.", "#631e2a")

    def fit_video(self):
        if self._pixmap is not None:
            # Scale in physical pixels so Retina / high-DPI Windows displays stay sharp.
            ratio = self.video.devicePixelRatioF()
            scaled = self._pixmap.scaled(self.video.size() * ratio, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(ratio)
            self.video.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_video()

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None:
            self._closing = True
            self.stop_camera()
            event.ignore()  # Never destroy a running QThread or block the GUI.
        else:
            event.accept()


def main():
    application = QApplication(sys.argv)
    # Segoe UI on Windows, SF Pro on macOS: whatever the platform's UI font is.
    application.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
