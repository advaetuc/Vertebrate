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
If a fall lasts longer than the user-set delay, a siren rings on this device
until the person gets up or someone presses Silence.
"""
from collections import deque
from datetime import datetime
from pathlib import Path
import csv
import os
import platform
import re
import sys
import threading
import time
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
    QCloseEvent, QColor, QDesktopServices, QFont, QFontDatabase, QImage, QLinearGradient, QPainter,
    QPainterPath, QPalette, QPen, QPixmap,
)
from PySide6.QtPositioning import QGeoPositionInfo, QGeoPositionInfoSource
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
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
    """Paint this result onto its own frame.
    Returns (whether any person matches, number of people detected)."""
    if result.boxes is None:
        return False, 0
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
    return bool(any_fallen), len(boxes)


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
    # Annotated frame, debounced alert, debounce count (0..5), people detected.
    frame_ready = Signal(QImage, bool, int, int)
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
                matched, people = annotate(frame, result)
                is_fallen = debounce.update(matched)
                # The evidence count is shown by the sidebar meter, not drawn on the frame.
                # At most ONE queued frame signal: a slow GUI cannot accumulate
                # large image buffers. Inference/debounce still see every frame.
                if not self._frame_pending.is_set() and not self.isInterruptionRequested():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    self._frame_pending.set()
                    self.frame_ready.emit(image, is_fallen, debounce.count, people)
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

    def problem_text(self) -> str:
        """Why there is no usable fix yet (for the sidebar)."""
        return self._problems.get("system") or self._problems.get("gps") or "Locating…"

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
# ---------------------------------------------------------------------------
# Design system: a dark "control room" theme. One accent (teal→sky) for
# actions; green / orange / red reserved for state (clear / fall / alarm).
# ---------------------------------------------------------------------------
BG = "#0B0F14"
SURFACE = "#11161E"
SURFACE_2 = "#0D1218"
BORDER = "#1E2733"
TEXT = "#E6EDF5"
MUTED = "#8A97A8"
FAINT = "#5F6B7B"
ACCENT = "#2DD4BF"
ACCENT_2 = "#38BDF8"
OK = "#34D399"
AMBER = "#F5B83D"
WARN = "#FF8A4C"
DANGER = "#F04452"
DANGER_DEEP = "#A51D2B"

# state -> (header pill text, colour)
STATE_STYLE = {
    "idle": ("STANDBY", FAINT),
    "starting": ("STARTING", ACCENT_2),
    "monitoring": ("LIVE", OK),
    "fall": ("FALL DETECTED", WARN),
    "alarm": ("ALARM", DANGER),
    "stopping": ("STOPPING", FAINT),
    "error": ("ERROR", DANGER),
}


def rgba(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


def make_label(text: str = "", size: int = 14, weight=QFont.Weight.Normal, *, mono: bool = False,
               obj: str | None = None, spacing: float = 0.0, wrap: bool = False) -> QLabel:
    """Label with an explicit font, so typography never depends on stylesheet inheritance."""
    label = QLabel(text)
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont) if mono else QFont(QApplication.font())
    font.setPixelSize(size)
    font.setWeight(weight)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    label.setFont(font)
    if obj:
        label.setObjectName(obj)
    label.setWordWrap(wrap)
    return label


def caption(text: str) -> QLabel:
    return make_label(text.upper(), 11, QFont.Weight.Bold, obj="caption", spacing=1.4)


class LogoMark(QWidget):
    """Brand mark: five stacked vertebrae in the accent gradient."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 36)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(ACCENT))
        gradient.setColorAt(1, QColor(ACCENT_2))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        widths, height, gap = (12, 17, 22, 17, 12), 5.0, 2.4
        top = (self.height() - (len(widths) * height + (len(widths) - 1) * gap)) / 2
        for i, width in enumerate(widths):
            painter.drawRoundedRect(QRectF((self.width() - width) / 2, top + i * (height + gap), width, height),
                                    2.5, 2.5)
        painter.end()


class EvidenceMeter(QWidget):
    """Five segments = the five consecutive positive frames needed for an alert."""

    def __init__(self, segments: int = CONSECUTIVE_FRAMES, parent=None):
        super().__init__(parent)
        self.segments = segments
        self.value = 0
        self.setFixedHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: int) -> None:
        value = max(0, min(self.segments, value))
        if value != self.value:
            self.value = value
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        gap = 6.0
        width = (self.width() - gap * (self.segments - 1)) / self.segments
        fill = QColor(DANGER if self.value >= self.segments else WARN if self.value >= 3 else AMBER)
        for i in range(self.segments):
            painter.setBrush(fill if i < self.value else QColor("#1C2531"))
            painter.drawRoundedRect(QRectF(i * (width + gap), 0, width, self.height()), 4, 4)
        painter.end()


class StatTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)
        self.value = make_label("—", 22, QFont.Weight.Bold, mono=True)
        layout.addWidget(self.value)
        layout.addWidget(make_label(title, 12, obj="muted"))

    def set(self, text: str, color: str | None = None) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};" if color else "")


class VideoView(QWidget):
    """Live view drawn with rounded corners, a state-coloured border and a LIVE tag.
    Draws in device pixels, so Retina and high-DPI Windows screens stay sharp."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._title, self._detail = "Camera is off", "Press Start monitoring to begin."
        self._border, self._border_width = QColor(BORDER), 1.0
        self._live = False
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def show_message(self, title: str, detail: str) -> None:
        self._pixmap, self._title, self._detail = None, title, detail
        self.update()

    def set_border(self, color: QColor, width: float, live: bool) -> None:
        self._border, self._border_width, self._live = color, width, live
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        inset = self._border_width / 2 + 0.5
        rect = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        path = QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        painter.fillPath(path, QColor("#06090D"))
        if self._pixmap is not None and not self._pixmap.isNull():
            painter.save()
            painter.setClipPath(path)
            scale = min(rect.width() / self._pixmap.width(), rect.height() / self._pixmap.height())
            w, h = self._pixmap.width() * scale, self._pixmap.height() * scale
            target = QRectF(rect.x() + (rect.width() - w) / 2, rect.y() + (rect.height() - h) / 2, w, h)
            painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
            painter.restore()
            if self._live:
                self._paint_live_tag(painter, rect)
        else:
            self._paint_placeholder(painter, rect)
        painter.setPen(QPen(self._border, self._border_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()

    def _paint_live_tag(self, painter, rect):
        font = QFont(QApplication.font())
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setFont(font)
        text = "LIVE"
        width = painter.fontMetrics().horizontalAdvance(text) + 30
        tag = QRectF(rect.right() - width - 14, rect.top() + 14, width, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(tag, 12, 12)
        painter.setBrush(QColor(DANGER))
        painter.drawEllipse(QRectF(tag.left() + 10, tag.center().y() - 4, 8, 8))
        painter.setPen(QColor(TEXT))
        painter.drawText(tag.adjusted(22, 0, 0, 0),
                         int((Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft).value), text)

    def _paint_placeholder(self, painter, rect):
        # Subtle dot grid gives the empty panel depth without competing with content.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 10))
        step = 28
        y = rect.top() + step
        while y < rect.bottom():
            x = rect.left() + step
            while x < rect.right():
                painter.drawEllipse(QRectF(x - 1, y - 1, 2, 2))
                x += step
            y += step
        cx, cy = rect.center().x(), rect.center().y() - 34
        # Camera glyph: rounded body + lens.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(FAINT), 2))
        painter.drawRoundedRect(QRectF(cx - 30, cy - 20, 60, 40), 9, 9)
        painter.drawEllipse(QRectF(cx - 11, cy - 11, 22, 22))
        painter.drawRoundedRect(QRectF(cx - 12, cy - 27, 24, 7), 3, 3)
        title_font = QFont(QApplication.font())
        title_font.setPixelSize(19)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor(TEXT))
        painter.drawText(QRectF(rect.left() + 24, cy + 36, rect.width() - 48, 28),
                         int(Qt.AlignmentFlag.AlignCenter.value), self._title)
        detail_font = QFont(QApplication.font())
        detail_font.setPixelSize(13)
        painter.setFont(detail_font)
        painter.setPen(QColor(MUTED))
        painter.drawText(QRectF(rect.left() + 40, cy + 66, rect.width() - 80, 60),
                         int((Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop).value)
                         | int(Qt.TextFlag.TextWordWrap.value),
                         self._detail)


STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#root, QWidget#sideInner {{ background: transparent; }}
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#caption {{ color: {FAINT}; }}
QLabel#chip {{ color: #AEB9C7; background: {SURFACE}; border: 1px solid {BORDER};
              border-radius: 12px; padding: 5px 12px; }}
QLabel#dialogTitle {{ font-size: 22px; font-weight: 600; }}
QLabel#subtitle {{ color: {MUTED}; }}
QFrame#header {{ border: none; border-bottom: 1px solid #161D27; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#tile {{ background: {SURFACE_2}; border: 1px solid #19212C; border-radius: 12px; }}

QPushButton {{ color: {TEXT}; background: #18202B; border: 1px solid #263140;
              border-radius: 10px; padding: 11px 20px; font-weight: 600; }}
QPushButton:hover {{ background: #1E2835; border-color: #33404F; }}
QPushButton:pressed {{ background: #151C26; }}
QPushButton:disabled {{ color: #4E5A69; background: #121820; border-color: #1A222D; }}
QPushButton#primary {{ color: #04131A; border: none; font-weight: 700;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_2}); }}
QPushButton#primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5EEAD4, stop:1 #7DD3FC); }}
QPushButton#primary:disabled {{ color: #4E5A69; background: #121820; border: 1px solid #1A222D; }}
QPushButton#ghost {{ background: transparent; }}
QPushButton#ghost:hover {{ background: #18202B; }}
QPushButton#ghost:disabled {{ background: transparent; }}
QPushButton#step {{ padding: 0px; min-width: 34px; max-width: 34px; min-height: 34px; max-height: 34px;
                   border-radius: 9px; font-size: 18px; }}
QPushButton#silence {{ color: {DANGER_DEEP}; background: #FFFFFF; border: none; font-weight: 800;
                      padding: 11px 22px; }}
QPushButton#silence:hover {{ background: #FFE4E6; }}

QSpinBox {{ color: {TEXT}; background: {SURFACE_2}; border: 1px solid #263140; border-radius: 9px;
           padding: 6px 4px; min-width: 72px; min-height: 20px; font-weight: 700;
           selection-background-color: #2B6F8F; }}
QSpinBox:focus {{ border-color: {ACCENT}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #263140; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #33404F; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QToolTip {{ color: {TEXT}; background: #18202B; border: 1px solid #2A3644; padding: 6px 8px; }}

QTableWidget {{ color: {TEXT}; background: {SURFACE_2}; alternate-background-color: {SURFACE};
               gridline-color: #19212C; border: 1px solid {BORDER}; border-radius: 12px; }}
QTableWidget::item {{ padding: 6px 10px; }}
QTableWidget::item:selected {{ background: #1F3A4D; color: {TEXT}; }}
QHeaderView::section {{ background: #141B24; color: {MUTED}; border: none;
                       border-bottom: 1px solid {BORDER}; padding: 9px 10px; font-weight: 600; }}
"""


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


def apply_dark_theme(application: QApplication) -> None:
    """Fusion + a dark palette gives the same look on Windows and macOS."""
    application.setStyle("Fusion")
    try:  # Qt 6.8+: dark native title bars on Windows 11 and macOS.
        application.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass
    palette = QPalette()
    roles = QPalette.ColorRole
    for role, color in ((roles.Window, BG), (roles.WindowText, TEXT), (roles.Base, SURFACE_2),
                        (roles.AlternateBase, SURFACE), (roles.Text, TEXT), (roles.Button, "#18202B"),
                        (roles.ButtonText, TEXT), (roles.Highlight, "#2B6F8F"), (roles.HighlightedText, "#FFFFFF"),
                        (roles.ToolTipBase, "#18202B"), (roles.ToolTipText, TEXT), (roles.PlaceholderText, FAINT),
                        (roles.Link, ACCENT_2)):
        palette.setColor(role, QColor(color))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#4E5A69"))
    application.setPalette(palette)
    # Segoe UI on Windows, SF Pro on macOS: whatever the platform's UI font is.
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(14)
    application.setFont(font)


def main():
    application = QApplication(sys.argv)
    application.setApplicationName("Vertebrate")
    apply_dark_theme(application)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
