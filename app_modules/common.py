"""Shared imports, settings, constants, and helper functions."""
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

APP_DIR = Path(__file__).resolve().parent.parent

import csv

import os

import platform

import re

import sys

import threading

import time

import wave

os.environ["YOLO_OFFLINE"] = "true"

os.environ["ULTRALYTICS_OFFLINE"] = "1"

os.environ["YOLO_AUTOINSTALL"] = "false"

os.environ["YOLO_CONFIG_DIR"] = str(APP_DIR / "runtime" / "logs" / "standalone-yolo")

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

LOCATION_INTERVAL_MS = 2000  # Requested update interval; the OS may deliver less often.

STALE_AFTER_S = 60           # Older fixes are labelled with their age.

GPS_FRESH_S = 10             # A GPS fix younger than this beats the system fix.

GPS_BAUD = int(os.environ.get("VERTEBRATE_GPS_BAUD", "9600"))

GPS_VENDOR_IDS = {0x1546, 0x091E}  # u-blox, Garmin

GPS_WORDS = ("gps", "gnss", "u-blox", "ublox", "nmea", "sirf", "garmin")

SYSTEM_SOURCE = "winrt" if IS_WINDOWS else "corelocation" if IS_MACOS else ""

FALL_LOG_DIR = APP_DIR / "runtime" / "logs" / "falls"  # Git-ignored

END_GRACE_S = 2.0

LOG_FIELDS = (
    "device", "fall_detected_at", "ended_at", "utc_offset", "duration", "duration_seconds",
    "latitude", "longitude", "accuracy_m", "location_source", "ended_by",
    "alarm_after_seconds", "alarm_rang_at", "alarm_silenced_at",
)

ALARM_DEFAULT_S = 30

ALARM_MAX_S = 600

ALARM_SOUND = APP_DIR / "runtime" / "sounds" / "alarm.wav"  # generated

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
    root = APP_DIR
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

