r"""Standalone webcam pose demo. Run: .venv\Scripts\python.exe app.py

No VERTEBRATE imports, tracker, baseline, or configuration schema. Geometry is
per-frame; the only detection history is a five-frame, scene-wide debounce.
Uses existing local YOLO weights; does not download weights or upload footage.
"""
from pathlib import Path
import os
import sys
import threading

# Set before importing Ultralytics. The installed version expects "true".
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent / "runtime" / "logs" / "standalone-yolo")

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

CONSECUTIVE_FRAMES = 5
GREEN, RED = (80, 220, 100), (60, 60, 255)  # OpenCV BGR
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)


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
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                raise RuntimeError("Cannot open camera 0. Close other camera apps and check Windows camera permissions.")
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
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        layout.addLayout(controls)
        footer = QLabel("Local demo — no external emergency dispatch")
        footer.setObjectName("subtitle")
        layout.addWidget(footer)
        self.setCentralWidget(container)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #141820; color: #e7ecf4; font-family: 'Segoe UI'; font-size: 14px; }
            QLabel#title { font-size: 30px; font-weight: 600; }
            QLabel#subtitle { color: #9ba8bb; }
            QLabel#video { background: #090c11; border: 1px solid #303846; border-radius: 10px; }
            QPushButton { background: #344b68; border: 1px solid #50647d; border-radius: 7px; padding: 12px 24px; }
            QPushButton:hover { background: #425f83; }
            QPushButton:disabled { background: #222832; color: #737e8d; border-color: #303846; }
        """)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
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
        self.worker.start()

    @Slot()
    def stop_camera(self):
        if self.worker is None or self._stopping:
            return
        self._stopping = True
        self.flash_timer.stop()
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
            self._pixmap = QPixmap.fromImage(image)
            self.fit_video()
            if is_fallen:
                if not self._alert:
                    self._alert = True
                    self._flash_on = False
                    self.flash_alert()
                    self.flash_timer.start()
            else:
                self._alert = False
                self.flash_timer.stop()
                self.set_status("Monitoring · no confirmed fall", "#1e4336")
        finally:
            if worker is not None:
                worker.frame_consumed()

    @Slot()
    def flash_alert(self):
        self._flash_on = not self._flash_on
        self.set_status("🚨 FALL DETECTED 🚨", "#b52e3c" if self._flash_on else "#631e2a")

    @Slot(str)
    def show_error(self, message):
        self._error = message
        self.flash_timer.stop()
        self.set_status(message, "#631e2a")

    @Slot()
    def camera_finished(self):
        self.stop_timer.stop()
        self.flash_timer.stop()
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

    @Slot()
    def stop_timeout(self):
        if self.worker is not None:
            self.set_status("Camera/model has not stopped yet. Waiting for the driver; this window remains responsive.", "#631e2a")

    def fit_video(self):
        if self._pixmap is not None:
            self.video.setPixmap(self._pixmap.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))

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
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
