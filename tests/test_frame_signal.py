"""Regression test: the per-person fall analysis must reach the window intact.

PySide6 turns a `dict` signal argument into a QVariantMap, which only allows
text keys, so a dict keyed by person ID used to arrive EMPTY in the window:
no fall type, priority or people nearby in the banner, log or Discord alerts.

Run from the project folder (no camera needed):
    Windows  .venv\\Scripts\\python.exe -m unittest discover -s tests -v
    macOS    .venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QObject, Qt, QTimer, Slot  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from app_modules.camerathread import CameraThread  # noqa: E402


class _Receiver(QObject):
    def __init__(self):
        super().__init__()
        self.got = None

    @Slot(QImage, bool, int, int, int, list, object)
    def take(self, *args):
        self.got = args


class FrameSignalTest(unittest.TestCase):
    def test_analysis_survives_a_queued_signal(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        worker = CameraThread()
        receiver = _Receiver()
        worker.frame_ready.connect(receiver.take, Qt.ConnectionType.QueuedConnection)
        analysis = {1: {"fall_type": "sudden", "pattern": "slip", "priority": "urgent", "nearby": [2]}}
        worker.frame_ready.emit(QImage(4, 4, QImage.Format.Format_RGB888), True, 5, 1, 2, [1], analysis)
        QTimer.singleShot(200, app.quit)
        app.exec()
        self.assertIsNotNone(receiver.got, "frame_ready was not delivered")
        self.assertEqual(receiver.got[5], [1])
        self.assertEqual(receiver.got[6], analysis, "the analysis dict lost its person-ID keys on the way")
        self.assertEqual(list(receiver.got[6]), [1])

    def test_window_slot_matches_the_signal(self):
        source = (ROOT / "app_modules" / "mainwindow.py").read_text(encoding="utf-8")
        self.assertIn("@Slot(QImage, bool, int, int, int, list, object)", source)
        self.assertNotIn("@Slot(QImage, bool, int, int, int, list, dict)", source)


if __name__ == "__main__":
    unittest.main()
