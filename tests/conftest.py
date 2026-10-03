"""Create the widget-capable application before legacy QCoreApplication tests."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('PYTEST_QT_API', 'pyside6')
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])
