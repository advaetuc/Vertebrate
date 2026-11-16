"""Standalone webcam pose demo entry point.

Run with the project's Python environment: python app.py
"""
import sys
from PySide6.QtWidgets import QApplication
from app_modules.common import apply_dark_theme
from app_modules.mainwindow import MainWindow


def main():
    application = QApplication(sys.argv)
    application.setApplicationName("Vertebrate")
    apply_dark_theme(application)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
