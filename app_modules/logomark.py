"""LogoMark class."""
from .common import *  # shared Qt imports, constants, and helpers

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

