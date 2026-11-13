"""EvidenceMeter class."""
from .common import *  # shared Qt imports, constants, and helpers

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

