"""UrgencyMeter class: how urgent the ongoing fall is."""
from .common import *  # shared Qt imports, constants, and helpers
from .fallanalysis import PRIORITY_STYLE

# Segments from left to right; a level fills itself and every segment before it.
URGENCY_LEVELS = ("check", "urgent", "critical")


class UrgencyMeter(QWidget):
    """Three segments -- Check, Urgent, Critical -- filled up to the fall's
    priority in that priority's colour (the same colours as the banner and the
    Discord alerts). All three turn green when the person got up; all are dim
    when there is no fall."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.priority = None  # None, or a key of PRIORITY_STYLE
        self.setFixedHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Check · Urgent · Critical — the priority of the ongoing fall")

    def set_priority(self, priority: str | None) -> None:
        priority = priority if priority in PRIORITY_STYLE else None
        if priority != self.priority:
            self.priority = priority
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        gap = 6.0
        count = len(URGENCY_LEVELS)
        width = (self.width() - gap * (count - 1)) / count
        if self.priority == "recovered":
            filled = count
        elif self.priority in URGENCY_LEVELS:
            filled = URGENCY_LEVELS.index(self.priority) + 1
        else:
            filled = 0
        fill = QColor(PRIORITY_STYLE[self.priority][3]) if self.priority else QColor("#1C2531")
        for i in range(count):
            painter.setBrush(fill if i < filled else QColor("#1C2531"))
            painter.drawRoundedRect(QRectF(i * (width + gap), 0, width, self.height()), 4, 4)
        painter.end()
