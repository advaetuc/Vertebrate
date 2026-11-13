"""VideoView class."""
from .common import *  # shared Qt imports, constants, and helpers

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

