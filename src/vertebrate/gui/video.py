"""Main-thread painting of one sequence-aligned, owned image and overlay."""
from PySide6.QtCore import QPointF, QRectF, Qt, QThread
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def assert_gui_thread():
    app = QApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError('Widgets require the Qt main thread')


class VideoPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.result = None
        self.image = QImage()
        self.skeleton = True
        self.setAccessibleName('Video and person skeletons')

    def set_result(self, result):
        assert_gui_thread()
        self.result = result
        if result is None:
            self.image = QImage()
        else:
            frame = result.frame
            rgb = frame.frame_bgr[:, :, ::-1].copy()
            # QImage.copy owns its pixels independently of NumPy and the worker.
            self.image = QImage(rgb.data, frame.width, frame.height,
                                rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def set_skeleton(self, enabled):
        assert_gui_thread()
        self.skeleton = enabled
        self.update()

    def paintEvent(self, event):
        assert_gui_thread()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#18232b'))
        if self.image.isNull():
            painter.setPen(QColor('#dce5e8'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Choose a local video or camera, then Start')
            return
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF((self.width()-size.width())/2, (self.height()-size.height())/2, size.width(), size.height())
        painter.drawImage(target, self.image)
        obs = self.result.observation
        if not self.skeleton or obs is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = target.width()/obs.width, target.height()/obs.height
        def point(xy): return QPointF(target.left()+xy[0]*sx, target.top()+xy[1]*sy)
        painter.setPen(QPen(QColor('#8fdfcd'), 2))
        for box, xy, valid in zip(obs.boxes_xyxy, obs.keypoints_xy, obs.keypoint_valid_mask):
            painter.drawRect(QRectF(point(box[:2]), point(box[2:])))
            for a, b in EDGES:
                if valid[a] and valid[b]:
                    painter.drawLine(point(xy[a]), point(xy[b]))
