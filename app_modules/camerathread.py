"""CameraThread class."""
from .common import *  # shared Qt imports, constants, and helpers
from .debounce import Debounce

class CameraThread(QThread):
    # Annotated frame, debounced alert, debounce count (0..5), people detected.
    frame_ready = Signal(QImage, bool, int, int)
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
            capture = open_camera(0)
            debounce = Debounce()
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("The camera stopped delivering frames. Check its connection and restart.")
                if self.isInterruptionRequested():
                    break
                result = model.predict(frame, device="cpu", verbose=False)[0]
                matched, people = annotate(frame, result)
                is_fallen = debounce.update(matched)
                # The evidence count is shown by the sidebar meter, not drawn on the frame.
                # At most ONE queued frame signal: a slow GUI cannot accumulate
                # large image buffers. Inference/debounce still see every frame.
                if not self._frame_pending.is_set() and not self.isInterruptionRequested():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    self._frame_pending.set()
                    self.frame_ready.emit(image, is_fallen, debounce.count, people)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()  # Camera is released by its owning thread.

