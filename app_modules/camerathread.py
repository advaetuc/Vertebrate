"""CameraThread class."""
from .common import *  # shared Qt imports, constants, and helpers
from .tracking import PersonTracker

class CameraThread(QThread):
    # Annotated frame, alert (any person confirmed), leading person's streak (0..5),
    # leading person's ID (0 = none), people detected, IDs of confirmed people.
    frame_ready = Signal(QImage, bool, int, int, int, list)
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
            tracker = PersonTracker()  # fresh IDs and streaks for each camera session
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("The camera stopped delivering frames. Check its connection and restart.")
                if self.isInterruptionRequested():
                    break
                result = model.predict(frame, device="cpu", verbose=False)[0]
                confirmed, evidence, leader_id, people = annotate(frame, result, tracker)
                is_fallen = bool(confirmed)
                # The evidence count is shown by the sidebar meter, not drawn on the frame.
                # At most ONE queued frame signal: a slow GUI cannot accumulate
                # large image buffers. Inference/tracking still see every frame.
                if not self._frame_pending.is_set() and not self.isInterruptionRequested():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    self._frame_pending.set()
                    self.frame_ready.emit(image, is_fallen, evidence, leader_id, people, confirmed)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()  # Camera is released by its owning thread.

