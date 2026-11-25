"""CameraThread class."""
from .common import *  # shared Qt imports, constants, and helpers
from .fallanalysis import FPS_LOG_EVERY_SECONDS, HISTORY_SECONDS, FrameRateMeter, write_performance_log
from .tracking import PersonTracker

class CameraThread(QThread):
    # Annotated frame, alert (any person confirmed), leading person's streak (0..5),
    # leading person's ID (0 = none), people detected, IDs of confirmed people,
    # and {person ID: report} for everyone watched after a fall (fallanalysis.py).
    # The last one MUST be `object`, not `dict`: PySide6 turns a `dict` signal
    # argument into a QVariantMap, which only allows text keys, so a dict keyed
    # by person ID arrives EMPTY in the window (tests/test_frame_signal.py).
    # A fresh dict is built for every frame, so passing the object is thread-safe.
    frame_ready = Signal(QImage, bool, int, int, int, list, object)
    # Once a second: frames the detector processes per second, median model time
    # in ms, and how many frames a person's pose history holds at that rate.
    stats_ready = Signal(float, float, int)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_pending = threading.Event()

    def frame_consumed(self):
        self._frame_pending.clear()

    def run(self):
        capture = None
        meter = FrameRateMeter()
        started = datetime.now()
        frame_size = ""
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
            last_stats = last_log = time.monotonic()
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("The camera stopped delivering frames. Check its connection and restart.")
                if self.isInterruptionRequested():
                    break
                frame_time = time.monotonic()  # one clock for tracking, history and the rate
                result = model.predict(frame, device="cpu", verbose=False)[0]
                inference = time.monotonic() - frame_time
                confirmed, evidence, leader_id, people = annotate(frame, result, tracker, frame_time)
                meter.tick(frame_time, inference)
                if not frame_size:
                    frame_size = f"{frame.shape[1]}x{frame.shape[0]}"
                if frame_time - last_stats >= 1.0:
                    last_stats = frame_time
                    self.stats_ready.emit(meter.fps, meter.inference_ms, meter.history_frames())
                if frame_time - last_log >= FPS_LOG_EVERY_SECONDS:
                    last_log = frame_time
                    print(f"[performance] {meter.fps:.1f} fps · model {meter.inference_ms:.0f} ms · "
                          f"{HISTORY_SECONDS:.0f} s history = {meter.history_frames()} frames", file=sys.stderr, flush=True)
                is_fallen = bool(confirmed)
                # The evidence count is shown by the sidebar meter, not drawn on the frame.
                # At most ONE queued frame signal: a slow GUI cannot accumulate
                # large image buffers. Inference/tracking still see every frame.
                if not self._frame_pending.is_set() and not self.isInterruptionRequested():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    self._frame_pending.set()
                    self.frame_ready.emit(image, is_fallen, evidence, leader_id, people, confirmed,
                                          tracker.aftermath())
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()  # Camera is released by its owning thread.
            summary = meter.session_summary(time.monotonic())
            if summary is not None:
                path = write_performance_log(device_name(), started, summary, frame_size)
                print(f"[performance] session: median {summary['median_fps']} fps, slowest "
                      f"{summary['slowest_fps']} fps, model {summary['inference_ms']} ms"
                      + (f" (saved to {path})" if path else ""), file=sys.stderr, flush=True)

