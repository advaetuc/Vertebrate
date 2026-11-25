"""Fall analysis groundwork (phase 0): per-person pose history and frame-rate
measurement. Nothing here changes an alert yet; later phases (movement after
the fall, bystanders, sudden vs gradual, fall patterns) read this history.
Pure Python + NumPy, same on Windows and macOS. Worker thread only."""
from .common import *  # shared Qt imports, constants, and helpers

# Pose history ---------------------------------------------------------------
HISTORY_SECONDS = 5.0        # how far back each person's pose history reaches
HISTORY_MAX_SAMPLES = 300    # hard cap per person (60 fps x 5 s), whatever the frame rate
SHOULDERS = (5, 6)           # COCO keypoint indices (HIPS is in common.py)
MIN_TORSO_PX = 8.0           # a shorter shoulder-hip distance is too unreliable to scale by

# Frame rate -------------------------------------------------------------------
FPS_WINDOW_SECONDS = 2.0     # the live rate is measured over this window
FPS_LOG_EVERY_SECONDS = 10.0 # how often the rate is printed to the terminal
PERFORMANCE_LOG_DIR = APP_DIR / "runtime" / "logs" / "performance"  # Git-ignored
PERFORMANCE_FIELDS = (
    "session_start", "seconds", "median_fps", "slowest_fps", "inference_ms", "frames",
    "history_frames", "frame_size", "system", "python", "device",
)


def visible_joints(keypoints) -> np.ndarray:
    """17 x 2 float array with NaN for every joint the model did not find
    (YOLO reports a missing joint as 0, 0)."""
    joints = np.full((17, 2), np.nan, dtype=np.float32)
    if keypoints is None:
        return joints
    points = np.asarray(keypoints, dtype=np.float32)[:17, :2]
    ok = np.isfinite(points).all(axis=1) & (points[:, 0] > 0) & (points[:, 1] > 0)
    joints[: len(points)][ok] = points[ok]
    return joints


def joint_midpoint(joints: np.ndarray, indices) -> np.ndarray | None:
    """Mean of the visible joints among `indices`, or None if none is visible."""
    chosen = joints[list(indices)]
    chosen = chosen[np.isfinite(chosen).all(axis=1)]
    return chosen.mean(axis=0) if len(chosen) else None


def torso_length(joints: np.ndarray) -> float | None:
    """Shoulder-centre to hip-centre distance in pixels: the body's own ruler.
    It stays about the same whether the person stands or lies sideways, and
    shrinks or grows with distance to the camera, so movement measured in
    torso lengths means the same near and far."""
    shoulders, hips = joint_midpoint(joints, SHOULDERS), joint_midpoint(joints, HIPS)
    if shoulders is None or hips is None:
        return None
    length = float(np.hypot(*(shoulders - hips)))
    return length if length >= MIN_TORSO_PX else None


class PoseSample:
    """One person in one frame."""
    __slots__ = ("t", "box", "joints", "state", "torso")

    def __init__(self, t: float, box, joints: np.ndarray, state: str):
        self.t = t                         # time.monotonic() seconds
        self.box = tuple(map(float, box))  # x1, y1, x2, y2 in pixels
        self.joints = joints               # 17 x 2 pixels, NaN = not visible
        self.state = state                 # "standing" / "fallen" / "too_close"
        self.torso = torso_length(joints)  # pixels, or None


class PoseHistory:
    """The last HISTORY_SECONDS of one person's poses (a ring buffer by time).

    Positions can be read in pixels, or in torso lengths ("body units") from
    where the person's hips were at the start of the window, so a drop of
    1.5 means the same whether the person is 2 m or 6 m from the camera."""

    def __init__(self, seconds: float = HISTORY_SECONDS):
        self.seconds = seconds
        self._samples: deque[PoseSample] = deque(maxlen=HISTORY_MAX_SAMPLES)

    def append(self, t: float, box, keypoints, state: str = "standing") -> None:
        self._samples.append(PoseSample(t, box, visible_joints(keypoints), state))
        self.trim(t)

    def trim(self, now: float) -> None:
        while self._samples and now - self._samples[0].t > self.seconds:
            self._samples.popleft()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[PoseSample]:
        return list(self._samples)

    @property
    def latest(self) -> PoseSample | None:
        return self._samples[-1] if self._samples else None

    @property
    def duration(self) -> float:
        """Seconds between the oldest and newest sample."""
        return self._samples[-1].t - self._samples[0].t if len(self._samples) > 1 else 0.0

    def times(self) -> np.ndarray:
        return np.array([s.t for s in self._samples], dtype=np.float64)

    def joints(self) -> np.ndarray:
        """n x 17 x 2 pixel positions, NaN where a joint was not visible."""
        if not self._samples:
            return np.empty((0, 17, 2), dtype=np.float32)
        return np.stack([s.joints for s in self._samples])

    def body_scale(self) -> float | None:
        """Typical torso length in pixels over the window. Standing frames are
        preferred: a person lying towards the camera looks shorter."""
        upright = [s.torso for s in self._samples if s.torso and s.state == "standing"]
        lengths = upright or [s.torso for s in self._samples if s.torso]
        if lengths:
            return float(np.median(lengths))
        heights = [s.box[3] - s.box[1] for s in self._samples if s.state == "standing"]
        # Fallback: a standing person's torso is roughly a third of their height.
        return float(np.median(heights)) / 3.0 if heights and np.median(heights) > 0 else None

    def origin(self) -> np.ndarray | None:
        """Hip centre in the oldest sample that shows the hips (pixels)."""
        for sample in self._samples:
            hips = joint_midpoint(sample.joints, HIPS)
            if hips is not None:
                return hips
        return None

    def normalized(self) -> np.ndarray | None:
        """n x 17 x 2 positions in torso lengths from origin(); NaN where not
        visible. Image y points down, so a fall shows as y increasing.
        None until the history holds a usable body scale and origin."""
        scale, origin = self.body_scale(), self.origin()
        if not scale or origin is None:
            return None
        return (self.joints() - origin) / scale

    def summary(self) -> dict:
        """Small, readable description of the window (for logs and tests)."""
        scale = self.body_scale()
        return {
            "samples": len(self._samples),
            "seconds": round(self.duration, 2),
            "body_scale_px": round(scale, 1) if scale else None,
            "fallen_samples": sum(s.state == "fallen" for s in self._samples),
        }


class FrameRateMeter:
    """Frames the detector actually processes per second, measured in the
    camera thread. (The window may show fewer: it skips frames when busy.)
    Also keeps a per-second record of the session for the performance log."""

    def __init__(self, window: float = FPS_WINDOW_SECONDS):
        self.window = window
        self._times: deque[float] = deque()
        self._inference: deque[float] = deque(maxlen=120)
        self._per_second: deque[float] = deque(maxlen=7200)  # up to 2 hours of 1-s readings
        self._last_sample_t = None
        self.frames = 0
        self.started = None

    def tick(self, t: float, inference_seconds: float | None = None) -> None:
        if self.started is None:
            self.started = t
        self.frames += 1
        self._times.append(t)
        while self._times and t - self._times[0] > self.window:
            self._times.popleft()
        if inference_seconds is not None:
            self._inference.append(inference_seconds)
        if self._last_sample_t is None:
            self._last_sample_t = t
        elif t - self._last_sample_t >= 1.0:
            self._last_sample_t = t
            if self.fps > 0:
                self._per_second.append(self.fps)

    @property
    def fps(self) -> float:
        if len(self._times) < 2 or self._times[-1] <= self._times[0]:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    @property
    def inference_ms(self) -> float:
        return float(np.median(self._inference)) * 1000.0 if self._inference else 0.0

    def history_frames(self, fps: float | None = None) -> int:
        """How many frames a HISTORY_SECONDS window holds at this rate."""
        rate = self.fps if fps is None else fps
        return min(HISTORY_MAX_SAMPLES, int(round(rate * HISTORY_SECONDS)))

    def session_summary(self, now: float) -> dict | None:
        """Median and slowest (10th percentile) rate of the session, or None if
        it was too short to say anything."""
        if not self._per_second or self.started is None:
            return None
        readings = np.array(self._per_second)
        median = float(np.median(readings))
        return {
            "seconds": round(now - self.started, 1),
            "median_fps": round(median, 1),
            "slowest_fps": round(float(np.percentile(readings, 10)), 1),
            "inference_ms": round(self.inference_ms),
            "frames": self.frames,
            "history_frames": self.history_frames(median),
        }


def write_performance_log(device: str, started: datetime, summary: dict, frame_size: str) -> Path | None:
    """Append one row per camera session to runtime/logs/performance/<device>.csv,
    so teammates can compare laptops. Numbers only; never images."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", device).strip("._") or "this-device"
    path = PERFORMANCE_LOG_DIR / f"{safe}.csv"
    row = {
        "session_start": started.isoformat(timespec="seconds"),
        **summary,
        "frame_size": frame_size,
        "system": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "device": device,
    }
    try:
        PERFORMANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PERFORMANCE_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return path
    except OSError as exc:
        print(f"[performance] could not write {path}: {exc}", file=sys.stderr, flush=True)
        return None
