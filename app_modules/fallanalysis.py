"""Fall analysis: per-person pose history and frame-rate measurement (phase 0),
the watch after a confirmed fall: is the person moving, lying still, or
back on their feet? (phase 1), who else is around them: nearby, coming
over, or already with them (phase 2), and how the fall happened -- sudden or
gradual -- with a priority for each fall (phase 3), and the pattern of a
sudden fall: collapse, slip or trip (phase 5A), and of a gradual fall: hand to
chest, unsteady, or a controlled descent (phase 5B).
Pure Python + NumPy, same on Windows and macOS. Worker thread only."""
from .common import *  # shared Qt imports, constants, and helpers
import warnings

# Pose history ---------------------------------------------------------------
HISTORY_SECONDS = 8.0        # how far back each person's pose history reaches (a slow fall fits)
HISTORY_MAX_SAMPLES = 480    # hard cap per person (60 fps x 8 s), whatever the frame rate
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


UPRIGHT_MIN = 0.7  # vertical share of the shoulder-to-hip line for an upright torso (~45°)


def torso_upright(keypoints) -> bool:
    """Shoulders clearly above hips: kneeling, crouching, sitting or standing,
    as opposed to lying. False when shoulders or hips are not visible."""
    joints = visible_joints(keypoints)
    shoulders, hips = joint_midpoint(joints, SHOULDERS), joint_midpoint(joints, HIPS)
    if shoulders is None or hips is None:
        return False
    dx, dy = hips - shoulders  # image y points down: upright means hips below shoulders
    length = float(np.hypot(dx, dy))
    return length >= MIN_TORSO_PX and dy / length >= UPRIGHT_MIN


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

    def scale_source(self) -> str | None:
        """Where body_scale() came from: 'standing' (reliable), 'lying' (a torso
        lying towards the camera looks short), 'box' (a guess from height) or None."""
        if any(s.torso and s.state == "standing" for s in self._samples):
            return "standing"
        if any(s.torso for s in self._samples):
            return "lying"
        return "box" if self.body_scale() else None

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


# Movement after a fall (phase 1) -----------------------------------------------
MOVE_SPAN_S = 1.0          # movement is measured between now and about 1 s ago
MOVE_AVERAGE_S = 0.5       # ...comparing the average pose over this long at each end
MOVE_CONFIRM_FRAMES = 2    # frames in a row above the threshold before it counts as movement
MOVE_THRESHOLD = 0.15      # torso lengths a body part must move in that time to count
MOVE_MIN_JOINTS = 3        # joints that must be visible in both frames to judge at all
MOVE_MAX_GAP_S = 0.5       # longer gaps between frames do not add to "still" time
STILL_CRITICAL_S = 15.0    # lying still this long after a confirmed fall -> critical
MOVING_RECENT_S = 3.0      # moved within this long -> "moving"
RECOVER_S = 1.0            # standing this long after a fall -> got up
RECOVER_RISE = 0.8         # ...with the hips this many torso lengths above where they were
WATCH_KEEP_S = 10.0        # a recovery is still reported this long afterwards
LANDING_S = 1.5            # movement in this long after confirmation is the landing itself: ignored

# Sudden vs gradual (phase 3) ------------------------------------------------------
# Descent time = how long the hips take to go from standing height to the floor,
# timed between 10 % and 90 % of the drop (then scaled up to the whole drop), so
# a slow start or a bounce on landing does not distort it. Real falls speed up
# (and gradual ones also slow down at the end), so the middle 80 % of the drop
# takes about 65 % of the whole descent, not 80 %.
DESCENT_SHARE = 0.65       # share of the whole descent spent between 10 % and 90 % of the drop
SUDDEN_MAX_S = 1.5         # whole descent shorter than this = sudden, longer = gradual
BORDERLINE_S = (1.0, 2.0)  # descents in this range are labelled "borderline"
MIN_DROP_TORSOS = 0.6      # the hips must drop at least this much to judge the speed
MAX_FRAME_GAP_S = 0.75     # a longer gap between frames during the drop = cannot time it
TYPE_SETTLE_S = 1.5        # the type is re-measured for this long after confirmation

# Priority of a fall, highest first. The emoji and words are used on Discord.
PRIORITIES = ("critical", "urgent", "check", "recovered")
PRIORITY_STYLE = {          # emoji, word, Discord colour, colour in the window
    "critical":  ("🔴", "CRITICAL", 0xB0102A, DANGER),
    "urgent":    ("🟠", "URGENT", 0xFF8A4C, WARN),
    "check":     ("🟡", "CHECK", 0xF5B83D, AMBER),
    "recovered": ("🟢", "RECOVERED", 0x34D399, OK),
}


def descent_analysis(history: PoseHistory) -> dict:
    """How fast this person went down, from their pose history.

    Returns {"type": "sudden" | "gradual" | "unclear", "descent_s": seconds or
    None, "drop": torso lengths, "borderline": bool, "reason": text}."""
    def unclear(reason, drop=0.0):
        return {"type": "unclear", "descent_s": None, "drop": round(drop, 2), "borderline": False,
                "reason": reason}
    points = []  # (time, hip y in pixels, posture state)
    for sample in history.samples:
        hips = joint_midpoint(sample.joints, HIPS)
        if hips is not None:
            points.append((sample.t, float(hips[1]), sample.state))
    if len(points) < 4:
        return unclear("too few frames with the hips visible")
    scale = history.body_scale()
    if not scale:
        return unclear("body size unknown")
    floor_y = float(np.median([y for _, y, _ in points[-3:]]))
    upright = [y for _, y, state in points if state == "standing"]
    if not upright:
        return unclear("not seen standing before the fall")
    top = min(upright)
    stand_y = float(np.median([y for y in upright if y <= top + 0.3 * scale]))
    drop = (floor_y - stand_y) / scale
    if drop < MIN_DROP_TORSOS:
        return unclear("the hips hardly dropped (sitting, or already low)", drop)
    low = stand_y + 0.1 * (floor_y - stand_y)    # image y grows downwards
    high = stand_y + 0.9 * (floor_y - stand_y)
    crossing, last_low = None, None
    for i, (_, y, _) in enumerate(points):
        if y <= low:
            last_low = i
        elif y >= high and last_low is not None:
            crossing = (last_low, i)  # keep the last complete descent
            last_low = None
    if crossing is None:
        return unclear("the start of the fall is not in view", drop)
    i10, i90 = crossing
    if max(points[i + 1][0] - points[i][0] for i in range(i10, i90)) > MAX_FRAME_GAP_S:
        return unclear("too few frames during the fall (low frame rate)", drop)

    def when(i, level):  # interpolated time the hips passed `level` between samples i and i+1
        (t0, y0, _), (t1, y1, _) = points[i], points[i + 1]
        return t0 if y1 == y0 else t0 + (t1 - t0) * (level - y0) / (y1 - y0)
    i10_end = next(i for i in range(i10, i90) if points[i + 1][1] > low)
    i90_start = max(i for i in range(i10, i90) if points[i][1] < high)
    t_low, t_high = when(i10_end, low), when(i90_start, high)
    seconds = max((t_high - t_low) / DESCENT_SHARE, 0.0)
    return {
        "t_start": t_low, "t_end": t_high,  # when the hips passed 10 % and 90 % of the drop
        "type": "sudden" if seconds < SUDDEN_MAX_S else "gradual",
        "descent_s": round(seconds, 2),
        "drop": round(drop, 2),
        "borderline": BORDERLINE_S[0] <= seconds <= BORDERLINE_S[1],
        "reason": "",
    }


def fall_type_text(report: dict) -> str:
    """'Sudden fall (down in 0.6 s)', 'Gradual fall (down in 3.2 s)', 'Fall type unclear'.
    With a pattern: 'Possible slip — sudden fall (down in 0.6 s)'."""
    kind = report.get("fall_type", "unclear")
    if kind == "unclear" or report.get("descent_s") is None:
        return "Fall type unclear"
    text = f"{kind.capitalize()} fall (down in {report['descent_s']:.1f} s"
    text += ", borderline)" if report.get("borderline") else ")"
    pattern = report.get("pattern")
    if pattern in PATTERN_TEXT:
        text = f"{PATTERN_TEXT[pattern][0]} — {text[0].lower()}{text[1:]}"
    return text


def pattern_detail(report: dict) -> str:
    """One line on why: 'Hips reached the floor first; fell backwards; arms reached out.'"""
    if report.get("pattern") not in PATTERN_TEXT:
        return ""
    parts = []
    first = report.get("first_contact")
    if first:
        parts.append(f"{PART_NAMES[first]} reached the floor first")
    if report.get("direction"):
        parts.append(f"fell {report['direction']}")
    arms = report.get("arms")
    if arms is True:
        parts.append("arms reached out")
    elif arms is False:
        parts.append("no arm movement to break the fall")
    if report.get("chest_s"):
        if report["chest_s"] >= CHEST_MIN_S:
            parts.append(f"hand held to the chest for {report['chest_s']:.1f} s")
    if report.get("sway"):
        parts.append("swaying before the fall")
    if report.get("reaching"):
        parts.append("arms out, as if reaching for support")
    if report.get("pattern") == "controlled":
        parts.append("sat down first" if report.get("sat_first") else "went down slowly and smoothly")
    text = "; ".join(parts) + "." if parts else ""
    return text[:1].upper() + text[1:]


def priority_of(report: dict) -> str:
    """Priority for one fallen person:
    got up -> recovered; lying still, a collapse and not moving since, or a
    hand held to the chest -> critical; a slow, controlled descent -> check;
    everything else (sudden, unsteady, unclear) -> urgent."""
    if report.get("status") == "recovered":
        return "recovered"
    if report.get("status") == "still" or (report.get("critical") and report.get("status") != "moving"):
        return "critical"
    if report.get("pattern") == "collapse" and report.get("status") != "moving":
        return "critical"  # sudden, no protective reaction: possible loss of consciousness
    if report.get("pattern") == "chest":
        return "critical"  # possible chest pain: even if they are moving
    if report.get("pattern") == "controlled" and not report.get("borderline"):
        return "check"
    return "urgent"


def highest_priority(priorities) -> str | None:
    found = [p for p in priorities if p in PRIORITIES]
    return min(found, key=PRIORITIES.index) if found else None


# Patterns of a sudden fall (phase 5A) -------------------------------------------
# Which body part reaches the floor first, whether the arms reach out to break
# the fall, and the direction. Each part's "floor" is where that part ends up
# when lying, so this works whatever the camera angle; a part has "landed" when
# it has covered LANDED_SHARE of its own drop.
PARTS = {"hips": (11, 12), "knees": (13, 14), "hands": (9, 10), "upper": (0, 5, 6)}
PART_NAMES = {"hips": "hips", "knees": "knees", "hands": "hands", "upper": "head or shoulders"}
LANDED_SHARE = 0.85        # share of a part's own drop that counts as "on the floor"
MIN_PART_DROP = 0.4        # torso lengths a part must drop to be timed at all
TIE_S = 0.1                # parts landing this close together landed "at the same time"
ARM_OUT_TORSOS = 0.8       # wrist this far from the shoulder-hip segment = arm out
ARM_OUT_FRAMES = 2         # ...on at least this many frames of the fall
FACING_MIN = 0.15          # nose this far ahead of the shoulders (profile view) = facing that way
DIRECTION_MIN = 0.8        # head must end this far from the hips' start, sideways in the image
PATTERN_PRE_S, PATTERN_POST_S = 0.4, 0.6  # analysed window around the drop of the hips

PATTERN_TEXT = {           # label, what it may mean (always "possible": camera analysis only)
    "collapse": ("Collapse", "possible loss of consciousness (fainting, seizure or a heart problem)"),
    "slip": ("Possible slip", "fell backwards or onto the hips"),
    "trip": ("Possible trip", "fell forwards and tried to break the fall"),
    "chest": ("Hand to chest", "possible chest pain or breathing trouble"),
    "unsteady": ("Unsteady before falling", "possible dizziness, weakness, or low blood pressure or sugar"),
    "controlled": ("Slow, controlled descent", "may have sat or lain down on purpose, or felt faint"),
}

VIDEO_LABEL = {"chest": "hand to chest"}  # pattern names on the video (ASCII only)

# Patterns of a gradual fall (phase 5B) ------------------------------------------
GRADUAL_PRE_S = 3.0        # how long before the hips start dropping is looked at
CHEST_SPOT = 0.35          # chest = this far from the shoulders towards the hips
CHEST_RADIUS = 0.45        # a wrist this close (torso lengths) to that spot = hand on chest
CHEST_MIN_S = 1.0          # ...held at least this long (before or while going down)
CHEST_GAP_S = 0.3          # a gap this short (a missed wrist) does not break the hold
SWAY_STEP = 0.2            # torso lengths the hips move one way, then back, to count a sway
SWAY_MIN_REVERSALS = 2     # left-right-left (walking in one direction is not swaying)
REACH_MIN_S = 0.5          # arms out (ARM_OUT_TORSOS) this long before half-way down = reaching
SMOOTH_MAX_SPEED = 1.6     # torso lengths/s: a descent never faster than this is smooth
SIT_PAUSE_S = 0.6          # hips resting this long part-way down = sat first
SIT_PAUSE_MOVE = 0.1       # ...moving no more than this (torso lengths)


def _distance_to_segment(point, a, b) -> float:
    """Distance (pixels) from `point` to the segment a-b: an arm hanging along the
    body is close to the shoulder-hip segment; an arm reaching forwards, sideways
    or up over the head is far from it."""
    ab, ap = b - a, point - a
    u = float(np.clip(np.dot(ap, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
    return float(np.hypot(*(ap - u * ab)))


def _part_track(samples, indices):
    """(time, x, y) of the mean of the visible joints in `indices`, per sample."""
    track = []
    for sample in samples:
        point = joint_midpoint(sample.joints, indices)
        if point is not None:
            track.append((sample.t, float(point[0]), float(point[1])))
    return track


def _arm_out(sample: PoseSample, scale: float) -> bool | None:
    """A wrist well away from the shoulder-hip segment (None if not visible)."""
    shoulders, hips = joint_midpoint(sample.joints, SHOULDERS), joint_midpoint(sample.joints, HIPS)
    wrists = [sample.joints[i] for i in (9, 10) if np.isfinite(sample.joints[i]).all()]
    if shoulders is None or hips is None or not wrists or np.hypot(*(hips - shoulders)) < MIN_TORSO_PX:
        return None
    return max(_distance_to_segment(w, shoulders, hips) for w in wrists) / scale >= ARM_OUT_TORSOS


def _hand_on_chest(sample: PoseSample, scale: float) -> bool | None:
    shoulders, hips = joint_midpoint(sample.joints, SHOULDERS), joint_midpoint(sample.joints, HIPS)
    wrists = [sample.joints[i] for i in (9, 10) if np.isfinite(sample.joints[i]).all()]
    if shoulders is None or hips is None or not wrists:
        return None
    chest = shoulders + CHEST_SPOT * (hips - shoulders)
    return min(float(np.hypot(*(w - chest))) for w in wrists) / scale <= CHEST_RADIUS


def _longest_hold(samples, test) -> float:
    """Longest stretch of time (s) for which test(sample) is True, bridging short gaps."""
    best, start, last_true = 0.0, None, None
    for sample in samples:
        ok = test(sample)
        if ok:
            if start is None or sample.t - last_true > CHEST_GAP_S:
                start = sample.t
            last_true = sample.t
            best = max(best, last_true - start)
    return best


def _reversals(values, step) -> int:
    """How many times a series turns back after moving at least `step` (a zigzag count).
    Walking steadily one way gives 0; left-right-left gives 2."""
    if not values:
        return 0
    turns, direction, extreme = 0, 0, values[0]
    for v in values[1:]:
        if direction == 0:                      # not moved far enough yet to have a direction
            if abs(v - values[0]) >= step:
                direction, extreme = (1 if v > values[0] else -1), v
        elif (v - extreme) * direction > 0:
            extreme = v                         # still going the same way
        elif abs(v - extreme) >= step:
            turns += 1                          # turned back far enough
            direction, extreme = -direction, v
    return turns


def gradual_pattern(history: PoseHistory, descent: dict) -> dict:
    """Pattern of a GRADUAL fall from the seconds before and during the descent:
    hand held to the chest -> 'chest'; swaying or arms out reaching for
    support -> 'unsteady'; a smooth descent or sitting first -> 'controlled'."""
    out = {"pattern": "unclear"}
    scale = history.body_scale()
    samples = history.samples
    if not scale or len(samples) < 6:
        return out
    t_start, t_end = descent["t_start"], descent["t_end"]
    window = [s for s in samples if t_start - GRADUAL_PRE_S <= s.t <= t_end + 0.5]
    pre = [s for s in window if s.t <= t_start]
    early = [s for s in window if s.t <= t_start + 0.5 * (t_end - t_start)]  # before half-way down
    if len(window) < 6:
        return out

    # 1. Hand to the chest, before or while going down
    seen = [s for s in window if _hand_on_chest(s, scale) is not None]
    if len(seen) >= len(window) // 2:
        out["chest_s"] = round(_longest_hold(window, lambda s: bool(_hand_on_chest(s, scale))), 1)

    # 2. Unsteady: hips swaying side to side, or arms out (reaching for support)
    hip_x = [float(p[0]) / scale for p in (joint_midpoint(s.joints, HIPS) for s in pre) if p is not None]
    if len(hip_x) >= 4:
        smooth_x = [float(np.mean(hip_x[max(0, i - 1):i + 2])) for i in range(len(hip_x))]  # calm the wobble
        out["sway"] = _reversals(smooth_x, SWAY_STEP) >= SWAY_MIN_REVERSALS
    arms_seen = [s for s in early if _arm_out(s, scale) is not None]
    if len(arms_seen) >= max(3, len(early) // 2):
        out["reaching"] = _longest_hold(early, lambda s: bool(_arm_out(s, scale))) >= REACH_MIN_S

    # 3. Controlled: never fast, or a pause part-way down (sat first)
    hips = [(s.t, float(p[1]) / scale) for s, p in ((s, joint_midpoint(s.joints, HIPS)) for s in window)
            if p is not None and t_start - 0.2 <= s.t <= t_end + 0.2]
    if len(hips) >= 4:
        ys = np.array([y for _, y in hips]); ts = np.array([t for t, _ in hips])
        ys = np.convolve(ys, np.ones(3) / 3, mode="same") if len(ys) >= 5 else ys
        speeds = np.diff(ys[1:-1]) / np.maximum(np.diff(ts[1:-1]), 1e-3) if len(ys) >= 5 else np.diff(ys) / np.maximum(np.diff(ts), 1e-3)
        out["smooth"] = bool(len(speeds)) and float(np.max(speeds)) <= SMOOTH_MAX_SPEED
        sat = False
        for i in range(len(hips)):
            j = i
            while j + 1 < len(hips) and abs(hips[j + 1][1] - hips[i][1]) <= SIT_PAUSE_MOVE:
                j += 1
            if hips[j][0] - hips[i][0] >= SIT_PAUSE_S and i > 0 and j < len(hips) - 1:
                sat = True
                break
        out["sat_first"] = sat

    if out["chest_s"] is not None and out["chest_s"] >= CHEST_MIN_S:
        out["pattern"] = "chest"
    elif out["sway"] or out["reaching"]:
        out["pattern"] = "unsteady"
    elif out["sat_first"] or out["smooth"]:
        out["pattern"] = "controlled"
    return out


def fall_pattern(history: PoseHistory, descent: dict) -> dict:
    """Pattern of the fall. Sudden: 'collapse', 'slip', 'trip' or 'unclear';
    gradual: 'chest', 'unsteady', 'controlled' or 'unclear'; type unclear: None.
    Also returns the observations behind it: first_contact (a PARTS key or None),
    direction ('forward' / 'backward' / None), arms (True = reached out, False =
    did not, None = not visible); for gradual falls chest_s (seconds a hand was
    on the chest), sway (bool), reaching (bool), sat_first (bool), smooth (bool)."""
    result = {"pattern": None, "first_contact": None, "direction": None, "arms": None,
              "chest_s": None, "sway": None, "reaching": None, "sat_first": None, "smooth": None}
    if descent.get("type") == "gradual" and "t_start" in descent:
        return {**result, **gradual_pattern(history, descent)}
    if descent.get("type") != "sudden" or "t_start" not in descent:
        return result
    result["pattern"] = "unclear"
    scale = history.body_scale()
    samples = history.samples
    if not scale or len(samples) < 4:
        return result
    t0, t1 = descent["t_start"] - PATTERN_PRE_S, descent["t_end"] + PATTERN_POST_S
    before = [s for s in samples if s.t < descent["t_start"] and s.state == "standing"]
    during = [s for s in samples if t0 <= s.t <= t1]
    after = samples[-3:]
    if not before or len(during) < 3:
        return result

    # 1. First contact: when each part covered LANDED_SHARE of its own drop.
    landed = {}
    for name, indices in PARTS.items():
        start = _part_track(before[-4:], indices)
        end = _part_track(after, indices)
        track = _part_track(during, indices)
        if not start or not end or len(track) < 3:
            continue
        y_start = float(np.median([y for _, _, y in start]))
        y_end = float(np.median([y for _, _, y in end]))
        if (y_end - y_start) / scale < MIN_PART_DROP:
            continue
        level = y_start + LANDED_SHARE * (y_end - y_start)
        hit = next((t for t, _, y in track if y >= level), None)
        if hit is not None:
            landed[name] = hit
    first_parts = []
    if landed:
        earliest = min(landed.values())
        first_parts = [name for name, t in landed.items() if t - earliest <= TIE_S]
        if "hands" in first_parts:
            result["first_contact"] = "hands"   # hands down with anything else = hands first
        elif len(first_parts) == 1:
            result["first_contact"] = first_parts[0]
        elif set(first_parts) == {"hips", "knees"}:
            result["first_contact"] = None      # sat straight down: no clear order

    # 2. Arms: did a wrist move well away from the body's axis during the fall?
    out_frames = visible_frames = 0
    for sample in during:
        shoulders = joint_midpoint(sample.joints, SHOULDERS)
        hips = joint_midpoint(sample.joints, HIPS)
        wrists = [sample.joints[i] for i in (9, 10) if np.isfinite(sample.joints[i]).all()]
        if shoulders is None or hips is None or not wrists:
            continue
        axis = hips - shoulders
        length = float(np.hypot(*axis))
        if length < MIN_TORSO_PX:
            continue
        visible_frames += 1
        # distance of each wrist from the infinite line through shoulders and hips
        away = max(_distance_to_segment(w, shoulders, hips) for w in wrists) / scale
        out_frames += away >= ARM_OUT_TORSOS
    if visible_frames >= max(2, len(during) // 2):
        result["arms"] = out_frames >= ARM_OUT_FRAMES

    # 3. Direction (profile views only): did the head end up on the side the person faced?
    facing = []
    for sample in before[-4:]:
        nose, shoulders = sample.joints[0], joint_midpoint(sample.joints, SHOULDERS)
        if np.isfinite(nose).all() and shoulders is not None:
            facing.append((float(nose[0]) - float(shoulders[0])) / scale)
    head_end = _part_track(after, PARTS["upper"])
    hip_start = _part_track(before[-4:], HIPS)
    if facing and head_end and hip_start and abs(float(np.median(facing))) >= FACING_MIN:
        side = float(np.median([x for _, x, _ in head_end])) - float(np.median([x for _, x, _ in hip_start]))
        if abs(side) / scale >= DIRECTION_MIN:
            result["direction"] = "forward" if np.sign(side) == np.sign(np.median(facing)) else "backward"

    # 4. Pattern from the three observations.
    first, arms, direction = result["first_contact"], result["arms"], result["direction"]
    if arms is False and first in ("knees", "upper"):
        result["pattern"] = "collapse"      # buckled or toppled with no protective reaction
    elif first == "hands" or (first == "knees" and arms) or (arms and direction == "forward"):
        result["pattern"] = "trip"
    elif first == "hips" or direction == "backward":
        result["pattern"] = "slip"
    elif arms is False and first is None and set(first_parts) == {"hips", "knees"}:
        result["pattern"] = "collapse"      # dropped straight down, arms limp
    return result


# People around a fallen person (phase 2) ---------------------------------------
# Distances are between the people's floor points (bottom centre of each box),
# in the fallen person's torso lengths. A standing adult is about 3 torso lengths
# tall, so 6 torso lengths is roughly two body lengths.
NEARBY_TORSOS = 6.0        # closer than this = "nearby"
WITH_THEM_TORSOS = 2.5     # closer than this (about arm's reach + kneeling) ...
WITH_THEM_S = 2.0          # ...for this long = "with them" (someone is helping)
APPROACH_TORSOS = 1.0      # got this much closer ...
APPROACH_S = 2.0           # ...within this many seconds = "coming over"
NO_RESPONSE_S = 20.0       # people nearby this long and nobody with them = "not responded"
DEPTH_RATIO = 2.0          # torso sizes differing more than this = different depth, not near
BYSTANDER_FORGET_S = 3.0   # a bystander unseen this long is dropped from the list
REKEY_S = 1.5              # a new ID appearing this soon after a bystander vanished ...
REKEY_TORSOS = 1.5         # ...about this close to where they were is the same person


def movement_score(history: PoseHistory, span: float = MOVE_SPAN_S) -> float | None:
    """How far the person moved in the last `span` seconds, in torso lengths.

    Compares the average pose of the last MOVE_AVERAGE_S with the average pose
    `span` seconds earlier, so keypoint wobble on a still person averages out.
    Uses the second-largest joint movement: one jittery keypoint is ignored,
    while a moving arm (wrist and elbow) or a turning head still counts.
    None when there is not enough to judge (too few frames or joints, a gap)."""
    samples = history.samples
    if len(samples) < 2:
        return None
    now = samples[-1].t
    recent = [s for s in samples if now - s.t <= MOVE_AVERAGE_S]
    before = [s for s in samples if span <= now - s.t <= span + MOVE_AVERAGE_S]
    if not before:
        return None
    # a gap: the newest "before" frame must be close to `span` ago
    if min(now - s.t for s in before) > span * 1.5:
        return None
    scale = history.body_scale()
    if not scale:
        return None
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN joints are expected
        pose_now = np.nanmean(np.stack([s.joints for s in recent]), axis=0)
        pose_before = np.nanmean(np.stack([s.joints for s in before]), axis=0)
    moved = np.hypot(*(pose_now - pose_before).T) / scale  # NaN where either is missing
    moved = moved[np.isfinite(moved)]
    if len(moved) < MOVE_MIN_JOINTS:
        return None
    return float(np.sort(moved)[-2])


class PostFallWatch:
    """What one person does after their fall is confirmed.

    Still time only grows on frames where the person is actually seen and can
    be judged, so being hidden behind furniture never counts as lying still."""
    __slots__ = ("confirmed_at", "last_t", "still_s", "longest_still_s", "moved", "last_move_t",
                 "critical_at", "moved_after_critical", "standing_since", "recovered_at",
                 "fall_hip_y", "scale", "above", "bystanders", "helpers", "helped_at",
                 "nearby_since", "ever_nearby", "descent", "type_final", "pattern",
                 "scale_reliable", "bystander_debug", "announced")

    def __init__(self, t: float, history: PoseHistory):
        self.confirmed_at = self.last_t = t
        self.still_s = self.longest_still_s = 0.0
        self.moved = False
        self.last_move_t = None
        self.critical_at = None           # when still time first reached STILL_CRITICAL_S
        self.moved_after_critical = False
        self.standing_since = None
        self.recovered_at = None
        latest = history.latest
        hips = joint_midpoint(latest.joints, HIPS) if latest is not None else None
        self.fall_hip_y = float(hips[1]) if hips is not None else None  # pixels, at confirmation
        self.scale = history.body_scale()
        # A fallen person never seen standing is measured lying down; lying towards
        # the camera, their torso looks short and is a poor ruler for distances.
        self.scale_reliable = history.scale_source() == "standing"
        self.bystander_debug: dict[int, str] = {}  # person ID -> why near / not near (console)
        self.announced = False                     # analysis summary printed once
        self.above = 0  # consecutive frames over MOVE_THRESHOLD
        self.bystanders: dict[int, dict] = {}  # person ID -> first/current distance, times
        self.helpers: list[int] = []           # who has been with them, in order of arrival
        self.helped_at = None                  # when the first helper reached them
        self.nearby_since = None               # when someone was first nearby
        self.ever_nearby: set[int] = set()
        self.descent = descent_analysis(history)  # sudden / gradual, measured at confirmation
        self.pattern = fall_pattern(history, self.descent)  # collapse / slip / trip (sudden falls)
        self.type_final = False                   # re-measured for TYPE_SETTLE_S, then fixed

    def update(self, t: float, history: PoseHistory, seen: bool) -> None:
        dt = min(max(t - self.last_t, 0.0), MOVE_MAX_GAP_S)
        self.last_t = t
        if not self.type_final:
            # The hips may still be settling when the fall is confirmed: re-measure briefly.
            if seen:
                self.descent = descent_analysis(history)
                self.pattern = fall_pattern(history, self.descent)
            self.type_final = t - self.confirmed_at >= TYPE_SETTLE_S
        latest = history.latest
        if not seen or latest is None or latest.t != t or self.recovered_at is not None:
            return
        if latest.state == "standing" and self._hips_risen(latest):
            self.standing_since = self.standing_since if self.standing_since is not None else t
            if t - self.standing_since >= RECOVER_S:
                self.recovered_at = t
                return
        else:
            self.standing_since = None
        if t - self.confirmed_at < LANDING_S:
            return  # still landing: settling on the floor is not "moving"
        score = movement_score(history)
        if score is None:
            return
        self.above = self.above + 1 if score >= MOVE_THRESHOLD else 0
        if self.above >= MOVE_CONFIRM_FRAMES:
            self.moved = True
            self.last_move_t = t
            if self.critical_at is not None:
                self.moved_after_critical = True
            self.still_s = 0.0
        elif self.above == 0:
            self.still_s += dt
            self.longest_still_s = max(self.longest_still_s, self.still_s)
            if self.still_s >= STILL_CRITICAL_S and self.critical_at is None:
                self.critical_at = t

    def update_bystanders(self, t: float, fallen_box, others) -> None:
        """`others`: (person ID, box, torso px or None, posture state) for every
        other person seen in this frame. A new bystander must be standing when
        first seen (so someone else who fell is not a helper); after that they
        count whatever their posture, because a helper kneels or crouches."""
        if self.recovered_at is not None:
            return
        fx, fy = (fallen_box[0] + fallen_box[2]) / 2, fallen_box[3]
        scale = self.scale
        if not scale:
            return
        seen_now = {o[0] for o in others}
        for pid, box, torso, state in others:
            known = self.bystanders.get(pid)
            if self.scale_reliable:
                unit = scale
                if torso and max(torso, scale) / min(torso, scale) > DEPTH_RATIO:
                    self.bystander_debug[pid] = (f"skipped: body size {max(torso, scale) / min(torso, scale):.1f}x "
                                                 "different (different distance from the camera)")
                    continue  # much nearer to or farther from the camera: not really close
            else:
                # The fallen person was never seen standing: measure in the other
                # person's own torso lengths instead (their size is reliable when
                # they are standing), and skip the depth check, which needs both.
                unit = torso or scale
            bx, by = (box[0] + box[2]) / 2, box[3]
            distance = float(np.hypot(bx - fx, by - fy)) / unit
            self.bystander_debug[pid] = (f"{distance:.1f} torso lengths away"
                                         + (" (nearby)" if distance <= NEARBY_TORSOS else " (not nearby)"))
            if known is None and state != "standing":
                # Kneeling down can change a helper's box so much that the tracker
                # gives them a new ID: take over a bystander who just vanished here.
                lost = [old for old, b in self.bystanders.items() if old not in seen_now
                        and t - b["last_t"] <= REKEY_S and abs(b["d"] - distance) <= REKEY_TORSOS]
                if not lost:
                    continue  # never seen standing: someone else who fell is not a helper
                old = min(lost, key=lambda o: abs(self.bystanders[o]["d"] - distance))
                known = self.bystanders[pid] = self.bystanders.pop(old)
                self.helpers = [pid if h == old else h for h in self.helpers]
                if old in self.ever_nearby:
                    self.ever_nearby.discard(old)
                    self.ever_nearby.add(pid)
            if known is None:
                known = self.bystanders[pid] = {"trail": deque(), "close_since": None}
            known["d"], known["last_t"] = distance, t
            trail = known["trail"]  # (time, distance) over the last APPROACH_S + 1 s
            trail.append((t, distance))
            while trail and t - trail[0][0] > APPROACH_S + 1.0:
                trail.popleft()
            if distance <= NEARBY_TORSOS:
                self.ever_nearby.add(pid)
                if self.nearby_since is None:
                    self.nearby_since = t
            if distance <= WITH_THEM_TORSOS:
                known["close_since"] = known["close_since"] if known["close_since"] is not None else t
                if t - known["close_since"] >= WITH_THEM_S and pid not in self.helpers:
                    self.helpers.append(pid)
                    if self.helped_at is None:
                        self.helped_at = t
            else:
                known["close_since"] = None
        for pid in [pid for pid, b in self.bystanders.items() if t - b["last_t"] > BYSTANDER_FORGET_S]:
            del self.bystanders[pid]

    @staticmethod
    def _closing_in(bystander: dict, t: float) -> bool:
        """Distance dropped by APPROACH_TORSOS within the last APPROACH_S seconds."""
        recent = [d for when, d in bystander["trail"] if t - when <= APPROACH_S]
        return bool(recent) and max(recent) - bystander["d"] >= APPROACH_TORSOS

    def helper_ids(self, t: float) -> set[int]:
        """Bystanders (seen standing when they came) now near this person. While
        their torso stays upright, their kneeling or crouching is not counted as
        a fall of their own (see PersonTracker.update)."""
        if self.recovered_at is not None:
            return set()
        return {pid for pid, b in self._current(t).items() if b["d"] <= NEARBY_TORSOS}

    def _current(self, t: float) -> dict[int, dict]:
        return {pid: b for pid, b in self.bystanders.items() if t - b["last_t"] <= 0.5}

    def _hips_risen(self, sample: PoseSample) -> bool:
        """Standing posture AND hips clearly higher than at the fall, so a person
        lying towards the camera (which can look upright) is not called 'got up'."""
        hips = joint_midpoint(sample.joints, HIPS)
        if hips is None or self.fall_hip_y is None or not self.scale:
            return True  # nothing to compare: trust the posture rule
        return (self.fall_hip_y - float(hips[1])) / self.scale >= RECOVER_RISE

    def status(self, t: float) -> str:
        """'recovered', 'still' (critical), 'moving', or 'watching' (not judged yet)."""
        if self.recovered_at is not None:
            return "recovered"
        if self.still_s >= STILL_CRITICAL_S:
            return "still"
        if self.last_move_t is not None and t - self.last_move_t <= MOVING_RECENT_S:
            return "moving"
        return "watching"

    def report(self, t: float) -> dict:
        """Plain values for the GUI thread (sent with each frame)."""
        report = {
            "status": self.status(t),
            "still_s": round(self.still_s, 1),
            "longest_still_s": round(self.longest_still_s, 1),
            "since_fall_s": round(t - self.confirmed_at, 1),
            "moved": self.moved,
            "critical": self.critical_at is not None,
            "moved_after_critical": self.moved_after_critical,
            "down_s": round((self.recovered_at or t) - self.confirmed_at, 1),
            "fall_type": self.descent["type"],
            "descent_s": self.descent["descent_s"],
            "borderline": self.descent["borderline"],
            "type_note": self.descent["reason"],
            "type_final": self.type_final,
            **self.pattern,
            **self.bystander_report(t),
        }
        report["priority"] = priority_of(report)
        return report

    def bystander_report(self, t: float) -> dict:
        current = self._current(t)
        nearby = sorted(pid for pid, b in current.items() if b["d"] <= NEARBY_TORSOS)
        with_them = sorted(pid for pid, b in current.items()
                           if b["close_since"] is not None and t - b["close_since"] >= WITH_THEM_S)
        approaching = sorted(pid for pid, b in current.items() if pid not in with_them
                             and self._closing_in(b, t) and b["d"] <= NEARBY_TORSOS * 1.5)
        no_response = (self.helped_at is None and self.nearby_since is not None
                       and t - self.nearby_since >= NO_RESPONSE_S and not approaching)
        return {
            "nearby": nearby,              # other people within ~2 body lengths now
            "with_them": with_them,        # within reach for 2 s or more now
            "approaching": approaching,    # coming closer, not with them yet
            "helpers": list(self.helpers), # everyone who has been with them
            "help_after_s": round(self.helped_at - self.confirmed_at, 1) if self.helped_at is not None else None,
            "ever_nearby": sorted(self.ever_nearby),
            "no_response": no_response,    # people nearby for 20 s, nobody came
        }

    def expired(self, t: float) -> bool:
        return self.recovered_at is not None and t - self.recovered_at > WATCH_KEEP_S


def watch_label(report: dict | None) -> str:
    """Short text added to a person's label on the video."""
    if not report:
        return ""
    status = report["status"]
    kind = report.get("fall_type")
    prefix = f" ({kind})" if kind in ("sudden", "gradual") else ""
    if kind in ("sudden", "gradual") and report.get("pattern") == "unclear" and report.get("type_final"):
        prefix = f" ({kind}, pattern unclear)"
    if report.get("pattern") in PATTERN_TEXT:
        prefix = f" ({VIDEO_LABEL.get(report['pattern'], report['pattern'])})"
    elif kind == "unclear" and report.get("type_final"):
        prefix = f" (type unclear: {SHORT_REASON.get(report.get('type_note', ''), 'see console')})"
    return prefix + _status_label(report, status)


# Why a fall type is unclear, in a few ASCII words for the video label.
SHORT_REASON = {
    "too few frames with the hips visible": "hips not visible",
    "body size unknown": "body not visible",
    "not seen standing before the fall": "not seen standing first",
    "the hips hardly dropped (sitting, or already low)": "hips hardly dropped",
    "the start of the fall is not in view": "start not in view",
    "too few frames during the fall (low frame rate)": "frame rate too low",
}


def watch_summary(person_id: int, watch: "PostFallWatch") -> str:
    """One console line explaining what the analysis decided for a fallen person."""
    report = watch.report(watch.last_t)
    kind = report["fall_type"]
    how = (f"{kind} ({report['descent_s']:.2f} s)" if report.get("descent_s") is not None
           else f"unclear ({report.get('type_note') or 'no reason'})")
    pattern = report.get("pattern")
    if pattern:
        signs = pattern_detail(report)
        how += f", pattern {pattern}" + (f" [{signs}]" if signs else "")
    scale = f"{watch.scale:.0f} px" if watch.scale else "unknown"
    ruler = "standing frames" if watch.scale_reliable else "not seen standing: other people's own size"
    others = "; ".join(f"#{pid} {text}" for pid, text in sorted(watch.bystander_debug.items())) or "nobody else seen"
    return (f"[analysis] #{person_id}: {how}; priority {report['priority']}; "
            f"body scale {scale} ({ruler}); others: {others}")


def _status_label(report: dict, status: str) -> str:
    if status == "still":
        return f" - NO MOVEMENT {report['still_s']:.0f} s"
    if status == "moving":
        return " - moving"
    if status == "recovered":
        return " - got up"
    if report.get("with_them"):
        return " - helper with them"
    return ""


def bystander_label(fallen_reports: dict, person_id: int) -> str:
    """Label suffix for a person who is with, or coming to, someone who fell."""
    for fallen_id, report in fallen_reports.items():
        if person_id in report.get("with_them", ()):
            return f" - with #{fallen_id}"
        if person_id in report.get("approaching", ()):
            return f" - going to #{fallen_id}"
    for fallen_id, report in fallen_reports.items():
        if person_id in report.get("nearby", ()):
            return f" - near #{fallen_id}"
    return ""

