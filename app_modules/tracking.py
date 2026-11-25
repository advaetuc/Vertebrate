"""Per-person tracking: follows each detected person so a fall is confirmed
for ONE person, not across the whole scene. Pure NumPy, same on Windows and macOS."""
from .common import *  # shared Qt imports, constants, and helpers
from .fallanalysis import PoseHistory, PostFallWatch, torso_upright

# Per-person tracking -------------------------------------------------------
# Detections are linked to people from frame to frame so a fall is confirmed for
# ONE person (five consecutive fallen frames of that person), not for the scene.
TRACK_MIN_IOU = 0.2           # box overlap that links a detection to a known person
TRACK_MAX_CENTER_SHIFT = 0.35  # or: centre moved less than this share of the box diagonal
                               # (a falling person's box changes shape, so overlap alone is weak)
TRACK_MAX_MISSES = 15         # frames a person may go undetected before their ID is dropped
ALERT_HOLD_MISSES = 3         # a confirmed person stays confirmed through this many missed frames


def box_iou(a, b) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def center_shift(a, b) -> float:
    """Distance between box centres, as a share of the larger box diagonal."""
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diagonal = max(np.hypot(a[2] - a[0], a[3] - a[1]), np.hypot(b[2] - b[0], b[3] - b[1]))
    return float(np.hypot(ax - bx, ay - by) / diagonal) if diagonal > 0 else 1.0


class Track:
    """One person followed across frames, with their own fallen-frame streak
    and the last few seconds of their poses (for fall analysis)."""
    __slots__ = ("id", "box", "streak", "misses", "history", "watch", "helping")

    def __init__(self, track_id: int, box):
        self.id = track_id
        self.box = tuple(map(float, box))
        self.streak = 0   # consecutive fallen frames of THIS person, capped at CONSECUTIVE_FRAMES
        self.misses = 0   # consecutive frames this person was not detected
        self.history = PoseHistory()  # ~5 s of this person's keypoints (fallanalysis.py)
        self.watch: PostFallWatch | None = None  # what they do after a confirmed fall
        self.helping = False  # kneeling/crouching beside someone who fell (not a fall)

    @property
    def confirmed(self) -> bool:
        return self.streak >= CONSECUTIVE_FRAMES and self.misses <= ALERT_HOLD_MISSES


class PersonTracker:
    """Lightweight per-person tracker in pure NumPy (no extra packages, same on
    Windows and macOS). Worker thread only; one instance per camera session.

    Each frame, detections are matched greedily to known people by box overlap
    (IoU), falling back to centre distance when a fall changes the box shape.
    A matched person's streak grows on a fallen frame and resets on a normal
    one. A frame where a person is not detected is not evidence either way, so
    their streak is kept; after TRACK_MAX_MISSES missed frames the ID is dropped.
    """

    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes, fallen, keypoints=None, states=None, timestamp=None) -> list[Track]:
        """Match this frame's detections; return the Track for each detection.
        With a timestamp (time.monotonic()), each matched person's keypoints and
        posture state are also added to their pose history."""
        boxes = [tuple(map(float, box)) for box in boxes]
        helpers = self.helper_ids()  # people helping someone who fell (previous frame)
        candidates = []
        for ti, track in enumerate(self.tracks):
            for di, box in enumerate(boxes):
                iou = box_iou(track.box, box)
                shift = center_shift(track.box, box)
                if iou >= TRACK_MIN_IOU or shift <= TRACK_MAX_CENTER_SHIFT:
                    candidates.append((-iou, shift, ti, di))
        candidates.sort()  # best overlap first, then nearest centre
        assigned: list[Track | None] = [None] * len(boxes)
        used_tracks = set()
        for _neg_iou, _shift, ti, di in candidates:
            if ti in used_tracks or assigned[di] is not None:
                continue
            used_tracks.add(ti)
            assigned[di] = self.tracks[ti]
        for di, box in enumerate(boxes):
            track = assigned[di]
            if track is None:
                track = Track(self._next_id, box)
                self._next_id += 1
                self.tracks.append(track)
                assigned[di] = track
            track.box = box
            track.misses = 0
            # A helper kneeling or crouching beside someone who fell is not a new
            # fall -- but only while their torso stays upright: a helper who
            # collapses and ends up lying down is counted as usual.
            joints = keypoints[di] if keypoints is not None and di < len(keypoints) else None
            track.helping = bool(fallen[di] and track.id in helpers and not track.confirmed
                                 and torso_upright(joints))
            counts = fallen[di] and not track.helping
            track.streak = min(track.streak + 1, CONSECUTIVE_FRAMES) if counts else 0
            if timestamp is not None:
                joints = keypoints[di] if keypoints is not None and di < len(keypoints) else None
                state = states[di] if states is not None else ("fallen" if fallen[di] else "standing")
                track.history.append(timestamp, box, joints, state)
        seen = {id(track) for track in assigned}
        for track in self.tracks:
            if id(track) not in seen:
                track.misses += 1
                if timestamp is not None:
                    track.history.trim(timestamp)  # a missed frame is a gap, not a sample
        self.tracks = [track for track in self.tracks if track.misses <= TRACK_MAX_MISSES]
        if timestamp is not None:
            self._update_watches(timestamp, seen)
        return assigned

    def _update_watches(self, t: float, seen: set) -> None:
        """Start watching a person when their fall is confirmed (again, if they
        had got up), and follow what they do until they get up, and who else
        is around them."""
        others_all = [(track.id, track.box, track.history.body_scale(), track.history.latest.state)
                      for track in self.tracks
                      if id(track) in seen and track.history.latest is not None and track.history.latest.t == t]
        for track in self.tracks:
            if track.confirmed and (track.watch is None or track.watch.recovered_at is not None):
                track.watch = PostFallWatch(t, track.history)
            elif track.watch is not None:
                track.watch.update(t, track.history, id(track) in seen)
                if track.watch.expired(t):
                    track.watch = None
                    continue
            else:
                continue
            # People around them: everyone else seen in this frame
            track.watch.update_bystanders(t, track.box, [o for o in others_all if o[0] != track.id])

    def helper_ids(self) -> set[int]:
        """IDs of people near someone who fell, who came over standing."""
        ids = set()
        for track in self.tracks:
            if track.watch is not None:
                ids |= track.watch.helper_ids(track.watch.last_t)
        return ids

    def aftermath(self) -> dict:
        """{person ID: report} for everyone being watched after a fall."""
        return {track.id: track.watch.report(track.watch.last_t) for track in self.tracks if track.watch}

    def confirmed_ids(self) -> list[int]:
        return sorted(track.id for track in self.tracks if track.confirmed)

    def leader(self) -> Track | None:
        """The person closest to a confirmed fall (highest streak, recently seen)."""
        present = [t for t in self.tracks if t.misses <= ALERT_HOLD_MISSES and t.streak > 0]
        return max(present, key=lambda t: (t.streak, -t.id)) if present else None
