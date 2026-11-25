"""Fall analysis on simple synthetic stick figures (no camera, no window).

Run from the project folder:
    Windows  .venv\\Scripts\\python.exe -m unittest discover -s tests -v
    macOS    .venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_modules import fallanalysis as fa  # noqa: E402
from app_modules.common import posture_state  # noqa: E402
from app_modules.tracking import PersonTracker  # noqa: E402

FPS = 15
W, H = 1280, 720


def standing(cx, floor_y, s):
    """17 COCO keypoints (pixels) and box for a person standing; s = torso length in px."""
    hip_y = floor_y - 1.8 * s
    k = np.zeros((17, 2), np.float32)
    k[0] = (cx + 0.15 * s, hip_y - 1.45 * s)
    k[5], k[6] = (cx - 0.05 * s, hip_y - s), (cx + 0.05 * s, hip_y - s)
    k[7], k[8] = (cx, hip_y - 0.5 * s), (cx + 0.05 * s, hip_y - 0.5 * s)
    k[9], k[10] = (cx + 0.05 * s, hip_y), (cx + 0.1 * s, hip_y)
    k[11], k[12] = (cx - 0.03 * s, hip_y), (cx + 0.03 * s, hip_y)
    k[13], k[14] = (cx, hip_y + 0.9 * s), (cx + 0.03 * s, hip_y + 0.9 * s)
    k[15], k[16] = (cx, floor_y), (cx + 0.03 * s, floor_y)
    return k


def lying(cx, floor_y, s, along_view=1.0):
    """Lying on the floor, head to the left. along_view < 1 shortens the body on
    screen, as when the person lies stretched out towards the camera."""
    y, f = floor_y - 0.25 * s, along_view
    k = np.zeros((17, 2), np.float32)
    k[0] = (cx - 2.0 * s * f, y)
    k[5], k[6] = (cx - 1.5 * s * f, y - 0.05 * s), (cx - 1.5 * s * f, y + 0.05 * s)
    k[7], k[8] = (cx - 1.0 * s * f, y - 0.1 * s), (cx - 1.0 * s * f, y + 0.1 * s)
    k[9], k[10] = (cx - 0.5 * s * f, y - 0.1 * s), (cx - 0.5 * s * f, y + 0.1 * s)
    k[11], k[12] = (cx, y - 0.05 * s), (cx, y + 0.05 * s)
    k[13], k[14] = (cx + 0.9 * s * f, y - 0.05 * s), (cx + 0.9 * s * f, y + 0.05 * s)
    k[15], k[16] = (cx + 1.8 * s * f, y), (cx + 1.8 * s * f, y + 0.05 * s)
    return k


def box_of(k, s):
    seen = k[(k[:, 0] > 0) & (k[:, 1] > 0)]  # eyes and ears are left at 0, 0 (not detected)
    lo, hi = seen.min(0), seen.max(0)
    return np.array([lo[0] - 0.25 * s, lo[1] - 0.25 * s, hi[0] + 0.15 * s, hi[1] + 0.1 * s], np.float32)


def run(frames):
    """frames: list of [keypoints per person]. Returns the tracker after feeding them."""
    rng = np.random.default_rng(0)
    tracker = PersonTracker()
    for i, people in enumerate(frames):
        # A little keypoint wobble, as from the pose model; joints that were not
        # detected stay at exactly 0, 0, as YOLO reports them.
        kps = [np.where(k > 0, k + rng.normal(0, 1.0, k.shape), 0).astype(np.float32) for k in people]
        boxes = [box_of(k, 90) for k in kps]
        states = [posture_state(b, k, W, H) for b, k in zip(boxes, kps)]
        tracker.update(boxes, [s == "fallen" for s in states], kps, states, 1000 + i / FPS)
    return tracker


class FallAnalysisTest(unittest.TestCase):
    def test_already_lying_with_someone_standing_nearby(self):
        """The camera test that failed: one person lying, one standing ~3 torso lengths away."""
        tracker = run([[lying(400, 650, 90), standing(670, 650, 90)] for _ in range(6 * FPS)])
        report = tracker.aftermath()[1]
        self.assertEqual(report["nearby"], [2])
        self.assertEqual(report["fall_type"], "unclear")       # how they fell was not seen...
        self.assertIn("type unclear: not seen standing first", fa.watch_label(report))  # ...and the label says so
        self.assertEqual(fa.bystander_label(tracker.aftermath(), 2), " - near #1")

    def test_lying_towards_the_camera_still_finds_the_bystander(self):
        tracker = run([[lying(400, 650, 90, along_view=0.3), standing(670, 650, 90)] for _ in range(6 * FPS)])
        self.assertEqual(tracker.aftermath()[1]["nearby"], [2])

    def test_sudden_fall_is_timed(self):
        frames = []
        for i in range(7 * FPS):
            t = i / FPS
            f = min(max((t - 2) / 0.5, 0), 1) ** 2
            frames.append([standing(400, 650, 90) * (1 - f) + lying(400, 650, 90) * f])
        report = run(frames).aftermath()[1]
        self.assertEqual(report["fall_type"], "sudden")
        self.assertLess(abs(report["descent_s"] - 0.5), 0.2)


if __name__ == "__main__":
    unittest.main()
