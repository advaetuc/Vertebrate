# P2A verification

2026-10-02. Source-time smoothing and kinematics implemented per blueprint section 7. Existing 313 tests preserved. New deterministic fixtures test arithmetic and state/history policies; they are not detection-accuracy tests. No network access or package changes were required.

API: one FeatureExtractor per person, update(TrackSample, candidate_active=False, in_zone=True). P2B must latch candidate_active after entry to freeze the baseline and invalidate missing observations explicitly. frame_height scales initial size eligibility. Outputs are frozen raw/filtered summaries; undefined evidence is None. Reset/generation changes clear retained history.

Baseline check before implementation: `python -m pytest -q`, exit 0, `313 passed in 25.09s`. During development the first 80 feature tests passed; adding boundary tests exposed an incorrect test expectation for nearest-target pairing (1 failed, 89 passed). The fixture was corrected so its intermediate sample is too recent to be eligible; final results follow.

| Command | Exit | Result |
|---|---:|---|
| `.\.venv\Scripts\python.exe -m pip check` | 0 | No broken requirements found. |
| `.\.venv\Scripts\python.exe -m pytest tests/unit/test_features.py -q` | 0 | 90 passed in 0.27s |
| `.\.venv\Scripts\python.exe -m pytest -q` | 0 | 403 passed in 13.56s |
| `.\.venv\Scripts\python.exe -m vertebrate doctor --offline` | 0 | All doctor checks pass; zero network/process attempts. |

Files: src/vertebrate/smoothing.py, src/vertebrate/features.py, tests/unit/test_features.py, reports/p2a.md.

Ready for P2B.
