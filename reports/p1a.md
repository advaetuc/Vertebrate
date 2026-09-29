# VERTEBRATE Phase P1A report

**Acceptance: PASS.** Verification completed 2026-09-28 at 17:20:24 UTC; report
finalized on the resumed session, 2026-09-29 local date. No network access,
dependency installation, hardware camera testing, or accuracy/performance
benchmark was performed for P1A.

## 1. Inspection and preservation

Before modifying files, inspected the repository listing and Git state;
`reports/baseline.md`; `src/vertebrate/contracts.py`, `config.py`, `cli.py`,
`offline.py`; package metadata; ignore rules; and applicable AGENTS.md locations
through the ancestor chain. No applicable AGENTS.md was present. The existing
repository was clean at `3df23dc` (`Initial commit`), with the P0 virtual
environment, cached local model, configs, tests and reports present. Unlike the
original P0 inspection, a Git repository and `.gitattributes` now existed.

The initial P0 regression command exited 0 with **174 passed in 2.52s**. The
original P0 classes and exception bases were preserved; only new types/helpers
were added to contracts.py. NumPy is imported lazily in the new array validation
helper, preserving the doctor's asset-before-dependency-import ordering.

The following P0 files remain unchanged: both existing test files, config.py,
cli.py, offline.py, canonical configs, model weights/manifest, pyproject.toml,
the dependency lock, and baseline.md. No dependency pins changed. Git diff
confirmed preservation of the P0 tests, configuration, CLI, guard and metadata.

All Python commands used
`C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe`.
The existing Windows 11 / Python 3.12.0 x64 CPU environment remains in use.

## 2. Files created or modified

| File | Change | Purpose |
|---|---|---|
| `src/vertebrate/contracts.py` | Extended | P1 errors/enums, immutable frame/pose/track records, identity and lifecycle events; all P0 contracts retained |
| `src/vertebrate/validation.py` | New | Shared strict scalar, identity and UTC validation |
| `src/vertebrate/clocks.py` | New | Injectable live monotonic clock, CFR/PTS/fallback clock, pause and historical UTC policies |
| `src/vertebrate/capture.py` | New | Bounded queue, reader protocol/OpenCV adapter, synchronous controller and owner-thread worker |
| `src/vertebrate/evaluation/__init__.py` | New | Evaluation package namespace |
| `src/vertebrate/evaluation/labels.py` | New | Frozen manifest/label models, strict JSON parsing, interval/path/hash checks and save/load helpers |
| `tools/generate_dev_fixtures.py` | New | Deterministic OpenCV AVI/MJPG generation and full decoded timing verification |
| `tools/verify_p1a.py` | New | Runs acceptance commands and retains verbatim output and exit codes |
| `data/manifests/dev.json` | New | Three actual local clips, SHA-256 values, grouping metadata and source-time annotations |
| `tests/fixtures/videos/dev_empty_scene.avi` | Generated | Empty scene; 227,286 bytes |
| `tests/fixtures/videos/dev_single_person_fall_sim.avi` | Generated | Synthetic falling figure; 475,878 bytes |
| `tests/fixtures/videos/dev_two_person_adl_sim.avi` | Generated | Two synthetic moving figures; 795,088 bytes |
| `tests/unit/test_clocks.py` | New | 108 clock, contract, queue, lifecycle, manifest and real-file tests |
| `README.md` | Extended | P1A APIs, usage, timing/queue decisions, limitations and next-phase integration |
| `reports/p1a-verification-20260928T172024Z.json` | New | Exact final acceptance commands/stdout/stderr/exit codes |
| `reports/p1a.md` | New | This structured report |

The fixture clips are deliverables in a non-ignored source directory, ready to
include with the new source files in version control. `git check-ignore` found
none ignored. This task did not stage or commit changes. The generator can also
recreate clips and their matching manifest locally; tests do not silently skip
missing assets or overwrite the project fixtures.

### Measured fixture integrity

All clips are **640x360, 25 FPS, 100 frames, 4.0 seconds**, AVI/MJPG. Actual
decoded timestamps used PTS_VERIFIED for all frames. Last PTS: 3.96 seconds.

| Clip | SHA-256 |
|---|---|
| dev_empty_scene.avi | `6e3883f1b1c3a59dc67df38c87d452179860e72872ec77d1c60d42b8294ed64c` |
| dev_single_person_fall_sim.avi | `b5e43fd1fb21cdfcca39faf14520700ac70464b75eb9b248a279a6e4bebca1ba` |
| dev_two_person_adl_sim.avi | `e6f285f043c7851e30c12faf53b68789a30703892eb77621d354ee08f331ac53` |

The test regenerates all three clips in a temporary directory and compares
their actual hashes against these delivered fixtures. Byte-for-byte equality
passed on this pinned environment; cross-platform codec determinism is not
claimed. `verify_files=True` checks every delivered clip's actual SHA-256.

## 3. Commands and tests actually run

The final full transcript is linked below and reproduced in section 6. These
earlier runs also occurred, before the final acceptance run:

| Exact command from project root | Exit | Verbatim result / output disposition |
|---|---:|---|
| `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q` | 0 | `174 passed in 2.52s` (summary line; full progress below) |
| `.\.venv\Scripts\python.exe tools/generate_dev_fixtures.py` | 0 | Printed the same three fixture records reproduced in section 6; initial docstring emitted a SyntaxWarning, subsequently fixed |
| `.\.venv\Scripts\python.exe -m pytest tests/unit/test_clocks.py -q` | 0 | `102 passed in 5.54s` before six additional boundary tests were added |
| `.\.venv\Scripts\python.exe tools/verify_p1a.py` | 0 | Ran all five child commands in section 6, each with exit 0 |

Initial P0 stdout:

```text
........................................................................ [ 41%]
........................................................................ [ 82%]
..............................                                           [100%]
174 passed in 2.52s
```

Initial P1A stdout:

```text
........................................................................ [ 70%]
..............................                                           [100%]
102 passed in 5.54s
```

Those two pytest runs had no stderr. The initial generator warning was:

```text
C:\Users\Advaet\Documents\Projects\Vertebrate\tools\generate_dev_fixtures.py:1: SyntaxWarning: invalid escape sequence '\.'
  """Generate deterministic local synthetic clips; these are NOT accuracy data.
```

Changing that docstring to a raw string removed the warning. The final generator
stderr is empty. Git also emitted its normal LF-to-CRLF working-copy notice for
contracts.py under the existing `.gitattributes` policy. No test failed or was
skipped during P1A. One README patch was rejected for unmatched context and made
no changes; the subsequent documentation patch applied successfully.

## 4. Acceptance gates

| Requirement | Status | Executed evidence |
|---|---|---|
| All original P0 tests/contracts preserved | PASS | 174 original tests unchanged; all included in 282-test regression |
| FramePacket / PoseObservation / TrackSample immutability | PASS | Frozen dataclasses; copied immutable bytes-backed arrays; writable-copy isolation and validation tests |
| Session/person generation keys and predicted-only validity | PASS | Identity range tests; invalid predicted/absent matches rejected |
| Live monotonic elapsed clock and UTC estimate | PASS | Injected time progression; backwards-clock rejection; UTC unaffected by wall-clock adjustment |
| CFR PTS and explicit zero/missing-PTS fallback | PASS | Clock and capture tests, including fallback latching and positive-PTS validation after fallback |
| Invalid FPS/PTS, nonmonotonic timing and VFR | PASS | Zero/missing/negative/NaN/Inf FPS, negative/nonfinite/backward/frozen PTS, jitter and cumulative drift rejected with ffmpeg hint |
| Twenty-second pause contributes zero source time | PASS | Injected monotonic time advances 20 seconds while clock/source reads remain frozen; resumed next frame is 0.04s |
| Capacity-2 LIVE drop-oldest | PASS | Ten frames leave sequences 8 and 9; dropped_frames_count=8 |
| Capacity-2 VIDEO lossless backpressure | PASS | Concurrent producer/consumer delivers all sequences 0..9, zero drops; pending retry does not reread/tick |
| Lifecycle events preserved; stale producers fenced | PASS | Control saturation has explicit finite timeout; atomic reset clears old frames and rejects blocked old-session producers |
| Seek/replay/switch/restart resets | PASS | New IDs, sequence/time reset, reset event before new frames; camera switch through injected reader |
| Reader ownership and cooperative shutdown | PASS | Open/read/seek/release all on owner thread; pause and stop acknowledged; blocked producer cancellation tested |
| EOF once, no synthetic final-frame extension | PASS | Fake and real files emit one terminal event, release source and keep last real source time; no subsequent frames |
| Historical imported-video UTC | PASS | None remains None; explicit recorded UTC plus seek offset verified |
| Manifest parsing and immutable labels | PASS | Strict schema/required fields, interval order/duration, numeric types, SHA-256 and local paths |
| Three actual deterministic local CFR fixtures | PASS | Real generation/decode of 300 frames; regeneration hashes match; manifest verify_files succeeds |
| `python -m pip check` | PASS | Exit 0; `No broken requirements found.` |
| `python -m pytest tests/unit/test_clocks.py -q` | PASS | Exit 0; **108 passed in 5.58s**, zero skips |
| Required combined P0/P1A pytest command | PASS | Exit 0; **282 passed in 8.10s**, zero skips |
| `python -m vertebrate doctor --offline` | PASS | Exit 0; all checks pass, `network_attempts: []` |

EOF/reset is the explicit lifecycle boundary on which a future FSM must close
incomplete evidence as inconclusive. P1A emits this boundary and does not own or
claim to implement a fall-detection FSM. The delivered producer never repeats
the last frame or adds terminal dwell time.

## 5. Decisions, limitations and P1B handoff

- No unresolved P1A acceptance blocker. No package or P0 schema change required.
- Actual OpenCV Windows AVI/MJPG decode provided correct 0.04-second PTS spacing;
  the zero-PTS quirk did not occur on these fixtures. Its handling is verified
  with injected timestamp readers, not claimed as an observed codec failure.
- Files undergo one full decoded timing/count verification pass before capture
  emission, keeping only the current frame. This favors strict validation over
  startup latency; no startup/performance benchmark is claimed.
- FPS alone cannot prove CFR for arbitrary opaque containers. Missing PTS is
  permitted for fixed-rate AVI, or when the caller supplies independently
  verified FPS through the reader adapter. Other unverified timing is rejected.
- Seek establishes a new session-relative time origin. Frame packets retain
  source_frame_index and source_time_offset_s so original labels and explicit
  recording UTC still align. Sequence resets to zero; pause/resume does not reset.
- Capacity includes lifecycle controls. A queue containing only controls must
  apply finite backpressure rather than lose them or grow without bound. Resets
  fail explicitly if controls alone fill capacity; drain and retry. A producer
  may retain one pending frame outside the two queue slots.
- A native camera read stuck inside a backend cannot safely be released from
  another thread. Worker stop is cooperative and reports a timeout; no physical
  camera/backend shutdown performance was tested.
- All synthetic clips are marked ambiguity_stress_test, and recorded_start_utc
  is None. Their drawings and labels test pipeline contracts; they do not prove
  human pose quality, fall accuracy or performance. The falling drawing's clip
  is too short to establish every future stillness decision threshold.
- The existing benchmark/evaluate/GUI CLI stubs remain unchanged. P1B can consume
  FramePacket, emit PoseObservation in original-image coordinates, build
  TrackSample with generation-aware PersonTrackKey, and run dev-manifest clips
  through the lossless capture path. Clear tracker/evidence state on reset and
  terminate incomplete observations at EOF. `pose.py`, `tracking.py`, and the
  actual benchmark remain to be implemented in P1B; the P1A interfaces and
  offline fixtures are ready for them.

## 6. Final acceptance transcript

Source: [p1a-verification-20260928T172024Z.json](p1a-verification-20260928T172024Z.json).
The exact commands, exit codes and verbatim stdout/stderr follow.

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe tools/generate_dev_fixtures.py

Exit code: 0

stdout:

```text
[
  {
    "clip_id": "dev_empty_scene",
    "sha256": "6e3883f1b1c3a59dc67df38c87d452179860e72872ec77d1c60d42b8294ed64c",
    "size_bytes": 227286,
    "fps": 25.0,
    "frame_count": 100,
    "duration_s": 4.0,
    "timing_modes": [
      "pts_verified"
    ],
    "codec": "MJPG",
    "dimensions": [
      640,
      360
    ]
  },
  {
    "clip_id": "dev_single_person_fall_sim",
    "sha256": "b5e43fd1fb21cdfcca39faf14520700ac70464b75eb9b248a279a6e4bebca1ba",
    "size_bytes": 475878,
    "fps": 25.0,
    "frame_count": 100,
    "duration_s": 4.0,
    "timing_modes": [
      "pts_verified"
    ],
    "codec": "MJPG",
    "dimensions": [
      640,
      360
    ]
  },
  {
    "clip_id": "dev_two_person_adl_sim",
    "sha256": "e6f285f043c7851e30c12faf53b68789a30703892eb77621d354ee08f331ac53",
    "size_bytes": 795088,
    "fps": 25.0,
    "frame_count": 100,
    "duration_s": 4.0,
    "timing_modes": [
      "pts_verified"
    ],
    "codec": "MJPG",
    "dimensions": [
      640,
      360
    ]
  }
]
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pip check

Exit code: 0

stdout:

```text
No broken requirements found.
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pytest tests/unit/test_clocks.py -q

Exit code: 0

stdout:

```text
........................................................................ [ 66%]
....................................                                     [100%]
108 passed in 5.58s
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py tests/unit/test_clocks.py -q

Exit code: 0

stdout:

```text
........................................................................ [ 25%]
........................................................................ [ 51%]
........................................................................ [ 76%]
..................................................................       [100%]
282 passed in 8.10s
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m vertebrate doctor --offline

Exit code: 0

stdout:

```text
{
  "status": "pass",
  "offline": true,
  "checks": [
    {
      "name": "configuration",
      "status": "pass",
      "detail": "83100099a2655ddf63395d3f306b9f448c4255305817bbfa8589f2d6848822f0"
    },
    {
      "name": "model_asset",
      "status": "pass",
      "detail": "C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\models\\yolo11n-pose.pt: 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0 (6255593 bytes)"
    },
    {
      "name": "runtime",
      "status": "pass",
      "detail": "outbox write/fsync/remove succeeded: C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\runtime\\outbox; logs: C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\runtime\\logs"
    },
    {
      "name": "tracker_config",
      "status": "pass",
      "detail": "ByteTrack YAML matches JSON tracker settings"
    },
    {
      "name": "dependencies",
      "status": "pass",
      "detail": "{\"PySide6\": \"6.9.2\", \"lap\": \"0.5.12\", \"numpy\": \"2.2.6\", \"opencv-python\": \"4.12.0.88\", \"torch\": \"2.8.0+cpu\", \"torchvision\": \"0.23.0+cpu\", \"ultralytics\": \"8.3.203\"}"
    },
    {
      "name": "inference_and_assignment",
      "status": "pass",
      "detail": "CPU pose boxes=(0, 6), keypoints=(0, 17, 3); ByteTrack two-frame assignment shape=(1, 8), stable ID=1"
    },
    {
      "name": "offline_guard",
      "status": "pass",
      "detail": "zero network/process attempts"
    }
  ],
  "network_attempts": [],
  "config": {
    "profile": "demo-3s",
    "revision": 1,
    "sha256": "83100099a2655ddf63395d3f306b9f448c4255305817bbfa8589f2d6848822f0"
  }
}
```

stderr:

(empty)
