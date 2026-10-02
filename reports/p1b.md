# VERTEBRATE Phase P1B completion report

Completed 2026-10-02. **P1B acceptance: PASS.** All commands used the existing project virtual environment. No packages, weights, human footage, or online services were downloaded for P1B.

## Inspection and preserved baseline

Inspected the supplied full P1B attachment, repository/ancestor instructions (no applicable AGENTS.md), primary blueprint sections 3, 4, 6, 7, 12, supporting validation report, baseline evidence, configuration, contracts, clocks, capture, CLI, offline guards, labels and development manifest. Also inspected the installed Ultralytics 8.3.203 ByteTrack and pose postprocessor source.

The repository was clean before P1B changes. Existing P0/P1A contracts, configuration, clocks, capture, offline.py, dataset labels, pins, weights, manifest and all 282 prior tests are unchanged. Before adding P1B modules this command exited 0:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py tests/unit/test_clocks.py -q
```

Verbatim baseline output:

```text
........................................................................ [ 25%]
........................................................................ [ 51%]
........................................................................ [ 76%]
..................................................................       [100%]
282 passed in 8.32s
```

## Files created or modified

| File | Change |
|---|---|
| src/vertebrate/pose.py | New guarded local CPU adapter, asset verification, model load/warmup timing, original-image observations and validity masks. |
| src/vertebrate/tracking.py | New ByteTrack adapter preserving original row indices, matched-only evidence, source-time generations and lifecycle fences. |
| src/vertebrate/benchmark.py | New hash-verified serial capture/pose/tracking benchmark with strict JSON, stage timing, freshness and provenance. |
| src/vertebrate/cli.py | Implements benchmark with --config and --imgsz; doctor/default/evaluate interfaces preserved. |
| tests/integration/test_capture_tracking.py | 31 new cases: actual pose/capture smoke and real association/postprocessing with controlled tensors. |
| tools/verify_p1b.py | Reproducible five-command acceptance runner retaining stdout, stderr and exit codes. |
| tools/verify_p0.py | Benchmark expected exit changed from stub code 2 to implemented code 0. |
| README.md | Current P1B APIs, usage, benchmark method and limits. |
| reports/benchmark.json | Actual measured results for all three sizes and all three existing clips. |
| reports/p1b-verification.json | Final passing acceptance command evidence. |
| reports/p1b-verification-initial.json | Preserved first run, including benchmark failure before the Windows metadata fix. |
| reports/p1b-verification-interrupted.json | Preserved incomplete retry; only two commands had recorded results. |
| reports/p1b.md | This completion report. |

Runtime settings/cache files under ignored runtime/logs and Python/pytest caches were updated by execution. No fixture clips were regenerated or modified.

## Implementation decisions and coverage

- Pose verifies size/SHA-256 before any YOLO construction, runs float32 CPU batch one, and uses Results coordinates after Ultralytics has already undone letterboxing. Inputs 416/512/640 are benchmarked; mappings at 640x360 and 1280x720 are tested using the real pinned postprocessor and controlled tensors.
- Invalid/non-finite or subpixel boxes are discarded before association. Non-finite keypoints become zero placeholders with false masks; validity requires finite coordinates, in-frame location and configured confidence. Empty observations preserve exact (0, ...) shapes.
- Pinned ByteTrack assigns detection indices after confidence filtering. A narrow init_track override retains original detection rows across both association passes. Tests reverse two-person confidence/row order and check all keypoints, confidences, masks, boxes and numeric IDs.
- Exactly 0.10 does not enter ByteTrack's low-score pass; 0.10001 does. The internal frame-based buffer remains separate from application expiry. Reappearance gaps greater than 0.75 source-seconds increment generation; gaps equal to or below it retain generation.
- Unmatched predictions emit no valid samples and cannot refresh last-match time. Seek/replay/restart/source switch reset identity state; stale old-session frames and frames after EOF are rejected. Real capture reads all 100 frames to EOF without extending 3.96s final source time.
- Guards deny Python network/download/process APIs during model work and fail even if a dependency swallows a blocked call. Existing offline.py is unchanged. Runtime benchmark and doctor report zero network/process attempts. Tests intentionally probe guard rejection locally; these probes do not reach network connections.

## Actual measured benchmark

All runs used the same verified local YOLO11n-Pose asset and three 640x360, 25 FPS, 100-frame AVI/MJPG clips. There are 300 measured frames per size and 900 total. Each size has one blank warmup frame; warmup, initialization, full-file CFR verification and EOF are outside steady frame timing. Capture, pose and tracking measurements include immutable copies and guard overhead. FPS = measured frames / sum of total frame processing seconds. No outliers were removed. This is serial file processing, not live-camera latency or a GUI benchmark.

Later size loads share process and OS caches, so only the first load includes cold process imports; no cold-disk claim is made. initialization_ms additionally covers asset verification. Frame-age statistics use the existing Windows monotonic acquisition clock and show coarse timestamp quantization; stage durations use the higher-resolution perf_counter clock. Historical recorded UTC remains null for every clip.
Observed environment:

```json
{
  "python": "3.12.0",
  "platform": "Windows 11 10.0.26200 AMD64",
  "processor": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel",
  "torch_threads": 8,
  "versions": {
    "numpy": "2.2.6",
    "opencv-python": "4.12.0.88",
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "ultralytics": "8.3.203",
    "lap": "0.5.12"
  }
}
```
| Input | Frames | Load ms | Warmup ms | Total mean ms | Median ms | p95 ms | Max ms | Processed FPS | Drops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 416 | 300 | 1680.124 | 74.382 | 26.725 | 26.042 | 30.691 | 33.284 | 37.418 | 0 |
| 512 | 300 | 48.050 | 57.245 | 30.681 | 29.931 | 34.720 | 38.225 | 32.593 | 0 |
| 640 | 300 | 97.272 | 73.290 | 37.575 | 36.609 | 43.122 | 48.584 | 26.613 | 0 |

Per-stage results (milliseconds):

| Input | Stage | Mean | Median | p95 | Max |
|---|---|---:|---:|---:|---:|
| 416 | capture | 0.480 | 0.472 | 0.550 | 0.858 |
| 416 | pose | 26.049 | 25.370 | 30.005 | 32.578 |
| 416 | tracking | 0.196 | 0.194 | 0.210 | 0.282 |
| 416 | frame_age_at_completion | 26.247 | 31.000 | 32.000 | 47.000 |
| 512 | capture | 0.520 | 0.519 | 0.598 | 0.824 |
| 512 | pose | 29.965 | 29.211 | 33.949 | 37.641 |
| 512 | tracking | 0.196 | 0.193 | 0.216 | 0.267 |
| 512 | frame_age_at_completion | 30.577 | 31.000 | 32.000 | 47.000 |
| 640 | capture | 0.517 | 0.517 | 0.585 | 0.649 |
| 640 | pose | 36.862 | 35.886 | 42.447 | 48.029 |
| 640 | tracking | 0.196 | 0.193 | 0.207 | 0.285 |
| 640 | frame_age_at_completion | 37.290 | 32.000 | 47.000 | 47.000 |

All sizes produced **0 detections, 0 matched samples, 0 tracks and 0 generations** on the synthetic drawings. Keypoint availability is 0 valid / 0 possible, represented as **null**, not 100%. Thus tracking stage timing here measures empty updates, not the cost or stability of tracking real people. Actual ByteTrack matching is separately exercised by controlled detection tests. This benchmark does not establish pose quality, fall/ADL accuracy, release-gate performance or the winning input size. The default remains 512.

## Acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| Baseline before implementation | PASS | 282 passed, exit 0. |
| pip check | PASS | No broken requirements, exit 0. |
| P1A + new integration suite | PASS | 139 passed, zero skips, exit 0. |
| Real offline benchmark | PASS | All sizes, 300 frames each, 0 drops, network_attempts=[], exit 0. |
| Full regression suite | PASS | 313 passed, zero skips, exit 0; all 282 previous tests preserved. |
| Offline doctor | PASS | Asset hash, Qt/dependencies, outbox, pose and ByteTrack smoke, zero network/process attempts, exit 0. |
| Coordinate and row alignment | PASS | Real postprocessor coordinate tests and real controlled ByteTrack association. |
| Expiry, resets, no predicted evidence and EOF | PASS | New integration cases plus unchanged P1A lifecycle tests. |

## Final commands: verbatim exit codes, stdout and stderr

The runner was invoked with `.\.venv\Scripts\python.exe tools/verify_p1b.py` and itself exited 0. Each child command below used that same project interpreter. Empty stderr is stated explicitly; fenced stdout preserves the command output.

### 1. `.\.venv\Scripts\python.exe -m pip check`

Exit code: **0**

Stdout:

```text
No broken requirements found.
```

Stderr: empty.

### 2. `.\.venv\Scripts\python.exe -m pytest tests/unit/test_clocks.py tests/integration/test_capture_tracking.py -q`

Exit code: **0**

Stdout:

```text
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 17.85s
```

Stderr: empty.

### 3. `.\.venv\Scripts\python.exe -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json`

Exit code: **0**

Stdout:

```text
{
  "status": "pass",
  "output": "reports/benchmark.json",
  "network_attempts": [],
  "runs": [
    {
      "imgsz": 416,
      "frames": 300,
      "processed_fps": 37.41818887188351,
      "dropped_frames": 0
    },
    {
      "imgsz": 512,
      "frames": 300,
      "processed_fps": 32.59318979652977,
      "dropped_frames": 0
    },
    {
      "imgsz": 640,
      "frames": 300,
      "processed_fps": 26.613438842720484,
      "dropped_frames": 0
    }
  ]
}
```

Stderr: empty.

### 4. `.\.venv\Scripts\python.exe -m pytest -q`

Exit code: **0**

Stdout:

```text
........................................................................ [ 23%]
........................................................................ [ 46%]
........................................................................ [ 69%]
........................................................................ [ 92%]
.........................                                                [100%]
313 passed in 13.50s
```

Stderr: empty.

### 5. `.\.venv\Scripts\python.exe -m vertebrate doctor --offline`

Exit code: **0**

Stdout:

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

Stderr: empty.

## Earlier attempts, deviations and remaining work

The first new-test run exited 0 with `31 passed in 5.73s`. The first complete acceptance attempt passed its tests/doctor but benchmark exited 1 with `OfflineNetworkError: Offline mode blocked subprocess.Popen`. Investigation reproduced that failure in Python 3.12.0 platform.platform(): Windows version lookup invokes the local `ver` command even after uname was cached. The benchmark now reads cached uname fields and PROCESSOR_IDENTIFIER instead. The offline guard was not weakened, and the final runtime run made zero network/process attempts. The original command evidence is preserved in p1b-verification-initial.json.

A subsequent retry was interrupted and retained only pip check and the 139-test result; its incomplete evidence is preserved in p1b-verification-interrupted.json and is not counted as a completed acceptance pass. Final evidence above comes from the completed resumed run. A read-only process inventory via CIM was denied by the environment; no permissions were changed, and it was not an acceptance gate. `git diff --check` exited 0; Git emitted only its configured LF-to-CRLF working-copy notices.

No dependency-pin changes or model changes were needed. Existing AVI/MJPG clips decoded with valid source timestamps in these runs; no new codec workarounds were required. Use a single vision owner: the library numeric ID counter and Python offline guards are process-global. The guards cover the pinned Python stack, not arbitrary native-code networking.

P2A can build feature extraction on immutable original-image PoseObservation/TrackSample records, masks, source times, lifecycle events and (session_id, tracker_id, generation) keys. Real consenting human footage and eventual holdout evaluation are still required for observation quality, identity stability and calibration. Features, fall FSM, GUI and evaluation remain subsequent-phase work; no clinical, alert-accuracy or full-product release claim is made.
