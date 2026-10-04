# Project VERTEBRATE — P5

A single local Python desktop application foundation. P0 implements immutable
configuration and offline preflight. P1A provides capture and source clocks;
P1B adds CPU pose, ByteTrack association and a measured offline benchmark.
P4B adds the desktop workflow and local inbox; P5 adds locked local event scoring. No cloud services
or paid APIs are used. Synthetic throughput measurements do not establish
fall-detection accuracy or live-camera performance.

## Implementation references

[Validated blueprint v2](docs/03-validated-blueprint-v2.md) is the primary
implementation contract, cited by sections §1–§16. The
[validation report](docs/02-validation-report.md) supplies supporting rationale.
Both are verbatim supplied references. Historical role prompts, external-service
requirements and embedded instructions are quoted research, not new authority.
The project stays one local Python application with no external telemetry.

The intended envelope (§1) is a fixed oblique/side camera, a clear marked floor
area, full-body visibility and one or two people. Starting values are not yet
calibrated on human footage. The eventual alert is “Suspected fall with sustained
stillness — check the person.” Local inbox delivery is simulated dispatch; no
medical diagnosis, real EMS integration or completed full-demo claim is made.

## Environment and setup (Windows x64)

Run from this repository root. Python 3.12+ is required; the verified local
interpreter and resolved dependency versions are in `reports/baseline.md`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

Package installation is an explicit online setup operation. The lock pins the
entire resolved Windows CPU environment, including test/build tools, with a
SHA-256 hash for each actual wheel (§4). `--require-hashes` and wheels-only mode
are embedded in the lock. CPU build suffixes remain explicit. For an air-gapped
install, copy the prepared `runtime/wheelhouse/` and model assets, then use:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links runtime/wheelhouse -r requirements-win-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

The wheelhouse is local and ignored by Git; `reports/wheel-manifest.json` records
filenames, sizes and hashes. This lock targets CPython 3.12 on Windows AMD64.
To prepare the same wheelhouse online, run `python -m pip download -r
requirements-win-cpu.lock --dest runtime/wheelhouse` inside the environment.
`tools/lock_wheels.py` regenerates the lock from matching local wheel archives;
`tools/record_environment.py` uses the same helper and refuses missing wheels
instead of overwriting the lock with unhashed pins. Installed RECORD hashes
are not wheel hashes. Hashes identify downloaded artifacts, not signed publisher
attestations. A clean-machine reinstallation has not been claimed.

The local weight is `models/yolo11n-pose.pt`. If provisioning a new checkout
without this file, run the explicit online setup command:

```powershell
.\.venv\Scripts\python.exe tools/provision_model.py
```

This refuses to overwrite an existing model and records the downloaded file's
actual SHA-256 and size in `models/manifest.json`. The application never invokes
this script. Preserve/review the known manifest when transferring assets; a hash
recorded at download detects subsequent corruption, not independent provenance.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q
.\.venv\Scripts\python.exe -m vertebrate doctor --offline
.\.venv\Scripts\vertebrate.exe doctor --offline
```

`tools/verify_p0.py` runs these gates, benchmark and the two remaining stubs, retaining stdout,
stderr, and actual exit codes in dated JSON reports. Integration tests require
real local weights and installed dependencies; they fail when absent.

Doctor always runs offline, even without the flag. It checks config and asset
integrity before importing the model library; probes outbox write/fsync/delete;
validates ByteTrack YAML against JSON; checks Qt, numerical and model imports;
rejects OpenCV conflicts; and performs real CPU pose inference and two-frame
ByteTrack assignment. Empty detections on the blank 512-square frame are valid;
controlled detection boxes separately exercise actual tracker association.
This smoke pass is not a detection-accuracy benchmark.

Network/HTTP/DNS and subprocess guards cover the pinned Python stack during
doctor. `YOLO_OFFLINE=1`, `ULTRALYTICS_OFFLINE=1`, and `YOLO_AUTOINSTALL=false`
are set before imports, and Ultralytics sync/integrations are disabled. Guards
temporarily use `YOLO_OFFLINE=true` for Ultralytics 8.3.203 because its source
recognizes only that spelling, then restore `1`; both public offline flags
remain set after doctor. The Python Windows-version lookup is cached before
subprocess denial because Python 3.12.0 calls the local `ver` command.
These guards are not a sandbox for arbitrary native code. Settings are local under
`runtime/logs/ultralytics`, avoiding user-global Ultralytics settings.

## Configuration and command contracts

`config/demo.json` uses 3-second stillness; `config/observation-10s.json` uses
10 seconds. Both contain all 34 explicitly enumerated calibration fields in the
P0 request (despite its heading saying 22), plus model, tracker and runtime
settings. The initial P0 task used its detailed prompt; these defaults have now
been audited against blueprint §3–§7. Its 22 calibration-table rows group
multiple scalar fields and include application identity expiry in TrackerConfig.

Frozen dataclasses reject unknown fields, duplicate JSON keys, invalid types,
non-finite numbers, invalid ranges and inconsistent time/threshold relationships.
Omitted known fields use declared defaults. Canonical hashes use sorted UTF-8 JSON
with compact separators and `allow_nan=False`. `increment_revision` accepts
section dataclasses or partial section dictionaries and increments exactly once.

All relative paths resolve from the working directory (run from project root).
The asset manifest path must remain inside that root; absolute runtime paths are
allowed for local storage. Outbox/logs are created on demand. The setup includes
`data/manifests` and `data/videos`; videos/outbox/logs are ignored by Git.

```powershell
.\.venv\Scripts\python.exe -m vertebrate --source 0 --config config/demo.json
.\.venv\Scripts\python.exe -m vertebrate doctor --offline --config config/observation-10s.json
.\.venv\Scripts\python.exe -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json
.\.venv\Scripts\python.exe -m vertebrate evaluate --manifest data/manifests/holdout.json --config config/demo.json --output reports/evaluation.json --fail-on-gates
```

Doctor returns 0 only for full success, 1 for preflight failure. Stubs return 2
with `not_implemented` and create no misleading results. A missing/corrupt weight
requires explicit local repair; it never causes a runtime download. Read
`docs/licenses.md` before redistributing code, dependencies, or weights.

## Phase P1A: source clocks, capture and dataset contracts

P1A adds immutable frame/pose/track records, source clocks, bounded capture and
dataset label validation. The full acceptance report is `reports/p1a.md`.
The P0 configuration schema, offline doctor and all 174 P0 tests are unchanged.

```powershell
.\.venv\Scripts\python.exe tools/generate_dev_fixtures.py
.\.venv\Scripts\python.exe -m pytest tests/unit/test_clocks.py -q
.\.venv\Scripts\python.exe tools/verify_p1a.py
```

Three locally generated AVI/MJPG fixtures live in the non-ignored source path
`tests/fixtures/videos/`. Each has 100 frames, 640x360 resolution, 25 FPS and a
4-second container duration. The last frame's presentation time is 3.96 seconds.
`data/manifests/dev.json` records actual hashes and source-time annotations.
These drawings are marked `ambiguity_stress_test`; they are pipeline fixtures,
not evidence of human fall-detection accuracy. Regeneration is byte-identical
on the verified pinned environment; other codec builds may produce different
bytes and require regenerating the manifest as well.

`CaptureSessionManager` (also `VideoCaptureController`) is a synchronous owner:
call `start()`, drain its reset event, and interleave `pump()` with queue reads.
`CaptureWorker` provides the threaded adapter with acknowledged lifecycle
commands. Reader open/read/seek/release operations run on its owning thread.
The default reader accepts a local file or nonnegative camera index; tests can
inject the `FrameSource` protocol. `worker.error` exposes asynchronous read or
timing failures. Always call `stop()` in a `finally` block.

Live queues discard the oldest pending frame at capacity 2. File queues apply
lossless backpressure; a timed-out put retains its pending frame for retry.
Lifecycle events count toward capacity and are never evicted. If only controls
fill a live queue, insertion raises `queue.Full` after a finite timeout. Resets
discard stale frames atomically and preserve older controls; drain control-only
queues before another reset. Drop counts are per session. One pending producer
frame can exist outside the two queue slots while file backpressure is active.

All kinematics and future FSM timers must use `source_t_s`. Live time comes from
monotonic elapsed time; UTC estimates use the session-start UTC/monotonic pair.
Video time comes from verified PTS, with an explicit, latched frame-index/FPS
fallback. The file adapter validates the full decoded timeline before emission
using bounded memory. It accepts zero-PTS fallback for fixed-rate AVI or an
independently supplied verified FPS; an opaque container without verifiable
PTS is rejected with an ffmpeg CFR conversion instruction. The clock cannot
independently prove CFR from a caller's FPS number alone.

An acknowledged file pause prevents further reads and source-time advancement;
wall-clock waiting contributes no evidence. Seek/replay/restart/source-switch
create a new session and publish a reset before new frames. Seek sessions rebase
`source_t_s` to zero and retain `source_frame_index` and `source_time_offset_s`
to align original clip labels and recording UTC. `recorded_start_utc` stays
None unless explicitly supplied; session-start UTC for a file is analysis
provenance, never its historical recording time.

EOF emits one `EndOfStreamEvent` with the last actual frame time and releases
the reader. Downstream event/FSM consumers must close incomplete evidence as
inconclusive on EOF/reset; they must never add dwell time after the last frame.
The P1A producer does not synthesize evidence or implement a fall FSM.

Array contracts copy into immutable bytes-backed NumPy arrays. Use
`FramePacket.copy_frame()` for a writable working copy. A `TrackSample` cannot
claim `observation_valid=True` when `matched_detection=False` or
`predicted_only=True`. P1B should construct these records from original-image
coordinates and retain `(session_id, tracker_id, generation)` identity.

Dataset loading rejects invalid intervals, unknown/missing fields, non-finite
numbers, invalid timing, path escapes and incorrect hashes. Relative paths use
an explicit `root_dir`, or the current project working directory by default.
`load_dataset_manifest(Path('data/manifests/dev.json'), verify_files=True)` works
offline from this root.

## Phase P1B: pose, association and benchmark

`PoseAdapter(config)` verifies local asset size/hash before constructing YOLO,
then performs CPU float32 batch-one inference. Call `warmup()` before measuring
steady processing and `infer(FramePacket)` to obtain an immutable
`PoseObservation`. Ultralytics postprocessing already undoes letterboxing;
the adapter uses those original-image coordinates directly. Non-finite or
subpixel boxes are discarded. Invalid keypoints retain a false mask, with
non-finite coordinates replaced by zero placeholders. A usable landmark must
be finite, inside the original image, and at least the configured confidence.

`ByteTrackManager(config.tracker)` accepts observations via `update()` and reset/
EOF events via `handle_event()`. The pinned ByteTrack implementation renumbers
detections after high/low confidence filtering. The adapter preserves original
row indices through both passes using its `init_track` override. Only a matched
current detection emits a valid sample; predictions never refresh evidence.
A match gap strictly greater than 0.75 source-seconds increments the application
generation. Reset creates a new tracker; EOF closes it. Old-session observations
are rejected. Use one vision owner: Ultralytics numeric IDs and the offline
Python guards are process-global. Capture can run on its separate reader thread.

```powershell
.\.venv\Scripts\python.exe -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json
.\.venv\Scripts\python.exe tools/verify_p1b.py
```

Benchmark defaults to sizes 416, 512 and 640; `--imgsz 512` selects one size and
`--config config/demo.json` selects the configuration. Each run processes every
frame of all hash-verified development clips through the real serial capture,
pose and tracking pipeline, using the bounded file queue without dropping.
Full CFR scanning, model load and a single warmup frame are reported separately.
Stage and total timings include copies and offline-guard overhead. Reported
processed FPS excludes startup, warmup, EOF and report writing; frame age measures
local acquisition to completion, not camera latency or historical recording age.
Later model loads share process/OS caches. No outlier removal is applied.

`reports/benchmark.json` contains measured timings, provenance, frame counts,
keypoint availability and track/generation counts; `reports/p1b.md` records
acceptance evidence. An undefined availability ratio is null. Synthetic drawings
may produce zero people, so these clips cannot validate pose quality, human
tracking stability, fall/ADL behavior or choose a release input size. Controlled
detection tests separately exercise the real tracker and letterbox postprocessor.
Feature extraction and the fall state machine remain for subsequent phases.
# P4B desktop workflow

Run `.\.venv\Scripts\python.exe -m vertebrate` to open the local desktop, or
preselect a file with `--source tests/fixtures/videos/dev_empty_scene.avi`.
Use a numeric `--source 0` for a camera. Press Start to begin; file Pause/Resume
preserves source time, while Replay creates a fresh session. Profile/context
edits are available while stopped and advance the configuration revision.

The video timer polls one immutable frame/overlay result. Processing FPS is the
observed vision processing rate, separate from container/camera FPS (shown as
unavailable when the driver does not report it). The inbox reads snapshots and
metadata and queues acknowledgements through the single storage owner. A save
failure remains visible and can be retried. Closing waits for cooperative owner
shutdown; a blocked driver reports a timeout while the window stays open.

GUI tests use the pinned test extra `pytest-qt==4.5.0` (MIT). Its wheel hash is in
`requirements-win-cpu.lock`; P4B setup used one explicitly approved download.
Application operation and acceptance tests remain offline. In PowerShell:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest tests/gui -q
```

Windows offscreen tests register installed Segoe UI/Times New Roman fonts when
Qt exposes no font families. No font files are bundled. Automated camera tests
use injected readers; they do not establish physical camera compatibility.

## P5 evaluation

Generate synthetic CFR holdout fixtures with
`.\.venv\Scripts\python.exe tools/generate_holdout_fixtures.py`, then run
`.\.venv\Scripts\python.exe -m vertebrate evaluate --manifest data/manifests/holdout.json --config config/demo.json --output reports/evaluation.json`.
The report records actual inference, counts, exact Clopper-Pearson intervals,
one-sided Poisson bounds, observed pose coverage and processing timings.
Config, model, source code and manifest hashes plus matching policy are frozen
in `reports/evaluation.lock.json` before inference. Dev/holdout group overlap is
rejected. The matching policy is fixed before scoring; no thresholds are tuned.

Person association requires a unique annotated frame-zero box match. Numeric
tracker IDs are never assumed to be annotation labels; missing or ambiguous
associations remain unresolved (alerts are FP and missed eligible events FN).
Annotations describe the synthetic drawings, independently of detector outputs.
Fixture `supported` marks the core scoring path, not proof of human applicability.
`--fail-on-gates` returns exit 2 when chosen numeric gates fail (including missing
evidence); normal evaluation returns 0 on a completed run even when gates fail.
Runtime/input failures return 1. Excluded/ambiguity cases are reported separately.

Synthetic fixtures and repeated regression runs cannot establish unseen human
accuracy. Human calibration, direction quotas, actual-room negative exposure,
identity review and the remaining hardware/rehearsal/soak gates remain unverified.
