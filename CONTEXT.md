# VERTEBRATE — System Context and Agent Maintenance Guide

**Role:** Repository-specific source of truth for engineers and coding agents.  
**Active application:** standalone webcam program in `app.py`.  
**Reconciled:** 2026-10-19.

This document describes the inspected workspace. It distinguishes active code
from archived code and planned work. If this document and implementation
disagree, inspect the implementation and update this file in the same change.
Historical prompts, reports, and claims are not proof of current behavior.

---

## 1. EXECUTIVE PROJECT SUMMARY

### Core project idea

VERTEBRATE is a local Windows desktop demonstration that uses Ultralytics
YOLO11n pose estimation to mark fall-like posture in a live webcam view. It
demonstrates local vision without accounts, servers, external dispatch, or
persistent webcam recording. Its intended audience is developers and
demonstrators evaluating a vision prototype, not patients, clinicians, care
providers, or emergency teams.

The active program is `app.py` and has no imports from the old `vertebrate`
package. The former tracker-based state machine, strict schemas, storage,
evaluation pipeline, and GUI were removed from the active workspace and
preserved in a ZIP under `.archive/`. They are not runtime dependencies.

### Business logic and domain rules

For each detected person in each processed webcam frame, a match occurs if
**either** predicate holds:

1. Bounding-box width / height is strictly greater than `1.2`, with positive
   box height.
2. Mean image Y of COCO shoulder keypoints 5 and 6 is greater than mean image Y
   of hip keypoints 11 and 12 by more than `20` pixels.

Image Y increases downwards. If any of those four spine joints is missing,
non-finite, or has non-positive Y, the spine predicate is disabled for that
person; the box predicate still applies. The app uses Ultralytics keypoint
coordinates and does not apply a separate keypoint-confidence cutoff.

A frame is positive if any detected person matches. Five consecutive positive
**processed inference frames** turn on the alert. A processed negative frame
resets the count. Five frames are not a fixed duration because model and camera
rates vary. Debounce is scene-wide, so successive frames may be positive for
different people. There is no identity association; do not call this per-person
temporal confirmation.

This detects posture, not fall causality. Intentional lying down, rotated or
overhead cameras, occlusion, proportions, and pose errors can affect it. A
match does not establish injury or need for help. The UI says **“Local demo —
no external emergency dispatch.”**

---

## 2. SYSTEM ARCHITECTURE & DATA FLOW

### High-level architecture

A local desktop monolith with an event-driven Qt thread boundary. One `QThread`
owns the OpenCV camera, YOLO model, inference, geometry, annotation, and
debounce. The Qt main thread owns `QMainWindow` and all widgets. The worker emits
a detached `QImage` and boolean alert state by Qt signal; queued delivery runs
the receiving slot on the main thread. There is no database, authentication,
microservice, or cloud-processing tier.

| Responsibility | Code | State |
|---|---|---|
| Geometry | `fallen_geometry()` | None; frame-local |
| Annotation | `annotate()` | Current mutable frame only |
| Debounce | `Debounce` | Positive-frame count, worker-owned |
| Camera/inference | `CameraThread.run()` | Capture, model, debounce, pending-frame event |
| Desktop UI | `MainWindow` | Widgets, pixmap, flash timer, worker lifecycle |

The model and camera are constructed in `CameraThread.run()`. The camera is
released in that worker's `finally` block. Stop uses `requestInterruption()`
and is observed between blocking reads/inference. After five seconds, the UI
shows a wait message; it does not forcibly terminate a running thread.

### Core technology stack

`requirements.txt` pins direct Windows x64 CPU dependencies.
`requirements-win-cpu.lock` is the 48-distribution hash lock retained from the
former full project environment. It includes packages no longer required by
the standalone app.

| Technology | Version | Role |
|---|---|---|
| Python | 3.12; workspace interpreter 3.12.0 x64 | Language/runtime |
| Ultralytics | 8.3.203 | Local model loading and pose inference |
| YOLO11n pose weights | v8.3.0 asset | Person boxes and 17 COCO keypoints |
| PyTorch | 2.8.0+cpu | CPU inference |
| torchvision | 0.23.0+cpu | PyTorch vision operators |
| NumPy | 2.2.6 | Coordinates and finite checks |
| opencv-python | 4.12.0.88 | Camera, drawing, color conversion |
| PySide6 / Qt | 6.9.2 | Window, QThread, signals, timers |

Inference explicitly specifies CPU; CUDA is not selected. Use `opencv-python`
only, never install headless or contrib variants alongside it. Python 3.12 x64
on Windows is the documented environment. No inference-rate guarantee is
specified.

### Data architecture and state

There is no application database or user record schema. The app does not save
snapshots or webcam clips. The only persistent record is the per-device fall
log CSV (see "Fall log" below).

| Data | Lifetime/location | Notes |
|---|---|---|
| Webcam frame | Worker memory | BGR NumPy array, annotated in place |
| Model result | One inference iteration | Boxes and `keypoints.xy` only |
| Debounce count | Worker session | Integer capped at five; reset by negative |
| Display image | One pending signal | RGB QImage detached with `copy()` |
| Display pixmap | Main window memory | Replaced on delivery and scaled on resize |
| Model weights | Local filesystem | Root checked first, then `models/` |
| Ultralytics settings/cache | `runtime/logs/standalone-yolo/` | Local and ignored by Git |
| URFD footage | `data/videos/urfd/` | Local clips and ignored by Git |
| Fall log | `runtime/logs/falls/<device>.csv` | One row per finished fall event; ignored by Git |

### Fall log

`FallLog` (GUI thread, fed from `MainWindow.show_frame()`) turns the debounced
alert into events. An event **starts** when the alert turns on (i.e. after the
five-frame debounce, so slightly after the posture began) and **ends** when the
alert has stayed off for `END_GRACE_S` = 2 s; the end time is when it first
went off, so short flickers do not split one fall. Stopping the camera, a worker
error, or closing the app ends an ongoing event with that reason. Each finished
event appends one CSV row: `device, fall_detected_at, ended_at, utc_offset,
duration, duration_seconds, latitude, longitude, accuracy_m, location_source,
ended_by`. Location is the `LocationFeed` fix at the start (or the first fix
during the event). The device name is the host name (`QSysInfo`), and each
device writes its own file. The file is UTF-8 with BOM so Excel shows symbols.
The **Fall log** button opens a non-modal `FallLogDialog` listing this device's
events newest first, with an ongoing fall shown live on top, totals, and a
button that opens the log folder. Write/read failures are shown in that window
and printed to stderr; they never stop monitoring.

`models/manifest.json` records model name/path, SHA-256, size, source, version,
task, and license. At startup, `app.py` checks only that the selected file
exists and is non-empty. It does **not** read the manifest or verify the hash.
Never claim launch-time integrity checking.

### Critical runtime flow

1. The user clicks **Start camera**.
2. The GUI creates/starts `CameraThread`, disables Start, enables Stop.
3. The worker calls `local_weights()`. Missing weights produce a local error
   before `YOLO` is constructed; no download fallback is intended.
4. The worker selects the local Ultralytics config directory, disables `sync`
   and `hub` settings in memory, constructs a pose model, and opens
   `cv2.VideoCapture(0)`.
5. Each camera frame runs YOLO on CPU. `fallen_geometry()` evaluates the two
   predicates for each person. `annotate()` draws box and visible skeleton red
   for matches and green otherwise.
6. Debounce updates once per processed frame. The annotated BGR frame becomes
   RGB; a detached QImage and current boolean are emitted only if no previous
   display frame is pending. Inference continues when display frames are skipped.
7. `MainWindow.show_frame()` receives the queued signal, updates/scales a pixmap
   and status, then clears the pending flag. Widgets are updated on the GUI thread.
8. Stop/close requests interruption. When a blocking read/inference returns,
   the worker exits, releases the camera, and completes; the GUI restores controls.

### Network and privacy behavior

Before importing Ultralytics, `app.py` sets `YOLO_OFFLINE=true`,
`ULTRALYTICS_OFFLINE=1`, `YOLO_AUTOINSTALL=false`, and a local
`YOLO_CONFIG_DIR`. It disables Ultralytics `sync` and `hub` settings in memory.
Weights must be present locally. There is no upload, remote inference, or
dispatch code. Webcam frames stay in process memory. Package installation or a
manual weights download is separate setup and may require a network connection.

### Location watermark

`LocationFeed` (GUI thread) runs up to two Qt Positioning sources:

1. **GPS receiver** (`"nmea"` plugin), only when `find_gps_port()` finds a
   serial/USB/Bluetooth port whose vendor ID (u-blox, Garmin) or description
   looks like GPS/GNSS. `VERTEBRATE_GPS_PORT` overrides detection (`COM4`,
   `/dev/cu.usbmodem1101`, or `off`); `VERTEBRATE_GPS_BAUD` defaults to 9600.
2. **System location** (`"winrt"` on Windows, `"corelocation"` on macOS) with
   all positioning methods allowed, so the OS uses a built-in GPS chip when the
   device has one and Wi-Fi/network positioning otherwise.

A GPS-receiver fix received within 10 s wins; otherwise the system fix is
shown. Sources start with **Start camera** (the system source after a one-time
`QLocationPermission` request) and stop when the worker finishes.
`paint_watermark()` draws latitude/longitude, horizontal accuracy, local
date/time and the source in the bottom-left of each displayed frame inside
`MainWindow.show_frame()`; the worker thread is not involved. Only the latest
fix per source is held in memory; it is never saved or sent by the app. OS
Wi-Fi positioning may contact the OS vendor's location service. If access is
denied, the watermark shows the reason. Fixes older than 60 s show their age.

### Platform support

Windows 10/11 x64 and macOS, Python 3.12, CPU inference. `open_camera()` tries
DirectShow then Media Foundation on Windows and AVFoundation on macOS; camera
help text names each OS's privacy setting. The UI uses the platform system
font (`QFontDatabase.systemFont`) and scales the video in physical pixels for
Retina/high-DPI displays. Install with `requirements.txt` on Windows and
`requirements_macOS.txt` on macOS.

---

## 3. PROJECT SCAFFOLDING & CODEBASE STRUCTURE

### Current workspace tree

Observed 2026-10-19. Generated wheel/cache files and video filenames are grouped.
`.venv/` and `.git/` internals are not expanded.

~~~text
Vertebrate/
├── app.py                         # Standalone app/runtime
├── README.md                      # Setup, heuristic, limitations, licensing
├── CONTEXT.md                     # This architecture and agent guide
├── requirements.txt               # Pinned direct CPU dependencies
├── requirements-win-cpu.lock      # Former environment lock, 48 distributions
├── LICENSE                        # MIT for Advaet Chintakunta's original code
├── THIRD_PARTY_NOTICES.md         # Dependency/model license boundaries
├── .gitignore                     # Ignores venv, local data, weights, outputs
├── .gitattributes                 # Git text normalization/reference rules
├── cleanup-legacy.ps1             # Migration utility; -Apply is destructive
├── yolo11n-pose.pt                # Local root weights; ignored by Git
├── .venv/                         # Local Python environment; ignored
├── .archive/
│   ├── legacy-<id>.zip            # Removed legacy source/config/docs/tests/tools
│   └── release-prep/              # Earlier README/.gitignore copies
├── models/
│   ├── manifest.json              # Provenance and expected model hash
│   └── yolo11n-pose.pt             # Local copy matching manifest; ignored
├── data/videos/urfd/
│   ├── adl-01..40-cam0.mp4        # 40 local ADL clips
│   ├── fall-01..30-cam0.mp4       # 30 fall views, camera 0
│   ├── fall-01..30-cam1.mp4       # 30 fall views, camera 1
│   └── process_urfd.py            # Preparation helper; inspect before running
└── runtime/                        # Generated/ignored local state
    ├── calibration/                # Track caches, contact sheet, replay logs
    ├── logs/                       # Ultralytics settings/screenshots
    ├── outbox/                     # Legacy artifact; app.py does not use it
    ├── setup-wheels/                # Local setup wheel(s)
    └── wheelhouse/                  # Downloaded Windows wheels
~~~

The 100 MP4 files are local, not in the normal Git tree, and are not read by the
current webcam app. The root and `models/` weights were observed to have equal
size; the `models/` file matches the manifest hash. Runtime lookup prefers the
root copy if both are present.

The ZIP contains the old `src/vertebrate` package (capture/clocks,
contracts/config, features/FSM, tracking, pipeline, storage/incidents/context,
GUI, evaluation/calibration), configs, manifests, docs, reports, and synthetic
tests. It is archival, not runtime code. The cleanup script backs up and deletes
broad legacy paths when passed `-Apply`. Inspect targets and archive before
any future use.

### Component and code conventions

`app.py` keeps the demo compact. `fallen_geometry()` and `annotate()` are
standalone functions; `Debounce` owns its counter; `CameraThread` owns blocking
I/O and inference; `MainWindow` owns visible state. Preserve that boundary.
Extract modules only when they create a clear interface and retain a simple
`python app.py` launch.

Use `snake_case` for functions/variables/modules, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for constants. Use four spaces, type hints for new public
interfaces, concise docstrings, and comments for thread/lifecycle invariants.
There is no active root `pyproject.toml`, linter, formatter, or packaging
metadata. Do not add one as a side effect of unrelated work. Pin runtime
dependencies in `requirements.txt` and update notices if dependency or model
provenance changes.

---

## 4. DEVELOPMENT & DEPLOYMENT STANDARD OPERATING PROCEDURES (SOPS)

### Setup and launch

From the repository root in Windows PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe app.py
~~~

Keep a healthy existing environment. Fresh setup needs wheels and local weights;
prepare them before offline use. Put weights at root or `models/`. Missing
weights fail locally. The app opens camera index 0; file input and camera
selection are not implemented.

`requirements-win-cpu.lock` is the hash-checked lock for the larger old
environment, including pytest and former phase packages.
`requirements.txt` is the direct install surface for the standalone app. Review
both manifests if runtime dependencies change. Install only one OpenCV variant.

### Coding and threading standards

Camera reads, inference, and annotation stay off the Qt main thread. Create and
release VideoCapture in its owner thread. Avoid unbounded waits in GUI slots.
Pass detached Qt values across threads: `QImage.copy()` is required because a
NumPy-backed image may be mutated when reused. Use queued signals. Keep at most
one display frame pending or use an equivalent bounded latest-frame mechanism.

Signal pattern:

~~~python
class CameraThread(QThread):
    frame_ready = Signal(QImage, bool)
    error = Signal(str)

    # After inference and BGR -> RGB conversion:
    image = QImage(rgb.data, width, height, rgb.strides[0],
                   QImage.Format.Format_RGB888).copy()
    self.frame_ready.emit(image, is_fallen)

worker.frame_ready.connect(window.show_frame, Qt.ConnectionType.QueuedConnection)
~~~

Display backpressure must not skip inference or change debounce. Reset debounce
per worker session. Emit readable errors and release owned resources in
`finally`.

### Error handling and logging

The worker catches runtime exceptions and emits type/message. The GUI displays
the error and restores controls on thread completion. Camera open/read failures
have specific guidance. There is no global exception framework or rotating
app log; fall events go to the per-device fall log CSV. Ultralytics settings/cache are under
`runtime/logs/standalone-yolo/`.

Do not suppress failures that could leave the camera open or UI stuck. New
worker resources belong in try/finally. New lifecycle/error paths must restore
GUI controls to a known state.

### State and security

There are no credentials, accounts, roles, endpoints, or secrets. Do not add
cloud inference, telemetry, analytics, or dispatch without explicit scope
change. Set offline/autoinstall flags before importing Ultralytics and keep
sync/hub disabled. Never pass missing weights to YOLO, whose normal loading can
attempt a download. Webcam frames stay in process memory unless recording is
explicitly requested.

Startup does not verify model hashes. If integrity checking is added, compare
the selected weight against `models/manifest.json` before `YOLO(...)` and fail
on mismatch. Never fetch a replacement automatically. Weights and video are
separately licensed/provenance-bearing assets, not MIT source.

### Licensing and release

`LICENSE` grants MIT rights over Advaet Chintakunta's original code. Ultralytics
software and YOLO weights have separate AGPL-3.0 terms. The MIT file does not
relicense them or establish that a combined distribution is MIT-only. Before a
source/binary release, review applicable Ultralytics and Qt terms, retain
notices, and ensure the release model satisfies them. This is a technical guide,
not legal advice.

`.gitignore` excludes videos, model weights, environment, runtime, reports, and
archive. The model manifest preserves provenance without the binary. Do not
force-add weights or private video without an explicit release decision and
license review. Do not commit `.venv`, runtime cache, calibration output, or
wheelhouse.

### Testing protocol

There is no active `tests/` directory; old tests are inside the archive and do
not verify `app.py`. For app changes, add focused geometry/debounce tests and
offscreen Qt lifecycle tests where relevant. Mock the camera/model for
repeatable automated tests, then separately test a real webcam on target hardware.

Suggested future checks:

~~~powershell
.\.venv\Scripts\python.exe -m compileall app.py
.\.venv\Scripts\python.exe -m pip check
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
~~~

Pytest is prospective until tests are restored or added. Manually check camera
open, live frames, red/green overlays, five-frame alert/reset, Stop, close during
inference, and missing weights. Record actual outcomes. Mock camera evidence is
not physical-camera verification. Do not present a development score as a
safety/accuracy claim.

---

## 5. AGENT DRILLS & GUARDRAILS (CRITICAL)

### Modification rules

1. Inspect `app.py`, `requirements.txt`, `models/manifest.json`, `.gitignore`,
   and this document before architecture changes. Check archive/runtime for
   user data before moving or deleting anything.
2. Treat `app.py` as active. Do not revive archived imports, ByteTrack, schemas,
   storage, manifests, or old Qt workers without an explicit request.
3. Preserve exact predicates/debounce: ratio `>1.2`, spine
   `shoulders_y > hips_y + 20`, five consecutive processed positives, reset on
   a processed negative. If user requests changed behavior, update this file
   with code and document the semantics.
4. Preserve thread ownership. Never update widgets from `CameraThread`, share a
   NumPy-backed live QImage, create/release the camera on the GUI thread, or
   destroy an active QThread.
5. Preserve offline behavior. Check local weights before model construction,
   keep autoinstall/telemetry disabled, and do not implicitly add network
   dependencies, uploads, credentials, or remote services.
6. Keep local footage/weights out of Git by default. Preserve `data/videos/`.
   Do not routinely delete archive, venv, runtime data, weights, or footage.
   `cleanup-legacy.ps1 -Apply` deletes its listed targets after archiving them;
   inspect both before use.
7. Retain comments explaining resource ownership, offline behavior, and Qt
   threading. Correct stale comments. Add type hints to new public interfaces
   and avoid broad unrelated refactors.
8. Do not claim “30/30” or another benchmark for this app without checking
   dataset, labels, runner, and report. There is no active evaluation runner.
   Do not claim medical performance, fall causality, dispatch, or physical
   hardware validation without evidence.
9. Preserve `python app.py` and both local model paths. Update README,
   dependencies, provenance, and this document if they change.
10. Keep MIT and third-party license scope separate. Review notices whenever a
    dependency or model changes.

### Definition of Done

- [ ] Change targets standalone app and does not depend on archived modules.
- [ ] Geometry, invalid joints, debounce, and alert semantics match request and
      have focused tests where applicable.
- [ ] Model/OpenCV work stays in worker; GUI updates are queued; frame data is
      detached and bounded.
- [ ] Start, Stop, errors, missing weights, camera-open failure, and close leave
      worker/camera resources and GUI controls in defined states.
- [ ] No model auto-download, telemetry, upload, or unintended recording added.
- [ ] Dependency pins, model provenance/hash, ignore rules, and notices updated
      if inputs change.
- [ ] Relevant checks actually ran and exact outcomes are reported. Clearly
      state when tests are absent or could not run.
- [ ] Mocked camera results and physical hardware checks are distinguished.
- [ ] README and this context describe resulting code, not abandoned plans.
- [ ] Handoff lists material changes, verification, and limits without
      overstating performance or release readiness.

---

## Current known limitations

- Geometry detects posture and may classify ordinary lying down as fall-like.
- Fixed 20-pixel margin changes meaning with image scale and subject distance.
- Scene-wide debounce can combine detections from different people.
- Spine geometry has no explicit keypoint-confidence cutoff.
- No tracking, video input, camera selector, recording, benchmark CLI, or
  release evaluation is active. The fall log records alert times, not footage.
- A fall in progress when the app crashes is not logged (rows are written when
  an event ends).
- Root weights are preferred; manifest hash is not enforced at startup.
- Historical URFD reports belong to archived code and do not establish this
  app's performance.
- Prior tests are archived. Physical webcam validation remains necessary on
  each target environment.

