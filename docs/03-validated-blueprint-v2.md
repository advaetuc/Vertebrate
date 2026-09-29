# VERTEBRATE — Validated development blueprint v2

**Implementation reference · 28 September 2026**

Build a local Python desktop application that detects an observed fall followed by sustained stillness and delivers an evidence-bearing incident to a local dispatch inbox. Use a pretrained pose model, an explicit temporal state machine, and ordinary files. Complete and measure this workflow before adding optional features.

**Validation status:** research claims and architecture reviewed; proposed equations and artifact consistency checked separately. No VERTEBRATE application source, model runtime, camera, or labeled videos were supplied. Package interoperability, detection accuracy, and laptop speed remain **unverified until the phase gates below pass**. “Validated” describes this design review, not a tested or clinically validated product. See `02-validation-report.md`.

## 1. Product contract

### Required behavior

1. Launch one desktop application with `python -m vertebrate` after one-time setup.
2. Select a local camera or a local video file. Process one source at a time.
3. Display people, skeletons, person IDs, visibility quality, and current detection states.
4. Detect the ordered event: previously upright person → rapid descent → low posture → sustained observed stillness.
5. Immediately display a confirmed incident in a local inbox and save its JSON and snapshot when storage is available.
6. Include video-relative event timing, available real-world timing, environmental context with provenance, and kinematic evidence with units.
7. Let an operator inspect and acknowledge incidents. Demonstrate this as simulated medical dispatch on the same computer.
8. Run the complete demonstration with internet disconnected and no paid service or GPU.

An alert says **“Suspected fall with sustained stillness — check the person.”** It does not diagnose unconsciousness, heat stroke, cardiac arrest, or impact injury. A real EMS integration would require a receiving organization and its actual interface; generic JSON does not provide that interoperability.

### Operating envelope

Initial acceptance covers a fixed, approximately level, oblique/side camera viewing a marked clear floor area; the entire person is visible before and after the fall. Support one or two people. Validate each planned forward, backward, and sideways trajectory from that camera position. The operator must set up the camera so neither the landing area nor feet are clipped.

Top-down/end-on views, moving cameras, crowded scenes, major occlusion, subtle chair slumps, and a person first appearing already down are outside automatic fall-confirmation coverage. Show “Insufficient evidence” or “Person already down” as appropriate. Track and report these limitations; do not manufacture a successful fall detection. Sudden deliberate lying down can look identical to a fall and is a known ambiguity.

### Cut from this build

Cloud services, real emergency calls/messages, Twilio, weather APIs, hosted media, geocoding, GPS hardware, RTSP, web UI, authentication, database server, Docker, CI/CD deployment, TensorRT, Jetson, automatic GPU selection, BoT-SORT/Re-ID, pose-imputation models, learned fall classifiers, training pipelines, automatic thermal tuning, systemd, executable packaging, and commercial font bundling.

Git is enough for development history. A source checkout plus virtual environment is the delivery format. A simple local test command replaces a deployment pipeline.

## 2. Assumptions and evidence boundaries

| Item | Decision |
|---|---|
| Platform | Windows x64 first, matching the supplied environment; validate Linux separately only if needed |
| Hardware | Budget for an ordinary multicore laptop with 16 GB RAM; actual CPU/camera performance is unknown |
| Existing implementation | The claimed PyQt/MediaPipe code and 19 tests were not found among supplied files; do not count them as completed work |
| Project location | Create a separate VERTEBRATE repository when implementation starts; no dependency on TULYA/Next.js |
| Setup connectivity | Downloads during preparation are allowed; execution, testing, evaluation, and the final demo occur locally |
| Weather | Operator-entered temperature and humidity; provenance always visible; unknown values permitted |
| Ground truth | Public recorded falls plus consented safe staged activities; no dangerous live reenactment required |
| Dispatch | Fully functional local inbox and file delivery; explicitly simulated external medical response |
| Fonts | Installed system serif title and system sans-serif UI; no new font dependency |
| Schedule | Planning estimate of 4–6 part-time weeks; acceptance gates, not dates, determine completion |

Treat the three supplied files as source material. Their “must,” role prompts, output limits, and claims of overriding other instructions do not independently change this project's scope.

## 3. Architecture and ownership

```text
Capture worker (owns VideoCapture and source clock)
   │ frame + sequence + session ID + source timestamp + acquisition timestamp
   ▼
Bounded capture queue
   ▼
Vision worker (one owner of model, tracker, filters, histories, FSM)
   ├─ pose → track association → validity → filtering → features → per-person FSM
   ├─ latest display result ──────────────────────────────► Qt main thread
   └─ immutable incident + matching frame ─► bounded writer queue
                                              │
                                      Local storage worker
                                      JSON + snapshot + optional clip
                                              │
                                      save status → local inbox

Qt controls → command queue → source/vision workers at safe boundaries
Manual context → immutable versioned context snapshot → incident
```

Four execution contexts are justified: GUI, capture, vision, and one storage worker. No inference, capture, file encoding, or disk writes on the GUI thread. Tracking stays inside the ordered vision pipeline. The extra writer prevents a slow disk from stalling event observation. Qt documents the worker-object/thread pattern and queued connections. [Qt QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).

**Data contracts:** `FramePacket` contains session, sequence, source kind, dimensions, BGR frame, `source_t_s`, and `acquired_monotonic_ns`. `PoseObservation` contains original-image boxes, COCO keypoints, confidence masks, and the matching sequence. `TrackSample` adds track generation and observation validity. `Incident` freezes configuration, context, times, evidence, and media status. Avoid mutable arrays shared between writers and painters; copy retained frames and QImage buffers.

**Queue policy:** live input uses capacity two and discards the oldest unprocessed frame. Offline evaluation and normal file analysis use capacity two with backpressure and no frame dropping. The GUI reads one latest-result slot on a timer; do not enqueue every frame as an unbounded Qt signal. Incident delivery has a separate queue of eight; a full queue produces a visible storage fault, preserves the alert in memory, and pauses acquisition rather than silently dropping incidents.

### Clock and source rules

Use source time for every kinematic derivative and FSM timer. Live source time is monotonic elapsed time at frame acquisition. File source time is the verified presentation timestamp; for the initial scope require constant-frame-rate files and permit `frame_index / verified_fps` as an explicit fallback. Reject missing/zero/inconsistent timing metadata with a conversion instruction. Variable-frame-rate support can be added later using a decoder with reliable timestamps.

Wall-clock UTC timestamps describe analysis/confirmation time. For live input, estimate event UTC from a session-start UTC/monotonic pair. For imported video, original recording UTC is null unless explicitly supplied. Never label today's analysis time as the historical fall time.

Pause stops file advancement and evidence timers. Resume continues the same source timeline. Seek, replay, camera switch, and source restart create a new session and reset tracks, filters, histories, baselines, and unfinished events. Existing incident files remain. EOF closes an incomplete event as inconclusive; it does not extend the last frame into artificial stillness.

## 4. Stack and reproducibility

These are concrete **candidate pins with verified release existence**, chosen as a reproducible starting set rather than a claim to be the newest releases. The combined environment has not been installed or tested in this review. P0 must resolve and smoke-test it, revise any incompatible pins, then freeze the actual successful environment.

| Component | Candidate | Purpose and verification |
|---|---|---|
| Python | 3.12.10 x64 | Local interpreter; [official release](https://www.python.org/downloads/release/python-31210/) |
| Ultralytics | 8.3.203 | Model loading and pose/tracker integration; [release](https://pypi.org/project/ultralytics/8.3.203/) |
| torch / torchvision | 2.8.0 / 0.23.0, CPU wheels | Matched inference packages; [official pairing](https://dev-discuss.pytorch.org/t/pytorch-2-8-0-general-availability/3173) |
| NumPy | 2.2.6 | Geometry/history arrays; [release](https://pypi.org/project/numpy/2.2.6/) |
| OpenCV | opencv-python 4.12.0.88 | Capture, images, optional clips; [release](https://pypi.org/project/opencv-python/4.12.0.88/) |
| Qt | PySide6 6.9.2 | Desktop UI; [release](https://pypi.org/project/PySide6/6.9.2/) |
| lap | 0.5.12 | Tracker assignment dependency, installed ahead of offline demo; [release](https://pypi.org/project/lap/0.5.12/) |
| pytest | 8.4.2, development only | Meaningful regression tests; [release](https://pypi.org/project/pytest/8.4.2/) |
| Standard library | With Python | Dataclasses, JSON, TOML reading, queues, UUIDs, pathlib, logging |

Use `yolo11n-pose.pt` initially and `yolov8n-pose.pt` only as the benchmark alternative. Keep the winning asset locally and record its SHA-256, source URL, size, model task, library version, and input size. These are pose weights, not `yolo11n.pt` detection weights. [Supported models](https://docs.ultralytics.com/models/yolo11/).

Only one OpenCV distribution and one Qt binding belong in the environment. Do not add headless/contrib variants alongside `opencv-python`. Use PySide6 `Signal`/`Slot`; `pyqtSignal` belongs to PyQt. Document OpenCV/Qt startup smoke results on the selected OS.

P0 produces `requirements-win-cpu.lock` including transitive packages and wheel hashes, and records Python and OS details. Install CPU torch wheels from the official CPU index during setup; do not accidentally resolve a different backend during subsequent dependency installation. Save a matching local wheelhouse if reinstalling without internet is required. A candidate table alone is not a lockfile.

Preflight checks local assets before model construction, fails clearly if missing, and never silently downloads at launch. Exercise first tracker use during setup to catch lazy dependencies. Disable optional telemetry/update/integration behavior and verify startup with network access unavailable. Record dependency and model licenses; YOLO's published licensing includes AGPL-3.0 and enterprise options. Do not relabel third-party assets as MIT. [Ultralytics licensing](https://docs.ultralytics.com/models/yolo11/).

## 5. Repository and interfaces

```text
vertebrate/
  pyproject.toml                   # package, entry point, test configuration
  requirements-win-cpu.lock        # generated after successful P0 setup
  README.md
  config/demo.json                 # frozen calibrated profile
  config/observation-10s.json
  config/bytetrack.yaml
  models/manifest.json             # local asset names, hashes, licenses
  src/vertebrate/
    __main__.py  cli.py  contracts.py  config.py
    capture.py  clocks.py  pipeline.py
    pose.py  tracking.py  smoothing.py  features.py  fsm.py
    context.py  incidents.py  storage.py  recorder.py
    gui/window.py  gui/video.py  gui/inbox.py
    evaluation/runner.py  evaluation/metrics.py  evaluation/labels.py
  tests/unit/  tests/integration/  tests/gui/
  tests/fixtures/                  # synthetic samples and tiny permitted videos
  data/manifests/dev.json
  data/manifests/holdout.json
  data/videos/                     # local, ignored by Git as appropriate
  reports/baseline.md  reports/benchmark.json  reports/evaluation.json
  docs/architecture.md  docs/demo-runbook.md  docs/licenses.md
  runtime/outbox/  runtime/logs/   # ignored by Git
```

Expose planned CLI commands `doctor`, `benchmark`, `evaluate`, and normal GUI launch. Commands in this blueprint are implementation acceptance contracts; they do not exist in the supplied research folder yet. The evaluation runner imports the same feature/FSM modules as the GUI and headless pipeline.

Configuration is parsed into typed immutable dataclasses with explicit range validation. Use JSON to avoid another application parser dependency. GUI edits apply only while stopped, increment a configuration revision, and reset temporal state before resuming. Persist the entire applied config in each incident or an adjacent immutable config file identified by hash.

## 6. Pose and tracking decisions

### Model selection

Start with CPU float32, batch one, input size 512. Benchmark 416, 512, and 640 with YOLO11n-Pose on development clips, then YOLOv8n-Pose only if useful. Warm up, exclude load time from steady-state numbers, but report startup separately. Measure full processing time, freshness, keypoint availability, tracking stability, and fall/ADL behavior for one and two people. Use the smallest model/input that meets both observation quality and release gates.

Keep the initial PyTorch backend if it passes. If it fails, time-box one CPU ONNX Runtime experiment with explicit dependency pins and an output-equivalence test. Adopt it only after the same holdout gates pass; otherwise keep the simpler stack and report the unresolved performance issue. No automatic backend changes during the demo.

MediaPipe is not required. A legacy single-person Pose implementation would need careful crop and history isolation; current Pose Landmarker supports `num_poses`, so the research's blanket “MediaPipe is single-person” statement is incorrect. That capability alone does not provide this application's persistent identity policy. [Google Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

### Tracker integration

Use Ultralytics ByteTrack explicitly, one tracker per source session. Retain row alignment between boxes, IDs, and keypoints after association; empty detections and missing IDs are valid outputs. Configure detection confidence low enough to preserve candidates for ByteTrack's second association stage. Proposed detection floor 0.10, track-high 0.25, track-low 0.10, new-track 0.25, match threshold 0.80 are **starting values, calibrate**. In the pinned implementation, the lower low-score comparison is strict; test scores around the boundary. [Pinned tracker source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.203/ultralytics/trackers/byte_tracker.py).

ByteTrack's buffer ages in tracker updates, not elapsed seconds. Keep its internal buffer conservative, and independently expire application history after 0.75 source-seconds without a matched detection (**starting value, calibrate**). Even if ByteTrack later reuses that numeric ID, assign a new application generation and require reacquisition. Use `(session_id, tracker_id, generation)` as the person key. Never transfer a candidate fall to an unrelated new ID. Predicted boxes may aid association but cannot satisfy movement/stillness gates.

## 7. Feature specification

All geometry uses original-image pixels after undoing any letterboxing. Image y increases downward. A COCO landmark is usable only if finite, inside the frame, and above the configured keypoint confidence. Confidence is model output, not a calibrated probability of correctness.

### Baseline and visibility

Before arming, require a stable full-body upright observation interval with both ankles valid. Estimate median box height `H0`, hip y `Hy0`, and ankle/ground y `Fy0` across this interval; `Fy0` is the median of the two-ankle midpoint's y coordinate. During acquisition, use provisional medians to check torso length and drift, and publish a baseline only after the full interval passes. Update the baseline only while visibly upright and stable. Freeze it on candidate entry. While upright, a height change greater than 20% from the baseline triggers reacquisition; leaving the marked detection zone invalidates the baseline. Do not apply this depth-reset rule to a collapsing candidate's box height. The 20% trigger and every detector threshold in this section are starting values to calibrate.

Required shoulders are COCO 5 and 6; hips 11 and 12. Use both sides for midpoints. Reject a degenerate torso shorter than 0.08 H0. The stillness core requires both shoulders, both hips, and at least two valid knee/ankle landmarks (13–16), at both ends of each motion comparison. No one-sided midpoint substitution during an active event.

### Equations

```text
S = (p5 + p6) / 2                         shoulder midpoint
H = (p11 + p12) / 2                       hip midpoint
theta = degrees(atan2(abs(Sy-Hy), abs(Sx-Hx)))    0° horizontal; 90° upright
r = box_width / box_height
q = box_height / H0
d = (Hy - Hy0) / H0                       net descent in reference body heights
v = slope(t_i, Hy_i) / H0                 downward speed in reference heights/s
g = max(Sy, Hy) >= Fy0 - ground_margin * H0
```

Fit `slope` by least squares over recent reliable samples; reject nonpositive time spans. Use fixed `H0`, not the shrinking current box height. This prevents a box collapse from inflating apparent velocity. The velocity is an image-space normalized measure, never metres/second or impact force.

Use a timestamp-aware exponential filter for display/posture: `alpha = 1-exp(-dt/tau)` and `p_filtered = alpha*p + (1-alpha)*p_previous`. Reset on invalid gaps. Compute rapid descent from short-window valid raw hip midpoints to avoid smoothing away the fall; compute posture from filtered midpoints. Save raw and filtered feature summaries for debugging. A verified existing One Euro filter can replace this after matched regression tests.

For stillness, compare only the same usable landmark indices at times separated by approximately 0.20 seconds; choose a prior observed sample between 0.15 and 0.30 seconds old and use its actual `dt`. Define `m = median_j(norm(p_j(t)-p_j(t-dt)) / (H0*dt))` over the required common core. Also bound hip displacement `b = norm(H(t)-H(t-1.0))/H0`, using a prior sample 0.8–1.2 seconds old and a stable visibility interval. This cumulative check reduces slow-motion false stillness.

Do not interpolate through occlusion to create evidence. A motion estimate with too few common landmarks is unknown. Position gaps, missing poses, bad timestamps, and detector failure reset the uninterrupted stillness interval.

### Starting calibration values

**Every value in this table is a starting value to calibrate on development footage. None is a biological or universal threshold.** Durations use source time.

| Parameter | Start | Meaning |
|---|---:|---|
| Keypoint confidence | 0.35 | Minimum usable landmark confidence |
| Minimum initial person height | 160 px at 720p capture | Reject tiny subjects; scale proportionally with source height |
| Upright acquisition | 1.0 s | Baseline interval, valid upright observations |
| Upright/recovery posture | theta ≥ 60°, q ≥ 0.80 | q applies after baseline exists |
| Baseline stability | hip drift ≤ 0.05 H0 over 1 s | Reacquire if unstable |
| Filter tau | 0.06 s | Light posture smoothing |
| Velocity window | 0.25 s, ≥3 observations spanning ≥0.12 s | Raw hip regression |
| Rapid descent | v ≥ 0.60 H0/s for ≥0.12 s and ≥2 estimates | Persistent descent trigger |
| Initial downward displacement | d ≥ 0.15 | Required alongside speed |
| Completed descent | d ≥ 0.25 | More substantial position change |
| Low torso | theta ≤ 35° | Horizontal evidence |
| Shape evidence | r ≥ 1.20 OR q ≤ 0.60 | At least one additional low-posture cue |
| Ground margin | 0.25 H0 above Fy0 | Near the calibrated local floor |
| Down-state dwell | 0.20 s | Resist one-frame geometry glitches |
| Descent deadline | 1.20 s from candidate entry | Must reach down posture |
| Stillness speed | m ≤ 0.05 H0/s | Limited core landmark movement |
| One-second displacement | b ≤ 0.04 H0 | Bound cumulative drift |
| Stillness duration | 3.0 s demo; 10.0 s alternate | Separate labeled profiles |
| Maximum observation gap | 0.25 s | Beyond this, candidate becomes inconclusive |
| Maximum down verification | 12 s from down entry | Leaves room for the ten-second profile |
| Recovery dwell | 2.0 s | Required to rearm after an alert |
| Application identity expiry | 0.75 s | Discard stale track generation/history |

The ground test is a simple scene prior for the marked local demonstration zone, not a general reconstruction of a floor plane. A camera move invalidates calibration. If the geometry repeatedly fails for a planned demonstration trajectory, adjust the view or openly revise the supported envelope before freezing the test set.

## 8. Deterministic event state machine

Per person, define `D` = rapid speed plus initial displacement; `L` = completed descent AND low torso AND shape evidence AND ground test; `M` = valid stillness speed and one-second displacement checks. All dwell intervals require valid observations. Candidate evidence is latched only inside the current event.

| State | Entry/action | Exit | Deadline/failure |
|---|---|---|---|
| ACQUIRING | New identity or invalidated baseline; gather upright baseline | Upright stable interval → MONITORING | Already down: show informational status, remain unarmed |
| MONITORING | Valid baseline; update it only while stable/upright | D → DESCENT, freeze baseline, store onset/peak evidence | Invalid visibility → ACQUIRING; no alert |
| DESCENT | Rapid descent observed | L continuously for down dwell → VERIFYING_DOWN | Deadline, recovery, or long gap → ACQUIRING with reason |
| VERIFYING_DOWN | Store first down-posture time; wait for complete M history | L AND M → STILLNESS with zero elapsed credit | Loss of L resets to DESCENT only within original descent deadline; otherwise ACQUIRING; 12 s total down deadline |
| STILLNESS | Accumulate only contiguous valid source-time intervals satisfying L AND M | Elapsed ≥ configured stillness → ALERTED | Motion or loss of L → VERIFYING_DOWN and timer zero, applying that state's posture/deadline guards immediately; gap/invalidity resets timer; long gap or down deadline → ACQUIRING |
| ALERTED | Emit exactly one incident, latch incident ID | Upright recovery dwell → ACQUIRING | No timer-based repeat; visibility loss changes display status but does not erase the incident |

Evaluate guards in this order: source reset/error; validity and identity; recovery; deadline; posture/motion; duration completion. An invalid frame at the threshold boundary must not trigger an alert. A short invalid interval under the maximum gap preserves the candidate only within its original deadlines and sends STILLNESS back to VERIFYING_DOWN. It never adds elapsed credit. A timestamp jump larger than the maximum gap has the same effect as a long observation loss, even if the next frame has a confident pose. The one-second displacement history must become valid again before stillness restarts.

Acknowledging the alert only changes operator acknowledgement status. It does not change detection state. If a track generation expires, its successor must obtain a new upright baseline before another fall alert. Manual source reset likewise cannot turn an already-down subject into a newly witnessed fall.

Remove the weighted score. The ordered gates cannot be traded against one another: a highly horizontal person with no observed descent cannot accumulate enough “points” to become a fall. UI evidence is a set of boolean/unknown observations, never an invented percentage confidence. “Impact” is renamed “down posture” because contact force and true impact time are not measured.

## 9. Environmental context

The operator may enter measured or simulated temperature, humidity, observation time, and a location label. Label the source `manual_measured`, `manual_simulated`, or `missing`. Manual measured means operator-reported measurement, not independent verification by this app. Retain old input only with its observation time and a visible stale flag; use a configurable 30-minute freshness limit as a project policy. Do not infer historical clip weather from today's input.

Context is optional and never gates detection. If either input is missing, nonfinite, or RH is outside 0–100%, return null heat index with a reason. For this demonstration, support entered temperature from 20–40°C and RH 10–90%; outside that application envelope retain raw context but return `out_of_demo_range`. This is a conservative product scope choice, not the complete scientific validity domain.

Use Fahrenheit internally. Compute the NWS simple estimate, average it with ambient temperature for the branch test, and apply the Rothfusz polynomial with its humidity adjustments if the branch value is at least 80°F. Use the simple averaged estimate for the lower branch as the explicitly selected implementation convention. Round only at serialization/display. Store the method identifier. There is no blanket 40%-RH gate: the NWS procedure includes low-humidity adjustments. [NWS procedure](https://www.weather.gov/ctp/heat), [NWS polynomial](https://www.weather.gov/media/epz/wxcalc/heatIndex.pdf).

```text
T = 1.8 * temp_c + 32; R = relative_humidity_pct
simple = 0.5 * (T + 61 + 1.2*(T-68) + 0.094*R)
screen = (simple + T) / 2
if screen < 80: result_f = screen
else:
  result_f = -42.379 + 2.04901523*T + 10.14333127*R
             -0.22475541*T*R -0.00683783*T*T -0.05481717*R*R
             +0.00122874*T*T*R +0.00085282*T*R*R
             -0.00000199*T*T*R*R
  if R < 13 and 80 <= T <= 112:
      subtract ((13-R)/4)*sqrt((17-abs(T-95))/17)
  if R > 85 and 80 <= T <= 87:
      add ((R-85)/10)*((87-T)/5)
result_c = (result_f - 32) / 1.8
```

Regression fixtures include 90°F/70% RH → about 105.922°F, 100°F/10% RH → about 94.122°F, and 80°F/90% RH → about 86.342°F. Test conversions, both adjustments, lower branch, invalid input, and missing context. Heat index is not wet-bulb temperature or measured body temperature and does not classify the cause of the observed event.

## 10. Incident schema and local dispatch

Create UUIDv4 incident IDs. Validate required fields, types, finite numbers, relative media paths, enums, and timestamp ordering. Use null with a reason for unknown data. Suggested schema version is `1.0`; future incompatible changes require a version change.

Illustrative confirmed event from an imported clip; numerical kinematics are examples, not measured results:

```json
{
  "schema_version": "1.0",
  "incident_id": "2e44a204-7b9b-4b86-9669-48236a6fe910",
  "event_type": "suspected_fall_with_stillness",
  "mode": "local_demo",
  "session_id": "demo-session-001",
  "source": {"kind": "video", "name": "side_fall_01.mp4", "recorded_start_utc": null},
  "person": {"tracker_id": 4, "generation": 1},
  "timing": {
    "fall_onset_source_s": 12.1,
    "down_posture_source_s": 12.8,
    "stillness_start_source_s": 13.8,
    "confirmed_source_s": 16.8,
    "fall_onset_utc": null,
    "confirmed_at_utc": "2026-09-28T10:30:00Z"
  },
  "location": {"label": "College demo room", "latitude": null, "longitude": null},
  "configuration": {"profile": "demo-3s", "revision": 1, "snapshot_path": "config.json"},
  "kinematics": {
    "peak_downward_velocity_h0_per_s": 0.84,
    "net_hip_descent_h0": 0.36,
    "torso_angle_deg": 18.0,
    "bbox_width_height_ratio": 1.62,
    "bbox_height_h0_ratio": 0.48,
    "stillness_observed_s": 3.0,
    "common_core_keypoints": 8
  },
  "evidence": {"rapid_descent": true, "low_posture": true, "near_floor": true, "sustained_stillness": true},
  "environment": {
    "ambient_temp_c": 32.22,
    "relative_humidity_pct": 70.0,
    "heat_index_c": 41.06,
    "heat_index_method": "nws_rothfusz_adjusted",
    "source": "manual_simulated",
    "applies_to": "demo_scenario",
    "observed_at_utc": null,
    "stale": null
  },
  "media": {"snapshot_path": "snapshot.jpg", "clip_path": null, "clip_status": "not_requested"},
  "delivery": {"destination": "local_inbox", "status": "saved", "external_dispatch": false},
  "acknowledgement": {"status": "unacknowledged", "at_utc": null}
}
```

Store under `runtime/outbox/<incident_id>/`. GUI notification occurs immediately on event confirmation. Writer creates media/config temporary files, renames them when complete, then writes JSON through a temporary file and replacement. Final JSON is the completeness marker. On restart ignore temporary/incomplete records and report them for review. Use `allow_nan=False` for JSON. Only the writer mutates incident files.

Saving the same incident ID again is idempotent. Acknowledgement updates the same incident through the writer. If snapshot saving fails, save metadata with `snapshot_path: null` and an explicit error; preserve the GUI alert. If metadata cannot be saved, show `save_failed`, retain the in-memory incident, and offer retry. Never show “saved” or “delivered” before success. Startup tests must prove the outbox is writable.

**Optional clip:** use a downscaled, time-stamped RAM ring, 320×180 BGR at five sampled frames/second, 30-second retention. This has 25,920,000 bytes of raw pixels before object overhead. Limit all media buffers to 128 MiB. On confirmation, request a clip from three seconds before candidate onset through three seconds after confirmation. Preserve available frames, mark truncation, and store source timestamps. Downsample/duplicate only for the evidence playback encoder, never for detection evidence. Codec unavailable → snapshot plus JSON remains the supported fallback. Two simultaneous incidents must respect the combined memory cap. No routine continuous disk recording.

## 11. GUI and controls

Use a large left video pane, a right status/context column, and an inbox below or in a tab. Prefer readable typography and restrained color. Color is supplementary; every state has text. A serif title can fall back to the system's generic serif family. Keep implementation jargon in a collapsible diagnostics panel.

Required controls: source picker, camera selector, Start/Stop, Pause/Resume for files, Replay, skeleton toggle, manual context editor, profile picker while stopped, and Open incident/Acknowledge. Show a persistent **Local demo — no external emergency dispatch** label.

Display processing FPS separately from camera FPS, source-time elapsed, stillness progress, and data-quality warnings. More than two detected people produces a visible “Outside tested people-count limit” warning; do not silently ignore people or describe the scene as validated. An active candidate gets a non-confirmed “Possible fall — observing” status immediately; the final alert waits for confirmation. This makes the delay understandable without pretending to have confirmed an emergency early.

Preserve frame/overlay sequence alignment. Widget updates run on the Qt main thread. Workers communicate by bounded data/command queues and small queued signals; emitting an event must not expose mutable buffers. Use cooperative cancellation and release the camera in its owning worker. A blocked driver is surfaced as a stop failure; process isolation is a later fallback only if this happens on the actual demo machine.

## 12. Phased implementation plan

Each phase should be implemented as a complete task with its own acceptance evidence. Run commands inside the prepared VERTEBRATE virtual environment. All commands below are **to be implemented**, not claims of existing tests. Normal CLI success returns exit code 0; invalid assets/configuration return nonzero with readable diagnostics. Do not advance past a failed gate by changing reported results.

| Phase | Standalone implementation task and files | Runnable acceptance contract |
|---|---|---|
| P0 — Reproducible foundation, 2–3 days | Create package/CLI, candidate environment, license/model manifest, doctor, and baseline report. Inspect any subsequently supplied legacy code before reusing it. Files: `pyproject.toml`, lock, `cli.py`, `config.py`, `models/manifest.json`, `reports/baseline.md`. | `python -m pip check`; `python -m vertebrate doctor --offline`; `python -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q`. Doctor loads local pose weights, executes a local test inference and tracker assignment, verifies hashes/outbox/Qt import; missing model test fails clearly without a network request. |
| P1 — Source timing and pose, 3–4 days | Implement capture, clocks, frame contracts, model adapter, tracker association, reset rules, raw feature recording, and benchmark. Collect permitted development clips and label source time. Files: `capture.py`, `clocks.py`, `pose.py`, `tracking.py`, `contracts.py`, `evaluation/labels.py`, dev manifest. | `python -m pytest tests/unit/test_clocks.py tests/integration/test_capture_tracking.py -q`; `python -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json`. Test empty frames, two people, ID expiry, letterbox mapping, source switch, and EOF. Record actual performance; do not assert success from model inference time alone. |
| P2 — Deterministic detection, 4–5 days | Implement validity, baseline, features, filtering, FSM, and a headless evaluation path using synthetic trajectories. Files: `smoothing.py`, `features.py`, `fsm.py`, `pipeline.py`, `evaluation/runner.py`. | `python -m pytest tests/unit/test_features.py tests/unit/test_fsm.py tests/integration/test_replay_invariance.py -q`. Assert ordered fall detection, no alert on sit/slow lie/recovery, no timer credit through invalid observations, no stale identity transfer, no EOF stillness, and equal source-time results at different replay processing speeds. |
| P3 — Complete local incident path, 2–3 days | Implement manual context, NWS calculation, incident validator, single writer, local inbox persistence, idempotency, and acknowledgement. Files: `context.py`, `incidents.py`, `storage.py`; fixtures and schema description. | `python -m pytest tests/unit/test_context.py tests/unit/test_incident_schema.py tests/integration/test_dispatch.py -q`. A synthetic confirmed event saves exactly one valid record and decodable snapshot; injected disk/media failures remain visible; retry does not duplicate the incident. |
| P4 — Desktop workflow, 3–4 days | Build source controls, video pane, person cards, context editor, inbox and safe shutdown on the main-thread/worker design. Files: `gui/*`, `__main__.py`, `pipeline.py`. | `python -m pytest tests/gui -q`; `python -m vertebrate --source data/videos/demo.mp4 --config config/demo.json`. Automated GUI smoke uses Qt's own test facilities; manually verify camera start/stop, pause, replay, acknowledgement, source errors, and 20 start/stop cycles. |
| P5 — Calibration and locked evaluation, 4–6 days | Annotate development and held-out sets, tune on development only, freeze model/configuration, implement event matching and metrics, and generate a failure catalogue. Files: manifests, `evaluation/metrics.py`, frozen configs, reports. | `python -m vertebrate evaluate --manifest data/manifests/holdout.json --config config/demo.json --output reports/evaluation.json --fail-on-gates`. Nonzero exit on failed Section 13 release gates. Run the ten-second profile separately on suitable long clips. |
| P6 — Evidence and handoff, 2–3 days | Add optional bounded clips only after gates pass; finalize runbook, offline rehearsal, failure demonstrations, and artifact checks. Files: `recorder.py`, tests, README, runbook, final report. | `python -m pytest -q`; `python -m vertebrate doctor --offline`; repeat the demo three times with networking disabled. Verify every emitted incident/media file, the 30-minute soak, and the identical frozen model/config hashes. |

The estimates assume one student developer familiar with Python; they are planning estimates. If a gate fails, spend the reserved time fixing it, reduce optional scope, and rerun affected checks. Do not begin ONNX, clips, or cosmetic refinements while basic detection and dispatch remain broken.

## 13. Validation, calibration, and release gates

### Development and holdout data

Use URFD as an external source where permissions and camera geometry fit; add consented local recordings for the actual room and camera. MCFD is optional, not a download prerequisite. Public benchmark performance does not establish performance in the demonstration room. [URFD authors' dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), [MCFD authors' technical report](https://www-labs.iro.umontreal.ca/~labimage/Dataset/technicalReport.pdf).

Each manifest row identifies clip, hash, subject/session/source group, camera view, frame timing, eligibility, observed person label, fall-onset interval, down-posture interval, stillness eligibility interval, recovery, and visibility gaps. Human annotation is independent of predictions. Preserve the original video; never append a frozen frame to satisfy the stillness duration.

Prepare at least 15 development fall events plus 20 negative activity sequences. Holdout must contain at least 30 distinct supported fall events, at least eight each of forward/backward/sideways as viewed by the accepted camera, and at least six events with two people visible (these may overlap direction counts). Include at least 60 minutes of negative daily activity, including 30 minutes in the actual demo setup. Include deliberate fast lying down as an ambiguity stress test and report its results separately, not as proof the model can infer intent.

Split by person/session/original recording before threshold tuning. All views, crops, re-encodings, and adjacent segments from one event stay in the same partition. If subject IDs are unavailable, disclose that limitation and group by original recording. Never tune on holdout errors and then continue calling that same set unseen; reserve new footage or report the test as reused.

### Calibration procedure

1. Confirm camera placement, stable full-body visibility, and floor calibration in the marked zone.
2. Compare the candidate model/input sizes using development data; retain one.
3. Plot distributions for stationary pose jitter, rapid sits, bends, slow lies, and falls. Choose the stillness threshold from measured jitter, then check it against slow movement.
4. Tune speed/displacement gates first, low-posture/ground gates second, and stillness/deadlines third. Search a small documented grid around starting values; avoid a large opaque optimizer.
5. Inspect false alerts and missed falls with feature timelines. If cases are visually indistinguishable, record the limitation rather than hiding it behind a new threshold.
6. Freeze config, weights, and code revision, then run holdout exactly once for the headline result. Also perturb temporal/posture thresholds by ±10% on development data to reveal a brittle setting.

### Event metrics

Match each prediction to at most one annotated event of the correct visible person, using an onset within the annotated onset interval expanded by 0.5 seconds and a confirmation no earlier than the ground-truth eligible stillness completion (allow 0.25 seconds annotation tolerance), and no later than two seconds afterward. Lock these matching tolerances before evaluation; they are scoring policy choices. Associate predicted tracks with manually labeled people through annotated boxes/inspection; numeric tracker IDs alone are not ground-truth identities.

Calculate event recall `TP / (TP+FN)`, precision `TP / (TP+FP)`, false alerts per negative video-hour, duplicate count, confirmation latency, pose-valid coverage, and inconclusive fraction. Eligible missed events remain false negatives even when the app abstains. Report excluded/out-of-envelope events and reasons separately. Undefined ratios use null, never a flattering 100%.

Report accuracy with counts and a binomial confidence interval; 27/30 recalled falls is 90% on that sample, not a general guarantee. Zero false alerts in one hour still has an approximate one-sided 95% Poisson upper bound of three per hour. Do not repeat the research's unverified 3.1% figure as this system's result.

### Numeric acceptance gates

These are chosen project release criteria, not measured outcomes. Freeze them before holdout evaluation.

| Area | Required gate |
|---|---|
| Scripted demonstration | Three complete rehearsals; every planned alert and negative scenario behaves as documented |
| Held-out supported falls | Recall ≥90% and precision ≥90%; report exact counts and uncertainty |
| Negative exposure | Zero confirmed false alerts across the specified ≥60 minutes |
| Duplicates | Zero duplicate incident IDs/emissions for a single confirmed episode |
| Two-person scenarios | No evidence transferred between people in the planned test set; report ID switches separately |
| Evidence availability | At least 90% valid core observations during eligible holdout fall/stillness intervals; gaps never fabricate evidence |
| CPU processing | ≥10 processed frames/s average, one and two people, at frozen settings; 15 FPS is a stretch goal |
| Live freshness | p95 acquisition-to-display ≤300 ms; report hardware and 5-minute measurement per people-count condition |
| Observation continuity | p95 processed source-frame interval ≤150 ms; gaps >250 ms produce the specified degraded state |
| Confirmation delivery | p95 confirmation-to-visible-alert ≤500 ms; p95 confirmed incident metadata saved ≤1 s on healthy local disk |
| GUI responsiveness | p95 50-ms timer lateness <100 ms; no >500-ms stall in a five-minute loaded run |
| Lifecycle | 20 start/stop cycles without leaked camera handles or background workers; shutdown ≤3 s on demo hardware |
| Soak | 30 minutes without crash, deadlock, unbounded queues; process RSS ≤2 GB and ≤150 MB growth after warm-up |
| Offline | All rehearsals start with no network; missing assets fail locally with a clear message |
| Fault handling | Missing model, malformed video, camera disconnect, invalid context, full queue, and unwritable disk produce explicit states without fabricated alerts/success |

A slow laptop does not invalidate file analysis, which can run slower than real time. It does fail the live-performance gate. If local optimization cannot meet that gate, demonstrate accurately labeled prerecorded analysis and report the live limitation rather than silently weakening the requirement.

### Regression scenario matrix

| Scenario | Expected result |
|---|---|
| Visible fast fall, remains low and still | One confirmed alert after the profile's observed stillness |
| Quick sit, crouch, bend, slow controlled lie | No confirmed fall alert in supported setup |
| Trip then stand before confirmation | Candidate cancels; no incident |
| Person begins video already prone | Informational status, no newly observed fall |
| Fall then continued rolling/movement | Candidate visible; no stillness confirmation until criteria are actually met within deadline |
| One of two people falls | Correct person's incident; other's baseline/state stays independent |
| Brief occlusion during stillness | Timer resets; no credit for unobserved time |
| Long occlusion or exit | Candidate inconclusive; history eventually expires |
| Pause for 20 wall-clock seconds | Zero source-time/evidence advance |
| Seek, replay, source switch | New session and fresh baselines |
| EOF during verification | Inconclusive; never synthetic continuation |
| Detector returns empty result or NaNs | Invalid observation; no stillness credit |
| Deliberate rapid lie mimics fall | Record potential false alert/ambiguity; do not claim intent recognition |
| Storage failure | GUI incident survives with failure status; no false delivery claim |

## 14. Demo runbook and README

**Prepare:** install the frozen environment; copy model and allowed videos; run doctor; record hashes; connect the tested webcam; calibrate the marked clear area; verify outbox storage. Enter environmental values explicitly as measured or simulated. Use prerecorded falls or safe supported acting; do not perform uncontrolled falls for the demo.

**Launch:** activate the environment and run `python -m vertebrate`. Select the three-second profile and explain the stillness delay. Show a daily-activity clip, a recovery clip, then a supported fall clip. Open the resulting incident, inspect timestamps/context/snapshot, and acknowledge it. Show the two-person case. Disconnect networking and repeat the complete path. If a webcam live scenario is planned, demonstrate the same frozen configuration live after the recorded sequence.

**Demonstrate failure honestly:** obscure pose visibility or stop the video before stillness completes, show the inconclusive state, and explain why missing observations cannot prove immobility. Show manual context as simulated if no measurement was taken. The ten-second profile gets a separate sufficiently long clip.

**Afterward:** stop capture, inspect saved evidence, and delete the demo's incident data when no longer needed. Do not commit participant video, secrets, or private records. Keep permissions/notices with public datasets and obtain participant agreement for local recordings. No legal-compliance certification is claimed by storing data locally.

README contains install/launch steps, exact tested hardware/OS and package lock, model provenance/licenses, input envelope, profile definitions, calibration method, observed metrics with counts, known failures, offline preparation, troubleshooting, and the distinction between simulated dispatch and an external EMS connection. The local inbox is the completed demo workflow.

## 15. Principal risks and fallbacks

| Risk | Response and fallback |
|---|---|
| Pose fails on low/occluded bodies | Improve lighting/view; benchmark larger input or the alternate nano model; report uncovered cases and retain abstention |
| CPU cannot keep up | Profile entire pipeline; reduce input size or oversubscribed thread counts; try one measured ONNX experiment; preserve honest file-analysis fallback |
| ADL and falls overlap | Calibrate ordered sequence and floor evidence; retain ambiguity examples; never claim medical/intent diagnosis |
| Identity switches | Fixed camera, two-person envelope, per-generation histories and expiry; no cross-ID recovery heuristics until tested |
| GUI/storage failure | Bounded queues, independent writer, cooperative stop, visible failure/retry; snapshots remain the clip-codec fallback |
| Setup or evidence unavailable | Offline asset preflight, frozen successful environment, matching local wheels and consented sample videos; no “works offline” claim from code inspection alone |

## 16. Changes resulting from validation

Compared with the supplied Claude draft and this package's v1, v2: bounds the observable goal; makes existing-code reuse conditional; removes external services and deployment infrastructure; picks one initial pose/backend path; corrects MediaPipe and Qt API claims; adds a separate writer; separates live/file queue policy and clocks; defines track generations and update-count limitations; freezes pre-fall scale; adds visibility, floor, and cumulative-motion checks; replaces the score with ordered gates; defines every timer reset and recovery; versions config/context; separates event and analysis timestamps; makes incident persistence idempotent and fault-visible; bounds media memory; adds candidate version evidence without claiming tested compatibility; moves a complete incident path before GUI polish; and defines held-out event metrics, leakage controls, release gates, offline rehearsal, and residual limitations.

The detailed issue-to-correction mapping and external evidence are in `02-validation-report.md`. Implement this v2 as the current contract, and update it only alongside measured evidence and changed acceptance tests.
