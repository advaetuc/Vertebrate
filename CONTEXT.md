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

Confirmation is **per person** (`app_modules/tracking.py`). `PersonTracker`
(pure NumPy, no tracking package) links each detection to a person ID across
frames: greedy matching by box IoU ≥ `0.2`, or, because a falling person's box
changes shape, a centre shift ≤ `0.35` of the larger box diagonal (best IoU
first, then nearest centre). Each person has their own streak: a processed
frame where that person matches adds one (capped at five); a frame where they
are detected and do not match resets it to zero. A frame where they are not
detected leaves the streak unchanged; after `TRACK_MAX_MISSES` = 15 missed
frames the ID is dropped, and a returning person gets a new ID. A person is
**confirmed** at five consecutive fallen frames and stays confirmed through up
to `ALERT_HOLD_MISSES` = 3 missed frames. The alert is on while at least one
person is confirmed, so fallen frames from different people never add up.
Five frames are not a fixed duration because model and camera rates vary. IDs
restart with each camera session and identify a track, not who the person is.

This detects posture, not fall causality. Intentional lying down, rotated or
overhead cameras, occlusion, proportions, and pose errors can affect it. A
match does not establish injury or need for help. The UI says **“Local demo —
no external emergency dispatch.”**

---

## 2. SYSTEM ARCHITECTURE & DATA FLOW

### High-level architecture

A local desktop monolith with an event-driven Qt thread boundary. One `QThread`
owns the OpenCV camera, YOLO model, inference, geometry, annotation, and
per-person tracking. The Qt main thread owns `QMainWindow` and all widgets. The worker emits
a detached `QImage`, the alert state, the leading person's streak and ID, the
number of people detected and the confirmed person IDs by Qt signal; queued delivery runs
the receiving slot on the main thread. There is no database, authentication,
microservice, or cloud-processing tier.

| Responsibility | Code | State |
|---|---|---|
| Geometry | `fallen_geometry()` | None; frame-local |
| Annotation | `annotate()` | Current mutable frame only |
| Per-person tracking | `tracking.py`: `PersonTracker`, `Track`, `box_iou()`, `center_shift()` | IDs, per-person streaks, worker-owned |
| Camera/inference | `CameraThread.run()` | Capture, model, tracker, pending-frame event |
| Desktop UI | `MainWindow`, `VideoView`, `EvidenceMeter`, `StatTile` | UI state, timers, worker lifecycle |

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

### Fall analysis groundwork (phase 0)

`app_modules/fallanalysis.py` holds the base for the planned fall analysis.
`PersonTracker.update(boxes, fallen, keypoints, states, timestamp)` appends each
matched person's keypoints to their `Track.history` (`PoseHistory`, a 5 s ring
buffer by time, 8 s since phase 3, capped at 480 samples). A missed frame adds no sample; it only
trims old ones. `PoseHistory.normalized()` gives positions in torso lengths from
the hip centre at the start of the window (image y points down, so a fall is y
increasing); the body scale is the median torso length of standing frames.
`annotate(..., timestamp)` passes the camera thread's `time.monotonic()` frame
time. The camera thread's `FrameRateMeter` measures the rate the detector
processes, which can be higher than the rate the window displays; it is emitted
by `CameraThread.stats_ready(fps, model_ms, history_frames)` and shown in the
Frames / second tile. Calling `update(boxes, fallen)` without a timestamp keeps
the old behaviour and records no history. The post-fall watch (below) is the
first user of this history.

### After a fall (phase 1)

`fallanalysis.PostFallWatch` starts on a track when its fall is confirmed (again,
if the person had got up) and is updated by `PersonTracker` every frame.
`movement_score()` compares the average pose of the last 0.5 s with the average
pose 1 s earlier, in torso lengths, and takes the second-largest joint movement
(one jittery joint is ignored). Movement = score ≥ 0.15 on 2 frames in a row.
Still time grows only on frames where the person is seen and judged (gaps
capped at 0.5 s), so being hidden never counts as lying still. Still ≥ 15 s =
critical. Got up = posture "standing" for ≥ 1 s with the hips ≥ 0.8 torso
lengths above their height at confirmation (if hips are visible). The watch is
kept 10 s after recovery, then removed. `PersonTracker.aftermath()` gives
`{person ID: report}` (status `watching` / `moving` / `still` / `recovered`,
still_s, longest_still_s, moved, critical, moved_after_critical, down_s); it is
sent as the last argument of `frame_ready`. In the GUI, `FallLog.note_aftermath()`
records it for the people in the ongoing event and returns `critical` /
`moving_again` once each; `MainWindow` then sends the `critical` Discord alert
(new kind, colour `COLOR_CRITICAL`, always on when alerts are on) or a `fall`
update. A fall whose people got up (and none is lying still) ends as
"Got up" and sends a Recovered message. Log fields added: `after_fall`,
`longest_still_seconds`, `no_movement_alert_at`. Thresholds were checked on
synthetic poses with keypoint wobble (still person far away: worst score 0.14),
not yet on real footage; tune them in `fallanalysis.py`. Video labels must stay
ASCII (OpenCV's Hershey font).

### Sudden vs gradual and priority (phase 3)

`fallanalysis.descent_analysis(history)` uses the hip centre's y over the pose
history (8 s): standing level = median of the highest standing-frame hip
positions (within 0.3 torso of the top), floor level = median of the last 3
frames. Drop < 0.6 torso lengths, no standing frame, no complete 10 %→90 %
crossing, or a frame gap > 0.75 s during it → `unclear` with a reason. Otherwise
the last complete crossing is timed with interpolation and divided by
`DESCENT_SHARE` 0.65 (accelerating/decelerating falls spend ~65 % of the time in
the middle 80 % of the drop). Sudden < 1.5 s; borderline 1–2 s. Checked on
synthetic falls of 0.4–5 s at 8 and 15 fps (within ~10 %). `PostFallWatch`
measures at confirmation and re-measures for `TYPE_SETTLE_S` 1.5 s, then fixes
it (`type_final`). `priority_of(report)`: recovered → `recovered`; still (or
critical and not moving now) → `critical`; gradual and not borderline →
`check`; otherwise `urgent`. Reports add `fall_type`, `descent_s`, `borderline`,
`type_note`, `type_final`, `priority`. `PRIORITY_STYLE` maps priority → emoji,
word, Discord colour, window colour. `FallLog` keeps per-person types, the
highest priority reached (`priority` log field; recovered excluded) and the
current one (`priority_now()`); `types_final()` releases the first alert.
`MainWindow` holds the first Discord alert in `_start_pending` until the types
are final or 1.5 s passed; `_send_caregiver()` flushes it before any other
message and takes a `priority` for the embed colour (`caregiver_embed(color=)`).
The banner colour and title follow `priority_now()`. Log fields added:
`fall_type`, `descent_seconds`, `priority`; Fall log window columns Priority
(coloured) and Type. Known limits: at very low frame rates a fast fall can
change the tracker ID, giving `unclear`; falls from sitting are `unclear`.

### Patterns of a sudden fall (phase 5A)

`fallanalysis.fall_pattern(history, descent)` runs only for `sudden` falls
(gradual falls: `gradual_pattern()`, phase 5B). Window: hips' 10 % time − 0.4 s to 90 %
time + 0.6 s (`descent` now carries `t_start` / `t_end`). First contact: for
each part (`PARTS`: hips, knees, hands = wrists, upper = nose + shoulders) its
own start (median of the last 4 standing frames) and end (last 3 frames) y; a
part must drop ≥ 0.4 torso; it "landed" when it covered 85 % of its own drop
(perspective-independent). Parts within 0.1 s tie; hands in a tie = hands
first; hips+knees tie = no order. Arms: wrist distance from the shoulder–hip
*segment* ≥ 0.8 torso on ≥ 2 frames = reached out (overhead arms count);
judged only if wrists are visible on half the frames. Direction (profile only):
facing = nose x − shoulder x (≥ 0.15 torso); head's final x vs hips' start x
(≥ 0.8 torso) on the facing side = forward. Rules, in order: arms False and
first knees/upper → collapse; hands first, or knees first with arms, or arms
and forward → trip; hips first or backward → slip; arms False and hips+knees
tie → collapse; else unclear. `priority_of`: collapse and not moving → critical.
`PostFallWatch` ignores movement for `LANDING_S` 1.5 s after confirmation (the
landing itself), so a collapse is not "moving" at once. Report adds `pattern`,
`first_contact`, `direction`, `arms`; `fall_type_text()` and `pattern_detail()`
build the text; `PATTERN_TEXT` holds label and meaning. Log fields added:
`fall_pattern`, `pattern_signs`. Checked on keyframed stick-figure slips, trips
and collapses (profile, mirrored, frontal; 8/12/15 fps): all correct with
1.5 px wobble; with 3 px wobble on a small person 22/27 right, 4 unclear,
1 collapse read as trip. Not yet checked on real footage.

### Patterns of a gradual fall (phase 5B)

`fallanalysis.gradual_pattern(history, descent)` (called by `fall_pattern()`
for `gradual` falls) looks at `t_start − 3 s … t_end + 0.5 s`. Hand to chest:
chest spot = shoulders + 0.35 × (hips − shoulders); a wrist within 0.45 torso;
longest hold (gaps ≤ 0.3 s bridged) ≥ 1.0 s → `chest` (judged if wrists are
visible on half the frames). Unsteady: hip x before the descent, 3-frame
average, zigzag count with 0.2-torso steps ≥ 2 reversals (`sway`; walking one
way gives 0), or an arm out (≥ 0.8 torso from the shoulder–hip segment) for
≥ 0.5 s before half-way down (`reaching`). Controlled: hips' smoothed descent
speed never above 1.6 torso/s (`smooth`), or hips resting ≥ 0.6 s part-way down
(`sat_first`). Order: chest → unsteady → controlled → unclear. `priority_of`:
chest → critical (even if moving); controlled and not borderline → check;
everything else → urgent (gradual-unclear was check in phase 3). Report adds
`chest_s`, `sway`, `reaching`, `sat_first`, `smooth`; `pattern_detail()` turns
them into "Signs". Video label for chest is `(hand to chest)` (`VIDEO_LABEL`).
Checked on keyframed hand-to-chest, swaying/arms-out, smooth sink, sit-on-a-
chair-first, plain sink and walk-then-sink falls (profile, mirrored, frontal;
5–15 fps; up to 3 px wobble on a small person): 36/36 and 54/54 right. Known
limit: a phone or cup held at chest height reads as a hand on the chest.

### People around a fallen person (phase 2)

`PostFallWatch.update_bystanders()` runs each frame for every watched (not
recovered) person, with every other person seen in that frame. Distance =
floor point to floor point (bottom centre of each box) divided by the fallen
person's torso length; torso sizes differing more than 2× (different depth)
are skipped. A new bystander must be `standing` when first seen. Nearby ≤ 6,
with them ≤ 2.5 for ≥ 2 s (recorded in `helpers` with `helped_at`), coming
over = distance dropped ≥ 1 within 2 s. `no_response` = someone nearby for
≥ 20 s since first nearby, nobody with them and nobody coming over. When a
kneeling helper gets a new tracker ID (box shape change), a non-standing new
person within 1.5 torso lengths of where a bystander vanished ≤ 1.5 s ago takes
over that entry. `PersonTracker.helper_ids()` = bystanders currently nearby;
in `update()`, such a person's fallen posture does not grow their streak while
`torso_upright()` (shoulder→hip line within ~45° of vertical) — `Track.helping`,
labelled "Helping" in green. A helper lying down counts as a fall. Reports add
`nearby`, `with_them`, `approaching`, `helpers`, `help_after_s`, `ever_nearby`,
`no_response`. `FallLog.note_aftermath()` returns `helper` / `no_response` once
per event; `MainWindow` sends "Someone is with them" (no ping) and "Nobody has
responded" (ping) as `fall` messages, and adds the help sentence to the fall,
alarm and critical messages and the banner. Log fields added: `people_nearby`,
`helped_by`, `help_after_seconds`; Fall log window column "Help". Known limits:
2D image only (depth guessed from body size); a helper touching the person can
mix keypoints, which may read as movement; helpers who were never seen standing
are not recognised.

### Data architecture and state

There is no application database or user record schema. The app does not save
snapshots or webcam clips. The only persistent record is the per-device fall
log CSV (see "Fall log" below).

| Data | Lifetime/location | Notes |
|---|---|---|
| Webcam frame | Worker memory | BGR NumPy array, annotated in place |
| Model result | One inference iteration | Boxes and `keypoints.xy` only |
| Person tracks | Worker session | ID, last box, streak (cap 5), missed frames; dropped after 15 misses |
| Pose history | Worker session, per track | Last 5 s of keypoints (NaN = not visible), box and posture state; `fallanalysis.PoseHistory`, read in torso lengths |
| Post-fall watch | Worker session, per track | `Track.watch` (`PostFallWatch`): moving / still / got up after a confirmed fall, plus bystanders (nearby / coming over / with them); reported each frame |
| Frame rate | Worker session | `FrameRateMeter`: detector frames/s, model ms; `stats_ready` signal once a second |
| Performance log | `runtime/logs/performance/<device>.csv` | One row per camera session: median/slowest fps, model ms, frame size, system; ignored by Git |
| Display image | One pending signal | RGB QImage detached with `copy()` |
| Display pixmap | Main window memory | Replaced on delivery and scaled on resize |
| Model weights | Local filesystem | Root checked first, then `models/` |
| Ultralytics settings/cache | `runtime/logs/standalone-yolo/` | Local and ignored by Git |
| URFD footage | `data/videos/urfd/` | Local clips and ignored by Git |
| Fall log | `runtime/logs/falls/<device>.csv` | One row per finished fall event; ignored by Git |
| Alarm delay | `QSettings` ("Vertebrate") | Seconds before the alarm rings; 0 = off |
| Alarm sound | `runtime/sounds/alarm.wav` | Generated siren; ignored by Git |

### Fall log

`FallLog` (GUI thread, fed from `MainWindow.show_frame()`) turns the per-person
confirmed alert into events. An event **starts** when the alert turns on (i.e. after the
five-frame per-person check, so slightly after the posture began) and **ends** when the
alert has stayed off for `END_GRACE_S` = 2 s; the end time is when it first
went off, so short flickers do not split one fall. Stopping the camera, a worker
error, or closing the app ends an ongoing event with that reason. Each finished
event appends one CSV row: `device, fall_detected_at, ended_at, utc_offset,
duration, duration_seconds, latitude, longitude, accuracy_m, location_source,
ended_by, alarm_after_seconds, alarm_rang_at, alarm_silenced_at, person_ids`
(the IDs confirmed during the event, `;`-separated, per camera session). Logs written
by an older version are rewritten once with the new columns (rows kept,
new cells empty). Location is the `LocationFeed` fix at the start (or the first fix
during the event). The device name is the host name (`QSysInfo`), and each
device writes its own file. The file is UTF-8 with BOM so Excel shows symbols.
The **Fall log** button opens a non-modal `FallLogDialog` listing this device's
events newest first, with an ongoing fall shown live on top, totals, and a
button that opens the log folder. Write/read failures are shown in that window
and printed to stderr; they never stop monitoring.

### User interface

Fusion style with a dark palette (`apply_dark_theme()`) so Windows and macOS
look the same; Qt 6.8+ also gets dark native title bars. Colours are defined
once as constants above `STYLESHEET`: one teal-to-sky accent for actions, and
green / orange / red reserved for clear / fall / alarm. Typography uses the
platform UI font, with the fixed-width system font for numbers.

- **Header:** logo mark, product name, device chip, live clock, state pill.
- **Main column:** an escalating banner (orange "Fall detected" with countdown;
  pulsing red "EMERGENCY ALARM" with a white **Silence alarm** button), the
  live view (`VideoView`: rounded, state-coloured border, LIVE tag,
  placeholder with a camera glyph when off), and Start monitoring / Stop /
  Fall log buttons.
- **Sidebar (scrolls on small windows):** Status card (tinted by state), Fall
  evidence meter (the leading person's five-frame streak and ID, sent by the
  worker; each box on the video is labelled with its person ID), Live stats (people in view, frames per second, falls
  today, alarm countdown), Location, and the Emergency alarm setting.

`MainWindow.set_state()` drives everything from one state: idle, starting,
monitoring, fall, alarm, stopping, error. Stylesheets change only when the
state (or the alarm pulse) changes; per-frame updates touch text only.

### Emergency alarm

The **Ring if fallen for** stepper (sidebar "Emergency alarm" card) (0–600 s, 0 = Off, default 30 s) is saved
with `QSettings` per user and device. `MainWindow.check_alarm()` runs every
250 ms while the camera is on and on each displayed frame: when the ongoing
`FallLog` event has lasted the chosen delay (counted from the fall alert, so
after the five-frame per-person check), `AlarmSound` loops a two-tone siren on this
device's speakers, the window comes to the front, the banner shows
"EMERGENCY ALARM · fallen for …", and a **Silence alarm** button appears.
Before that, the banner counts down "alarm in N s". The alarm rings at most
once per fall and stops when the fall event ends (posture cleared for 2 s),
when Silence is pressed, when the delay is set to Off, or when the camera stops.
Ring and silence times are written to the fall log. The siren WAV is generated
with NumPy into `runtime/sounds/alarm.wav` on first use (Git-ignored); without
Qt Multimedia, repeated system beeps are used. App volume is 100%, but the
device's system volume and mute still apply. The alarm is local only; there is
still no external dispatch.

`models/manifest.json` records model name/path, SHA-256, size, source, version,
task, and license. At startup, `app.py` checks only that the selected file
exists and is non-empty. It does **not** read the manifest or verify the hash.
Never claim launch-time integrity checking.

### Critical runtime flow

1. The user clicks **Start monitoring**.
2. The GUI creates/starts `CameraThread`, disables Start, enables Stop.
3. The worker calls `local_weights()`. Missing weights produce a local error
   before `YOLO` is constructed; no download fallback is intended.
4. The worker selects the local Ultralytics config directory, disables `sync`
   and `hub` settings in memory, constructs a pose model, and opens
   `cv2.VideoCapture(0)`.
5. Each camera frame runs YOLO on CPU. `fallen_geometry()` evaluates the two
   predicates for each person. `annotate()` draws box and visible skeleton red
   for matches and green otherwise.
6. The tracker updates once per processed frame. The annotated BGR frame becomes
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
Weights must be present locally. There is no upload or remote inference.
Webcam frames stay in process memory. Package installation or a manual weights
download is separate setup and may require a network connection.

The one outbound connection is the **optional, user-enabled caregiver alert**
to Discord (below). It is off until set up, sends text only (device name,
times, duration, location and a Google Maps link; never images or video), and
failures never affect detection, logging or the local alarm.

### Caregiver alerts (Discord)

`app_modules/caregivers.py`. `DiscordNotifier` (a `QObject` owning one daemon
thread and a queue) posts JSON to Discord with `urllib` from the standard
library, using certifi's CA bundle when present (python.org Python on macOS
has no system CA store for urllib). No Discord package, so behaviour is
identical on Windows and macOS. Two modes: a channel **webhook URL**, or a
**bot token + channel ID** (REST `POST /api/v10/channels/{id}/messages`,
header `Authorization: Bot …`). Every request sends a `User-Agent`, which
Discord requires. Up to three attempts; HTTP 429 waits for `retry_after`
(max 10 s); 401/403/404 fail at once with a readable reason. Each outcome
returns to the GUI thread through the `result` signal and is shown in the
sidebar's **Caregiver alerts** card.

`MainWindow` sends: **fall** when a `FallLog` event starts (orange embed naming
the person, e.g. “Person #2”, with the configured ping), **alarm** when the siren starts (red embed, ping),
**end** when the event is written (green embed with duration, how it ended and
alarm ring/silence times, from `FallLog.last_finished`; no ping).
`allowed_mentions` only allows pings when the user configured one. Settings
(enabled, mode, webhook, token, channel, mention, per-event switches) are
stored with `QSettings` group `caregivers` (per user, outside the repository;
the Windows registry's string booleans are handled). `CaregiverDialog` edits
them and has **Send test message**, which uses the unsaved values. If
`runtime/config/discord_webhook.txt` (Git-ignored) holds a valid webhook URL,
`load_caregiver_config()` returns a **fixed** config instead: always enabled,
all events, no ping; the dialog is not used and the sidebar button sends a
test. Every queued, skipped or failed alert prints a `[caregivers] …` line to
stderr with the reason. On close the
app waits at most 3 s for queued alerts.

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
shown. Sources start with **Start monitoring** (the system source after a one-time
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
standalone functions; `PersonTracker` owns IDs and streaks; `CameraThread` owns blocking
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
    frame_ready = Signal(QImage, bool, int, int)  # frame, alert, evidence, people
    error = Signal(str)

    # After inference and BGR -> RGB conversion:
    image = QImage(rgb.data, width, height, rgb.strides[0],
                   QImage.Format.Format_RGB888).copy()
    self.frame_ready.emit(image, is_fallen, evidence, leader_id, people, confirmed)

worker.frame_ready.connect(window.show_frame, Qt.ConnectionType.QueuedConnection)
~~~

Display backpressure must not skip inference or change tracking. Create a new
`PersonTracker` per worker session. Emit readable errors and release owned resources in
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
not verify `app.py`. For app changes, add focused geometry/tracker tests and
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
3. Preserve exact predicates and per-person confirmation: ratio `>1.2`, spine
   `shoulders_y > hips_y + 20`, five consecutive processed positives of the
   SAME tracked person, reset on that person's processed negative, streak kept
   on a missed detection. If user requests changed behavior, update this file
   with code and document the semantics.
4. Preserve thread ownership. Never update widgets from `CameraThread`, share a
   NumPy-backed live QImage, create/release the camera on the GUI thread, or
   destroy an active QThread.
5. Preserve offline behavior. Check local weights before model construction,
   keep autoinstall/telemetry disabled, and do not implicitly add network
   dependencies, uploads, credentials, or remote services. The Discord
   caregiver alert is the one explicit, opt-in exception: keep it off by
   default, text-only, and never in the detection path.
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
- [ ] Geometry, invalid joints, per-person tracking, and alert semantics match request and
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

- Caregiver alerts need an internet connection and a working Discord
  webhook/bot; an alert that fails after three attempts is not resent later.
  Discord credentials are stored in plain text in the user's QSettings.
- Geometry detects posture and may classify ordinary lying down as fall-like.
- Fixed 20-pixel margin changes meaning with image scale and subject distance.
- Tracking is frame-to-frame box matching only (no appearance model): two
  people who overlap heavily or cross can swap IDs, and someone hidden for
  more than 15 frames returns with a new ID.
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

