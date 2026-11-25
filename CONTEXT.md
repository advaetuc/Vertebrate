# VERTEBRATE — System Context and Agent Maintenance Guide

**Role:** Repository-specific source of truth for engineers and coding agents.  
**Active application:** `app.py` (launcher) + the `app_modules/` package.  
**Reconciled:** 2026-11-25, against `main` at `30308cf`.  
**Handoff state:** see `HANDOFF.md` (status, open items, next steps).

This document describes the inspected workspace. It distinguishes active code
from archived code and planned work. If this document and the implementation
disagree, inspect the implementation and update this file in the same change.
Historical prompts, reports, and claims are not proof of current behavior.

---

## 1. EXECUTIVE PROJECT SUMMARY

### Core project idea

VERTEBRATE is a local desktop fall monitor for **Windows and macOS**. It uses
Ultralytics YOLO11n pose estimation on a live webcam, confirms a fall per
person, then analyses it: how the person fell (sudden or gradual, and the
pattern), whether they move afterwards, and who is around them. The result is a
priority (Critical / Urgent / Check / Recovered) shown in the window, written
to a per-device fall log, and optionally sent to caregivers on Discord. A local
siren can ring after a user-set delay.

Everything runs offline on the CPU; webcam frames never leave process memory.
The only optional network use is text alerts to a Discord channel. The intended
audience is developers and demonstrators evaluating a vision prototype; it is
**not a medical device** and its patterns are not diagnoses.

`app.py` only applies the theme and opens `MainWindow`; all code lives in
`app_modules/`. The former tracker-based state machine, schemas, storage,
evaluation pipeline and GUI were removed and preserved in a ZIP under
`.archive/`. They are not runtime dependencies.

### Business logic and domain rules

**Fallen posture.** For each detected person in each processed frame,
`posture_state()` (`common.py`) returns `too_close`, `fallen` or `standing`:

1. **Too close** (checked first, never counts as a fall): the box covers more
   than `TOO_CLOSE_AREA` = 0.45 of the frame, or spans the full frame height
   (top and bottom within `EDGE_PX` = 4 px of the edges).
2. **Fallen** if either predicate holds (`fallen_geometry()`):
   - box width / height > `1.2` (positive height) **and** the lower body is
     visible (any hip and any knee or ankle), so a head-and-shoulders close-up
     is not "wide";
   - mean image Y of shoulders 5, 6 > mean image Y of hips 11, 12 + `20` px.
3. Otherwise **standing**.

Image Y increases downwards. A missing, non-finite or zero joint disables the
spine predicate for that person. There is no separate keypoint-confidence
cutoff (YOLO reports undetected joints as 0, 0).

**Per-person confirmation** (`tracking.py`). `PersonTracker` (pure NumPy)
links each detection to a person ID across frames: greedy matching by box IoU ≥
`0.2`, or a centre shift ≤ `0.35` of the larger box diagonal (best IoU first,
then nearest centre). Each person has their own streak: a fallen frame adds one
(capped at five), a detected non-fallen frame resets it, a missed frame leaves
it. After `TRACK_MAX_MISSES` = 15 missed frames the ID is dropped. A person is
**confirmed** at five consecutive fallen frames and stays confirmed through up
to `ALERT_HOLD_MISSES` = 3 missed frames. The alert is on while at least one
person is confirmed; fallen frames from different people never add up. IDs
restart with each camera session and identify a track, not who the person is.

**Helper exemption.** A person who came near a fallen person while standing
(a bystander) and now kneels or crouches beside them does not grow a fall
streak while their torso stays upright (`Track.helping`, label "Helping"). A
helper who ends up lying down is counted as a fall as usual.

**Fall analysis** (`fallanalysis.py`, details in section 2): movement after the
fall, people around, sudden vs gradual, the fall pattern, and the priority.

This detects and describes posture, not its cause. Lying down on purpose,
rotated or overhead cameras, occlusion, proportions and pose errors affect it.
Every pattern is camera analysis of a 2D image; the alerts say "not a medical
diagnosis". The thresholds were checked on synthetic stick-figure falls only.

---

## 2. SYSTEM ARCHITECTURE & DATA FLOW

### High-level architecture

A local desktop monolith with three threads and Qt signal boundaries:

- **Camera thread** (`CameraThread`, a `QThread`): owns the OpenCV camera, the
  YOLO model, inference, posture, annotation, per-person tracking and the whole
  fall analysis.
- **GUI thread**: owns `MainWindow` and every widget, the fall log, the alarm,
  the location feed and the first-alert timing.
- **Discord sender thread** (`DiscordNotifier`, a daemon thread with a queue):
  posts alerts so a slow network never blocks the window.

The camera thread emits `frame_ready(QImage, bool, int, int, int, list, object)`:
the detached annotated frame, the alert state, the leading person's streak and
ID, the number of people, the confirmed person IDs, and `tracker.aftermath()`
— `{person ID: report}` for everyone watched after a fall. **The last argument
must stay `object`** (see "Hand-off of the analysis to the window"). Queued
delivery runs `MainWindow.show_frame()` on the GUI thread. `stats_ready(float,
float, int)` sends the frame rate once a second; `error(str)` reports failures.
There is no database, authentication, microservice or cloud tier.

| Responsibility | Code | State |
|---|---|---|
| Posture rules | `common.py`: `posture_state()`, `fallen_geometry()`, `too_close()` | None; frame-local |
| Annotation (video labels) | `common.py`: `annotate()`, `watch_label()`, `bystander_label()` | Current frame only |
| Per-person tracking | `tracking.py`: `PersonTracker`, `Track` | IDs, streaks, pose history, post-fall watch; worker-owned |
| Fall analysis | `fallanalysis.py`: `PoseHistory`, `PostFallWatch`, `descent_analysis()`, `fall_pattern()`, `gradual_pattern()`, `priority_of()` | Per track, worker-owned |
| Camera/inference | `camerathread.py`: `CameraThread.run()` | Capture, model, tracker, frame-rate meter |
| Fall log | `falllog.py`: `FallLog`; `falllogdialog.py`: `FallLogDialog` | Ongoing event; CSV per device |
| Alarm | `alarmsound.py`: `AlarmSound`; `MainWindow.check_alarm()` | Ringing flag |
| Location | `locationfeed.py`: `LocationFeed` | Latest fix per source |
| Caregiver alerts | `caregivers.py`: `DiscordNotifier`, `CaregiverDialog` | Queue, settings |
| Desktop UI | `mainwindow.py`: `MainWindow`; `videoview.py`, `urgencymeter.py`, `stattile.py`, `logomark.py` | UI state, timers, worker lifecycle |

The model and camera are constructed in `CameraThread.run()` and the camera is
released in its `finally` block. Stop uses `requestInterruption()`, observed
between blocking reads/inference; after five seconds the UI shows a wait
message and never forcibly terminates the thread.

### Core technology stack

`requirements.txt` pins direct Windows x64 CPU dependencies;
`requirements_macOS.txt` the same for macOS (PyTorch unpinned there).
`requirements-win-cpu.lock` is the 48-distribution hash lock retained from the
former full project environment and includes packages no longer required.

| Technology | Version | Role |
|---|---|---|
| Python | 3.12 | Language/runtime |
| Ultralytics | 8.3.203 | Local model loading and pose inference |
| YOLO11n pose weights | v8.3.0 asset | Person boxes and 17 COCO keypoints |
| PyTorch / torchvision | 2.8.0+cpu / 0.23.0+cpu (Windows) | CPU inference |
| NumPy | 2.2.6 | Geometry, tracking, analysis |
| opencv-python | 4.12.0.88 | Camera, drawing, color conversion |
| PySide6 / Qt | 6.9.2 | Window, QThread, signals, timers, positioning, serial, multimedia |

Inference explicitly uses the CPU. Install `opencv-python` only, never headless
or contrib variants alongside it. Measured on a MacBook Air (25 Nov 2026):
15 fps median, about 47 ms per frame, 1280×720.

### Fall analysis

All of it runs in the camera thread, per tracked person, from that person's
pose history. Distances are in **torso lengths** (shoulder centre to hip
centre) so near and far people compare. Every threshold is a named constant at
the top of its section in `fallanalysis.py`.

**Pose history (phase 0).** `PersonTracker.update(boxes, fallen, keypoints,
states, timestamp)` appends each matched person's keypoints (NaN = not
visible), box and posture state to `Track.history` (`PoseHistory`), a ring
buffer of the last `HISTORY_SECONDS` = 8 s (capped at 480 samples). A missed
frame adds nothing; it only trims. `body_scale()` = median torso length,
preferring standing frames; `scale_source()` says whether it came from
`standing`, `lying` or a box-height guess. `annotate(..., timestamp)` passes the
camera thread's `time.monotonic()` frame time. `FrameRateMeter` measures the
rate the detector processes (can exceed the displayed rate); each session
appends a row to `runtime/logs/performance/<device>.csv`, and the terminal
prints `[performance] …` every 10 s.

**Post-fall watch (phase 1).** `PostFallWatch` starts on a track when its fall
is confirmed (again, if the person had got up). `movement_score()` compares the
average pose of the last 0.5 s with the average 1 s earlier and takes the
second-largest joint movement. Movement = score ≥ 0.15 on 2 frames in a row.
Movement in the first `LANDING_S` = 1.5 s after confirmation is the landing and
is ignored. Still time grows only on frames where the person is seen (gaps
capped at 0.5 s); still ≥ `STILL_CRITICAL_S` = 15 s = critical. Got up =
posture standing for ≥ 1 s with the hips ≥ 0.8 torso lengths above their height
at confirmation. The watch is kept 10 s after recovery, then removed.

**People around (phase 2).** `update_bystanders()` runs every frame for each
watched person with everyone else seen in that frame. Distance = floor point to
floor point (bottom centre of each box) in torso lengths. If the fallen person's
scale came from standing frames (`scale_reliable`), the unit is their torso and
people whose torso differs more than `DEPTH_RATIO` = 2× are skipped (different
distance from the camera); otherwise (already lying when first seen) the unit
is the other person's own torso and the depth check is skipped, because a torso
lying towards the camera looks short. A new bystander must be standing when
first seen. Nearby ≤ 6; with them ≤ 2.5 for ≥ 2 s (`helpers`, `helped_at`);
coming over = distance dropped ≥ 1 within 2 s; `no_response` = someone nearby
≥ 20 s and nobody with them or coming over. A kneeling helper given a new tracker
ID takes over the vanished bystander's entry (≤ 1.5 s, ≤ 1.5 torso lengths).
`PersonTracker.helper_ids()` feeds the helper exemption (section 1).

**Sudden vs gradual (phase 3).** `descent_analysis()` times the hip centre's
drop: standing level = median of the highest standing-frame hip positions,
floor level = median of the last 3 frames. The last complete 10 % → 90 %
crossing is timed with interpolation and divided by `DESCENT_SHARE` = 0.65
(real falls speed up and slow down). Sudden < 1.5 s; borderline 1–2 s. Drop
< 0.6 torso, no standing frame, no complete crossing, or a frame gap > 0.75 s →
`unclear` with a reason (`type_note`). Measured at confirmation and again for
`TYPE_SETTLE_S` = 1.5 s, then fixed (`type_final`). Synthetic falls of 0.4–5 s at
8 and 15 fps were timed within about 10 %.

**Sudden-fall patterns (phase 5A).** `fall_pattern()` looks from the hips' 10 %
time − 0.4 s to the 90 % time + 0.6 s. First contact: each part (`PARTS`: hips,
knees, hands = wrists, upper = nose + shoulders) "landed" when it covered 85 %
of its own drop (perspective-independent; a part must drop ≥ 0.4 torso). Parts
within `TIE_S` = 0.1 s tie; hands in a tie = hands first; a hips+knees tie = no
order. Arms reached out = a wrist ≥ 0.8 torso from the shoulder–hip segment on
≥ 2 frames (overhead arms count). Direction (profile views only) from the nose's
facing side and where the head ends up. Rules in order: arms not out and
knees/upper first → `collapse`; hands first, knees first with arms, or arms out
and forward → `trip`; hips first or backward → `slip`; arms not out and a
hips+knees tie → `collapse`; else `unclear`.

**Gradual-fall patterns (phase 5B).** `gradual_pattern()` looks from 3 s before
the descent to 0.5 s after it. Hand to chest: a wrist within 0.45 torso of the
chest spot (35 % of the way from shoulders to hips) held ≥ 1.0 s → `chest`.
Unsteady: the hips zigzag sideways (≥ 2 reversals of 0.2 torso; walking one way
gives none) or an arm out ≥ 0.5 s before half-way down → `unsteady`.
Controlled: hip descent speed never above 1.6 torso/s, or hips resting ≥ 0.6 s
part-way down → `controlled`. Order: chest → unsteady → controlled → unclear.

**Priority.** `priority_of(report)`, highest first:

| Priority | When |
|---|---|
| `recovered` | Got up |
| `critical` | Still ≥ 15 s (or critical earlier and not moving now); a collapse and not moving; hand to chest (even if moving) |
| `check` | Controlled descent, not borderline |
| `urgent` | Everything else: sudden, unsteady, unclear |

`PRIORITY_STYLE` maps each to an emoji, word, Discord colour and window colour.

**Report.** `PostFallWatch.report()` (sent each frame through
`tracker.aftermath()`): `status` (watching / moving / still / recovered),
`still_s`, `longest_still_s`, `since_fall_s`, `moved`, `critical`,
`moved_after_critical`, `down_s`, `fall_type`, `descent_s`, `borderline`,
`type_note`, `type_final`, `pattern`, `first_contact`, `direction`, `arms`,
`chest_s`, `sway`, `reaching`, `sat_first`, `smooth`, `nearby`, `with_them`,
`approaching`, `helpers`, `help_after_s`, `ever_nearby`, `no_response`,
`priority`. Built fresh every call.

**Labels and console.** Video labels (ASCII only — OpenCV's Hershey font):
`#2 FALL CONFIRMED (slip) - moving`, `(sudden, pattern unclear)`,
`(type unclear: not seen standing first)` (`SHORT_REASON`), `- NO MOVEMENT 16 s`,
`#3 Standing - near #2` / `- going to #2` / `- with #2`, `#3 Helping - with #2`,
`#2 Got up`. Once per fall, when its type settles, `watch_summary()` prints one
`[analysis] #2: …` line to stderr: type, pattern and signs, priority, body scale
and its source, and why every other person is near or not.

### Hand-off of the analysis to the window

`frame_ready`'s last argument is declared `object` (and
`@Slot(QImage, bool, int, int, int, list, object)` on `MainWindow.show_frame`).
It was `dict` until 25 Nov 2026: PySide6 converts a `dict` signal argument to a
QVariantMap, whose keys must be text, so the int-keyed reports arrived as `{}`
(confirmed on PySide6 6.9.2). The analysis ran, but the window, fall log and
Discord never received it. Passing the object between threads is safe because
`aftermath()` builds a fresh dict every frame; never send internal mutable state
through a signal. `tests/test_frame_signal.py` guards this with a real queued
signal.

### GUI side of the analysis

In `show_frame()`, `MainWindow` stores the reports (`_aftermath`), feeds the
alert state to `FallLog.update()`, then `FallLog.note_aftermath()`, which
records types, priority, movement and helpers for the people in the ongoing
event and returns one-shot changes: `critical`, `moving_again`, `helper`,
`no_response`. The first Discord alert of a fall waits in `_start_pending` until
everyone's type is final or 1.5 s has passed; any other message flushes it
first. `FallLog.priority_now()` drives the banner colour and title, and the
Urgency meter. A fall ends "Got up" when someone in it recovered and nobody is
lying still.

### Data architecture and state

There is no database or user record schema. No snapshots or clips are saved.

| Data | Lifetime/location | Notes |
|---|---|---|
| Webcam frame | Worker memory | BGR NumPy array, annotated in place |
| Model result | One inference iteration | Boxes and `keypoints.xy` only |
| Person tracks | Worker session | ID, box, streak (cap 5), misses, helping flag; dropped after 15 misses |
| Pose history | Worker session, per track | Last 8 s of keypoints, box, posture state |
| Post-fall watch | Worker session, per track | `Track.watch`: movement, bystanders, fall type, pattern |
| Display image | One pending signal | RGB QImage detached with `copy()`; at most one pending |
| Fall log | `runtime/logs/falls/<device>.csv` | One row per finished fall; Git-ignored |
| Cleared logs | `runtime/logs/falls/cleared/` | Backups made by Clear log |
| Performance log | `runtime/logs/performance/<device>.csv` | One row per camera session |
| Alarm delay, Discord settings | `QSettings` ("Vertebrate") | Per user, outside the repository |
| Fixed Discord webhook | `runtime/config/discord_webhook.txt` | Secret; Git-ignored |
| Alarm sound | `runtime/sounds/alarm.wav` | Generated siren |
| Ultralytics settings | `runtime/logs/standalone-yolo/` | Offline settings |
| Model weights | `yolo11n-pose.pt` at root or `models/` | Root checked first; Git-ignored |
| URFD footage | `data/videos/urfd/` | 100 local clips; Git-ignored; not read by the app |

### Fall log

`FallLog` (GUI thread) turns the confirmed alert into events. An event starts
when the alert turns on and ends when it has been off for `END_GRACE_S` = 2 s
(the end time is when it first went off), or when the camera stops, an error
occurs or the app closes. Each finished event appends one CSV row (UTF-8 with
BOM for Excel) with these 26 fields:

`device, fall_detected_at, ended_at, utc_offset, duration, duration_seconds,
latitude, longitude, accuracy_m, location_source, ended_by,
alarm_after_seconds, alarm_rang_at, alarm_silenced_at, person_ids, after_fall,
longest_still_seconds, no_movement_alert_at, people_nearby, helped_by,
help_after_seconds, fall_type, descent_seconds, priority, fall_pattern,
pattern_signs`

Logs written by an older version are rewritten once with the new columns (old
cells empty). `ended_by` is Got up, Posture cleared, Camera stopped, an error or
App closed. `priority` is the highest the fall reached (recovered excluded).
Location is the fix at the start (or the first during the event). The device
name is the host name; each device writes its own file.

The **Fall log** button opens a non-modal `FallLogDialog`: newest first, an
ongoing fall live on top, columns Fall detected at, Ended at, Duration, Person,
Priority (coloured), Type, After the fall, Help, Alarm, Location, Location
source, Ended by; totals; **Open log file location**; and **Clear log…**.
Clear log asks first (warning dialog, Cancel default and Escape), then
`FallLog.clear()` moves the file to
`runtime/logs/falls/cleared/<device>-cleared-<YYYYmmdd-HHMMSS>.csv` (a rename,
not a delete) and keeps an ongoing fall, which is written to a fresh file when it
ends. The dialog's `cleared` signal refreshes the main window's counts.
Read/write failures are shown in the dialog and printed; they never stop
monitoring. A fall in progress when the app crashes is not logged.

### User interface

Fusion style with a dark palette (`apply_dark_theme()`) so Windows and macOS
look the same. Colours are constants above `STYLESHEET`: a teal accent for
actions; priority colours amber (check), orange (urgent), red (critical) and
green (recovered / all clear). Numbers use the fixed-width system font.

- **Header:** logo mark, product name, device chip, live clock, state pill.
- **Main column:** the banner, the live view (`VideoView`: state-coloured
  border, LIVE tag, placeholder when off) and Start monitoring / Stop / Fall log
  buttons. The banner during a fall takes the priority colour and reads e.g.
  "Urgent · Possible slip — person #2" with the time down, movement, who is
  around and the alarm countdown; lying still turns it to "Critical · No
  movement — person #2 lying still". When the siren rings it becomes a pulsing
  red "EMERGENCY ALARM" with a **Silence alarm** button.
- **Sidebar (scrolls on small windows):** Status card; **Urgency** card
  (`UrgencyMeter`: Check / Urgent / Critical segments in the priority colour,
  green when recovered, with who and why from `refresh_urgency()`; before
  confirmation "Possible fall · #2 · 3 of 5 frames"); Live stats (people in
  view, detector frames per second, falls today, alarm countdown); Location;
  Emergency alarm setting; Caregiver alerts card.

`MainWindow.set_state()` drives the window from one state: idle, starting,
monitoring, fall, alarm, stopping, error. Stylesheets change only when the state
(or the alarm pulse) changes; per-frame updates touch text only.

### Emergency alarm

**Ring if fallen for** (0–600 s, 0 = Off, default 30 s) is saved with
`QSettings`. `check_alarm()` runs every 250 ms while the camera is on and on
each displayed frame: when the ongoing fall has lasted the delay, `AlarmSound`
loops a two-tone siren, the window comes to the front, and **Silence alarm**
appears. It rings at most once per fall and stops when the fall ends, Silence
is pressed, the delay is set to Off, or the camera stops. Ring and silence times
go to the fall log. The siren WAV is generated with NumPy on first use; without
Qt Multimedia, system beeps are used. Priority does **not** change the siren.

### Critical runtime flow

1. The user clicks **Start monitoring**; the GUI creates and starts
   `CameraThread`, the location feed and the alarm timer.
2. The worker calls `local_weights()` (missing weights → local error before
   `YOLO` is constructed; never downloaded), sets offline Ultralytics settings,
   builds the pose model and opens camera 0 with `open_camera()`.
3. Per frame: YOLO on the CPU → `annotate()`: `posture_state()` per person →
   `PersonTracker.update()` (IDs, streaks, helper exemption, pose history,
   post-fall watch and bystanders) → labels drawn → `FrameRateMeter.tick()`.
4. If no display frame is pending, the frame becomes a detached RGB QImage and
   `frame_ready` is emitted with the confirmed IDs and `tracker.aftermath()`.
   Inference continues when display frames are skipped.
5. `show_frame()` paints the location watermark, updates the view, the fall log,
   the analysis bookkeeping, alerts, the alarm and the state, then clears the
   pending flag.
6. Stop/close requests interruption; the worker exits, releases the camera and
   writes its performance row; the GUI ends any ongoing fall ("Camera stopped")
   and restores controls.

### Network and privacy behavior

Before importing Ultralytics, `common.py` sets `YOLO_OFFLINE=true`,
`ULTRALYTICS_OFFLINE=1`, `YOLO_AUTOINSTALL=false` and a local
`YOLO_CONFIG_DIR`; `sync` and `hub` are disabled in memory. Weights must be
local. There is no upload or remote inference; webcam frames stay in process
memory. Package installation and the one-time weights download are separate
setup steps.

The one outbound connection is the **caregiver alert** to Discord: text only
(device name, times, durations, priority, fall type and pattern, people around,
location and a Google Maps link; never images or video). Failures never affect
detection, logging or the local alarm. OS Wi-Fi positioning may contact the OS
vendor's location service.

### Caregiver alerts (Discord)

`caregivers.py`. `DiscordNotifier` (a `QObject` owning one daemon thread and a
queue) posts JSON with `urllib` and certifi's CA bundle when present. Two
modes: a **webhook URL**, or a **bot token + channel ID**. Every request sends
a `User-Agent`. Up to three attempts; HTTP 429 waits for `retry_after` (max
10 s); 401/403/404 fail at once with a readable reason. Outcomes return through
the `result` signal to the sidebar card, and every alert prints a
`[caregivers] …` line to stderr. On close the app waits at most 3 s for queued
alerts.

Messages (`MainWindow`), each an embed in the priority colour unless noted:

| Kind | When | Title example | Ping |
|---|---|---|---|
| `fall` | First alert, up to 1.5 s after confirmation | "🟠 URGENT — Possible slip — Person #2" + fields Priority, How they fell, People around, What it may mean, Signs, Note | Yes |
| `critical` | First time someone lies still 15 s | "🔴 CRITICAL — Person #2 not moving" | Yes |
| `fall` | Moving again after critical | "🟠 Update — Person #2 moving again" | No |
| `fall` | Someone reaches them | "🟡 Someone is with them — Person #2" | No |
| `fall` | Nearby 20 s, nobody came | "🟠 Nobody has responded — Person #2" | Yes |
| `alarm` | Siren starts | "🚨 EMERGENCY — Person #2 still down" (red) | Yes |
| `end` | Fall ends | "🟢 Recovered — Person #2 got up" or "✅ Fall ended — Person #2", with duration, how they fell, highest priority, after the fall, help | No |

`allowed_mentions` only allows the ping the user configured. Settings are
stored with `QSettings` group `caregivers`. If
`runtime/config/discord_webhook.txt` holds a valid webhook URL,
`load_caregiver_config()` returns a **fixed** config: always enabled, all
events, no ping; the sidebar button then sends a test message.

### Location watermark

`LocationFeed` (GUI thread) runs up to two Qt Positioning sources: a **GPS
receiver** (`"nmea"`, only when `find_gps_port()` finds a likely GPS serial
port; `VERTEBRATE_GPS_PORT` overrides or `off` disables; `VERTEBRATE_GPS_BAUD`
defaults to 9600) and the **system location** (`"winrt"` / `"corelocation"`).
A receiver fix newer than 10 s wins. `paint_watermark()` draws coordinates,
accuracy, local date/time and the source on each displayed frame. Fixes are
held in memory only.

### Platform support

Windows 10/11 x64 and macOS, Python 3.12, CPU inference. `open_camera()` tries
DirectShow then Media Foundation on Windows and AVFoundation on macOS. The UI
uses the platform system font and scales for high-DPI displays.

---

## 3. PROJECT SCAFFOLDING & CODEBASE STRUCTURE

### Current workspace tree

Tracked files as of `30308cf` (25 Nov 2026), plus local-only folders.

~~~text
Vertebrate/
├── app.py                         # Launcher: theme + MainWindow
├── app_modules/
│   ├── common.py                  # Imports, constants, posture rules, annotate(), theme
│   ├── camerathread.py            # CameraThread: capture, model, frame_ready
│   ├── tracking.py                # PersonTracker, Track, helper exemption
│   ├── fallanalysis.py            # Pose history, post-fall watch, bystanders, type, patterns, priority
│   ├── mainwindow.py              # MainWindow: UI, fall/alert/alarm flow
│   ├── falllog.py                 # FallLog: events, CSV, clear()
│   ├── falllogdialog.py           # Fall log window, Clear log
│   ├── caregivers.py              # Discord notifier and setup dialog
│   ├── locationfeed.py            # GPS receiver / OS location
│   ├── alarmsound.py              # Siren
│   ├── urgencymeter.py            # Urgency meter widget
│   ├── videoview.py, stattile.py, logomark.py   # Widgets
│   ├── evidencemeter.py           # UNUSED (replaced by urgencymeter.py) — remove
│   └── debounce.py                # UNUSED — remove
├── tests/                         # unittest: frame signal, fall analysis, fall log
├── README.md                      # Setup and features for users
├── CONTEXT.md                     # This architecture and agent guide
├── HANDOFF.md                     # Current state, open items, next steps
├── requirements.txt               # Pinned Windows CPU dependencies
├── requirements_macOS.txt         # macOS dependencies
├── requirements-win-cpu.lock      # Former environment lock, 48 distributions
├── LICENSE, THIRD_PARTY_NOTICES.md
├── .gitignore, .gitattributes
├── cleanup-legacy.ps1             # Migration utility; -Apply is destructive
├── docs/acknowledgements/         # Clinic acknowledgement letter (PDF)
├── models/                        # manifest.json (+ local weights, ignored)
├── data/videos/urfd/              # 100 local URFD clips; ignored
├── runtime/                       # Generated local state; ignored
└── .archive/                      # Legacy ZIP and release-prep copies; ignored
~~~

The URFD clips (40 ADL, 30 falls × 2 cameras) are not read by the app. The ZIP
in `.archive/` contains the old `src/vertebrate` package and its tests; it is
archival, not runtime code. Inspect `cleanup-legacy.ps1` targets before any use.

### Component and code conventions

`CameraThread` owns blocking I/O, inference, tracking and analysis;
`MainWindow` owns visible state; `FallLog` owns events and the CSV;
`DiscordNotifier` owns the network. Preserve those boundaries and a simple
`python app.py` launch. Modules use `from .common import *` for shared imports
and constants; `fallanalysis.py` imports `common.py`, so `common.py` reaches it
only through small lazy wrappers (`watch_label`, `bystander_label`).

`snake_case` for functions/variables/modules, `PascalCase` for classes,
`UPPER_SNAKE_CASE` for constants (tunable thresholds sit at the top of their
section with a comment). Four spaces, type hints on new public interfaces,
concise docstrings, comments for thread/lifecycle invariants. There is no
`pyproject.toml`, linter or formatter; do not add one as a side effect.

---

## 4. DEVELOPMENT & DEPLOYMENT STANDARD OPERATING PROCEDURES (SOPS)

### Setup and launch

~~~powershell
# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
~~~

~~~bash
# macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_macOS.txt
.venv/bin/python app.py
~~~

Put `yolo11n-pose.pt` at the root or in `models/`; missing weights fail
locally. The app opens camera index 0; file input and camera selection are not
implemented. Install only one OpenCV variant.

### Coding and threading standards

Camera reads, inference, annotation, tracking and analysis stay off the GUI
thread. Create and release `VideoCapture` in its owner thread. Pass detached
values across threads: `QImage.copy()`, and a freshly built dict for the
analysis. Use queued signals. Keep at most one display frame pending.

**Signal argument types:** a Python `dict` declared as `dict` is converted to a
QVariantMap (text keys only) and arrives empty if its keys are numbers. Declare
anything that is not a plain number, string or list as `object`, and change the
signal and the receiving `@Slot` together.

~~~python
class CameraThread(QThread):
    # frame, alert, leading streak, leading ID, people, confirmed IDs, analysis
    frame_ready = Signal(QImage, bool, int, int, int, list, object)

    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    self.frame_ready.emit(image, is_fallen, evidence, leader_id, people, confirmed,
                          tracker.aftermath())

worker.frame_ready.connect(window.show_frame, Qt.ConnectionType.QueuedConnection)
~~~

Display backpressure must not skip inference or change tracking. Create a new
`PersonTracker` per camera session. Video labels drawn with OpenCV must be
ASCII. Emit readable errors and release owned resources in `finally`.

### Error handling and logging

The worker catches runtime exceptions and emits type/message; the GUI shows the
error and restores controls. There is no global exception framework or rotating
app log. Diagnostic lines go to stderr: `[analysis]` (one per fall),
`[performance]` (every 10 s and per session), `[caregivers]` (every alert).
Do not suppress failures that could leave the camera open or the UI stuck.

### State and security

The only secrets are the Discord webhook URL / bot token: in
`runtime/config/discord_webhook.txt` (Git-ignored) or the user's `QSettings`.
Never commit them, print them or paste them into issues or chats. Do not add
cloud inference, telemetry, analytics or dispatch without an explicit scope
change. Keep offline flags set before importing Ultralytics. Never pass missing
weights to YOLO (it may try to download). Startup does not verify the model
hash in `models/manifest.json`; never claim it does.

### Licensing and release

`LICENSE` grants MIT rights over Advaet Chintakunta's original code.
Ultralytics software and YOLO weights have separate AGPL-3.0 terms; the MIT file
does not relicense them. Review Ultralytics and Qt terms before any release.
This is a technical guide, not legal advice. Do not force-add weights, videos,
`runtime/` or `.venv`.

### Testing protocol

Automated tests use `unittest` and need no camera:

~~~bash
.venv/bin/python -m unittest discover -s tests -v        # macOS
.\.venv\Scripts\python.exe -m unittest discover -s tests -v   # Windows
~~~

| File | Tests | Covers |
|---|---|---|
| `tests/test_frame_signal.py` | 2 | Analysis dict survives a real queued `frame_ready`; slot type matches |
| `tests/test_fall_analysis.py` | 3 | Lying from the start + bystander (near, label explains unclear type); lying towards the camera; a timed sudden fall |
| `tests/test_fall_log.py` | 3 | Clear log keeps a byte-identical backup; a fall in progress survives; nothing to clear |

As of 25 Nov 2026 the analysis and fall-log tests pass with Qt stubbed; the
signal test needs the real PySide6 and has not yet been run on a Mac or PC.
Add focused tests for every change to posture rules, tracking, analysis or the
fall log. Then test with a real webcam: one person stands and falls while a
second stands nearby; check the video labels, the `[analysis]` line, the
Urgency card, the banner, the fall log row and the Discord message. Mock and
synthetic results are not physical-camera verification; never present them as
accuracy figures.

---

## 5. AGENT DRILLS & GUARDRAILS (CRITICAL)

### Modification rules

1. Before architecture changes, read `app_modules/`, `requirements*.txt`,
   `models/manifest.json`, `.gitignore`, this file and `HANDOFF.md`.
2. Treat `app.py` + `app_modules/` as active. Do not revive archived imports,
   ByteTrack, schemas, storage or old Qt workers without an explicit request.
3. Preserve the posture rules and per-person confirmation: too-close first;
   ratio `> 1.2` with lower body visible; spine `shoulders_y > hips_y + 20`;
   five consecutive fallen frames of the SAME person; reset on that person's
   non-fallen frame; streak kept on a missed detection; the helper exemption
   only while the torso is upright. Document any requested change here.
4. Preserve thread ownership. Never update widgets from `CameraThread`, share a
   NumPy-backed live QImage, or destroy an active QThread. Keep `frame_ready`'s
   analysis argument `object`, built fresh per frame.
5. Analysis thresholds live as named constants in `fallanalysis.py`. Change
   them only with a test that shows the effect, and record the evidence here.
6. Preserve offline behavior. The Discord alert is the one explicit exception:
   text-only, off unless configured (or the fixed webhook file), never in the
   detection path.
7. Keep local footage, weights, `runtime/` and secrets out of Git.
8. Never describe patterns as diagnoses, and never claim medical performance,
   fall causality, accuracy figures or hardware validation without evidence.
   There is no evaluation runner yet.
9. Video labels stay ASCII. Discord and window text may use symbols.
10. Update README, this file and `HANDOFF.md` in the same change when behavior,
    files or setup change.

### Definition of Done

- [ ] Change targets the active app and does not depend on archived modules.
- [ ] Posture, tracking, analysis and alert semantics match the request and
      have focused tests in `tests/`.
- [ ] Model/OpenCV/analysis work stays in the worker; GUI updates are queued;
      cross-thread data is detached, freshly built and declared `object` when
      it is a dict or custom object.
- [ ] Start, Stop, errors, missing weights, camera-open failure and close leave
      resources and controls in defined states.
- [ ] No model auto-download, telemetry, upload or recording added.
- [ ] `unittest discover -s tests` actually ran and outcomes are reported;
      state clearly when the real PySide6 or a camera was not available.
- [ ] Synthetic, mocked and physical-camera results are distinguished.
- [ ] README, CONTEXT and HANDOFF describe the resulting code.

---

## Current known limitations

- **Analysis validated on synthetic falls only.** No real-footage evaluation
  yet; expect wrong or unclear labels until a replay of the URFD clips and
  staged falls has been checked.
- **Slip read as trip:** in a synthetic 0.6 s slip at 15 fps the hips land one
  frame (0.067 s) before the hands; the 0.1 s tie window and "hands win a tie"
  make it a trip on 8 of 8 noise seeds.
- **Collapse read as trip** in 1 of 27 synthetic cases with heavy keypoint
  wobble on a small person (it becomes Critical after 15 s still anyway).
- **Hand to chest** is triggered by a phone or cup held at chest height.
- **2D only:** a front-on camera cannot tell forward from backward; depth is
  guessed from body size.
- **Tracker ID changes** during fast falls or when people overlap (box matching
  only, no appearance model) give "type unclear: not seen standing first"; a
  person hidden for more than 15 frames returns with a new ID.
- A helper touching the fallen person can mix keypoints, which may read as
  movement.
- Below about 5 fps a fast fall cannot be timed ("frame rate too low").
- Priority does not change the siren; Critical still waits for the alarm delay.
- Ordinary lying down is a fall posture (it may be labelled "Slow, controlled
  descent"). The 20-pixel spine margin changes meaning with distance.
- Caregiver alerts need internet; an alert that fails after three attempts is
  not resent. Discord credentials are stored in plain text.
- A fall in progress when the app crashes is not logged.
- Root weights are preferred; the manifest hash is not checked at startup.
- No video-file input, camera selector, recording or evaluation runner.
