# Project VERTEBRATE

One local Python desktop application for observing a suspected fall followed by
sustained stillness. CPU pose inference, per-person tracking, source-time
kinematics and an ordered state machine feed a local incident inbox.
**Local demo — no external emergency dispatch.** No diagnosis, EMS connection,
cloud service, paid API or external telemetry is provided.

P0–P6 implementation is ready for physical human calibration. Human accuracy,
physical-camera compatibility and live release gates remain unverified.
Follow the [rehearsal runbook](docs/demo-runbook.md).

## References and input envelope

The [validated blueprint](docs/03-validated-blueprint-v2.md), §§1–16, is the
implementation contract; the [validation report](docs/02-validation-report.md)
supplies rationale. Both supplied documents are preserved verbatim. Embedded
historical prompts and external-service instructions are quoted research.

Use a fixed, level, oblique/side camera, a clear marked floor zone, one or two
people, and full-body visibility including feet before and after the event.
Overhead/end-on or moving cameras, crowds, major occlusion, subtle chair slumps
and people already down are outside the supported detection envelope. Deliberate
rapid lying down can look like a fall; the system cannot infer intent.
The alert is **“Suspected fall with sustained stillness — check the person.”**

## Observed hardware and environment

Rechecked locally on 2026-10-04:

| Fact | Observed value |
|---|---|
| OS | Windows-11-10.0.26200-SP0, AMD64 |
| Python | CPython 3.12.0, 64-bit |
| Processor identifier | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| Logical processors | 32 |
| CPU trade name, installed RAM, physical webcam | Not independently verified |

See [baseline evidence](reports/baseline.md) and the
[alignment audit](reports/p0-alignment-audit.md). Tested pins include Ultralytics
8.3.203, torch 2.8.0+cpu, torchvision 0.23.0+cpu, NumPy 2.2.6, opencv-python
4.12.0.88, PySide6 6.9.2, lap 0.5.12, PyYAML 6.0.2, pytest 8.4.2 and pytest-qt
4.5.0. Keep exactly one OpenCV distribution; use PySide6, not PyQt.

## Offline installation and model provenance

Run from the repository root. requirements-win-cpu.lock includes exact wheel
hashes and targets CPython 3.12 / Windows AMD64. Other platforms need separately
resolved and tested wheels. Transfer the repository, trusted model/manifest,
allowed videos and every locked wheel to the demonstration machine first.

The existing local setup has most wheels in runtime/wheelhouse and pytest-qt
in runtime/setup-wheels; both ignored directories are included below.
A clean-machine reinstall has not been performed as part of P6.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links runtime/wheelhouse --find-links runtime/setup-wheels -r requirements-win-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m vertebrate doctor --offline
~~~

For an explicitly online setup, prepare a complete wheelhouse using
pip download -r requirements-win-cpu.lock --dest runtime/wheelhouse inside the
project environment, then transfer it. tools/lock_wheels.py hashes actual wheel
archives, not installed RECORD files. Do not replace the lock with unhashed pins.

The model is models/yolo11n-pose.pt (6,255,593 bytes), from
[Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt).
Recorded SHA-256:
869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0.
models/manifest.json records task, size, source, version and license.
tools/provision_model.py is an explicit online setup helper and refuses to
overwrite an existing model. The application never calls it. A hash checks local
integrity, not an independent publisher signature.

Ultralytics code/weights carry an AGPL-3.0 notice. Review
[third-party license obligations](docs/licenses.md) before redistribution.
No custom font files are bundled.

## Launch and desktop workflow

~~~powershell
.\.venv\Scripts\python.exe -m vertebrate
.\.venv\Scripts\python.exe -m vertebrate --source tests/fixtures/videos/dev_empty_scene.avi --config config/demo.json
.\.venv\Scripts\python.exe -m vertebrate --source 0 --config config/demo.json
~~~

Pick a file/camera and press Start. File Pause/Resume freezes source time; Replay,
seek, restart and source switch create a fresh session and baseline. Stop before
editing profiles/manual context; UI edits advance the config revision.
demo-3s requires three seconds of observed stillness; observation-10s requires
ten seconds and a sufficiently long clip. Starting thresholds remain uncalibrated
on people in the intended room.

The display separates processing FPS, source FPS and source-time elapsed.
“Possible fall — observing” is a candidate, not confirmation. Open the confirmed
incident in the local inbox to inspect metadata/snapshot and acknowledge it.
Delivery means saved locally only. Storage faults pause acquisition and retain
the alert for Retry. Closing waits for cooperative shutdown; a blocked driver
reports a timeout without pretending resources have been released.

The GUI polls one immutable latest result. Separate capture, vision and writer
owners use bounded queues. Files have lossless backpressure; live capture
discards the oldest queued frame at capacity two. All evidence uses source time.
EOF never repeats the final frame. Historical video UTC stays unknown unless
explicitly supplied; analysis UTC is not the historical event time.

## Optional playback clips

Clips are disabled by default. While stopped, save a separate config with
runtime.enable_optional_clip set to true and increment its revision; launch
with --config pointing to that file. Do not change detection size/rate to enable
playback recording.

The vision owner retains immutable 320×180 BGR frames sampled at most five times
per source second, for at most 30 seconds: 150 frames and 25,920,000 raw pixel
bytes. Clip/snapshot allocations and in-flight requests share
runtime.max_media_buffer_bytes, capped at 128 MiB. This is a media allocation
limit, not a total process-RSS or native codec workspace guarantee. Simultaneous
incidents share immutable ring pixels. Capacity gaps are marked as truncation.

Snapshot/config/incident JSON are saved immediately after confirmation. Optional
post-roll collects available footage from onset minus three seconds through
confirmation plus three source seconds. EOF, Stop, replay and source switch
flush available footage as truncated. clip_status is pending while collecting;
an interrupted application's pending clips become unavailable on restart.

The writer encodes temporary MJPG/AVI, verifies every decoded frame, then commits
clip.avi and clip-timestamps.json before replacing incident.json. The sidecar
records frame sequences, source timestamps, requested bounds, truncation reasons
and seek offset. Playback is fixed at 5 FPS; use timestamps for irregular gaps.
No interpolation or sampling alters detection evidence. Open clips locally from
their incident directory; the inbox displays snapshot and metadata.

Unavailable codecs/clip writes fall back to snapshot plus JSON with clip_path
null and explicit clip_error. Metadata failure retains the request for retry.
Clip queue overflow surfaces the existing fault/pause policy. No continuous disk
recording occurs.

## Calibration and locked evaluation

Use consented development footage for the actual camera and room. Measure
stationary jitter, sits, bends, slow lies, supported falls and recovery. Compare
input sizes on development data; tune speed/displacement, posture/ground, then
stillness/deadlines in a small documented grid. Check ±10% threshold sensitivity.
Freeze code, config, weights and matching policy before headline holdout.

Split by subject/session/original recording, keeping derived views/crops together.
Blueprint §13 requires at least 15 development falls and 20 negative sequences;
holdout needs at least 30 supported falls, eight per direction, six with two
people, and 60 minutes of negatives including 30 minutes in the actual room.
Never tune on reused holdout results and keep calling them unseen.

~~~powershell
.\.venv\Scripts\python.exe -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json
.\.venv\Scripts\python.exe -m vertebrate evaluate --manifest data/manifests/holdout.json --config config/demo.json --output reports/evaluation.json --fail-on-gates
~~~

Tracked synthetic CFR fixtures work offline. Their generators are
tools/generate_dev_fixtures.py and tools/generate_holdout_fixtures.py.
Regenerate manifest hashes alongside clips if a different codec changes bytes.
Drawings test pipeline/scoring mechanics, not human accuracy.

The historical P5 [evaluation](reports/evaluation.json) produced 0 TP, 0 FP, 1 FN,
recall 0 and precision null on synthetic annotations. Release gates failed. Its
lock records that run's hashes; it is not a measurement for later code revisions.
Normal evaluate returns 0 after scoring even when gates fail; --fail-on-gates
returns 2. Runtime/input failures return 1.

Human holdout accuracy/visibility coverage, actual-room negatives, three
rehearsals, five-minute live/UI measurements, physical-camera lifecycle and the
30-minute soak/RSS gate remain unverified. Automated lifecycle tests use injected
readers and are not physical-camera evidence.

## Verify and troubleshoot

~~~powershell
.\.venv\Scripts\python.exe -m pip check
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m vertebrate doctor --offline
Remove-Item Env:QT_QPA_PLATFORM
~~~

Unset QT_QPA_PLATFORM for visible desktop use. Doctor validates config/tracker
YAML, Qt and numerical imports, single OpenCV installation, outbox
write/fsync/delete, model hash and real CPU pose/ByteTrack smoke inference.
Missing/corrupt weights fail before model construction with no auto-download.
Offline Python network/HTTP/DNS/subprocess guards and disabled Ultralytics
integrations cover the pinned stack; they are not an OS firewall for arbitrary
native code.

| Symptom | Local action |
|---|---|
| Missing/hash-mismatched weight | Restore verified asset and rerun doctor |
| Invalid/VFR timing | Use the displayed local ffmpeg CFR conversion command; preserve original and update hash/labels |
| Inconclusive/repeated acquisition | Restore full-body visibility; never fill missing evidence with frozen frames |
| Clip unavailable | Inspect clip_error; snapshot/JSON is the fallback |
| Save failure/full queue | Repair storage, Retry retained records, then Resume; failed initial save and clip update may each need retry |
| Driver disconnect/timeout | Wait for the owning driver; use validated file analysis if hardware stays blocked |
| Missing/stale context | Enter measured/simulated provenance; context never gates detection |

Keep participant permissions and dataset notices. Delete incidents when no longer
needed; do not commit private videos, outbox records or secrets. Local execution
alone is not a privacy/legal compliance certification.
