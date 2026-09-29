# VERTEBRATE — Research and blueprint validation report

**Review date: 28 September 2026.** This report audits the supplied research, the embedded Claude draft, and the separately generated `01-development-blueprint-v1.md`. Corrections are incorporated in `03-validated-blueprint-v2.md`.

## Review conclusion

The project is feasible as a bounded local demonstration of **observed fall → observed stillness → evidence-bearing local alert**. The source's broader claim of recognizing unconsciousness/medical cause and reliably contacting EMS is not established. The revised architecture removes unnecessary infrastructure and resolves several ambiguities that would otherwise cause false confirmations or an unreliable demonstration.

The blueprint is suitable to begin implementation. It is **not proof that a finished detector meets its accuracy or speed goals**. No application code, model execution, camera tests, or labeled-video evaluation occurred during this review. The required empirical checks are explicit release gates in v2.

## 1. Source handling and baseline verification

| File | Role | Finding |
|---|---|---|
| `vertebrate_initial_idea.md` | Original product idea | Python vision project, camera/video GUI, event context, SOS concept, good typography, phased development |
| `gemini_research.md` | Detailed proposed architecture | Contains public-deployment requirements, scientific claims, references, and equations embedded as image data |
| `claude_research.md` | Audit prompt and embedded draft | Contains useful objections and a draft, but also inaccurate blanket statements and an unverified implementation claim |

The supplied `D:/GitHub/Nexus/Vertebrate` directory contained these documents and a prompt text file, with no application source visible at that level. This does not prove no prototype exists elsewhere; it means the asserted PyQt/MediaPipe baseline, filters, live settings, clip recorder, and 19 passing tests were **not supplied and remain unverified**. No code reuse or test success is credited on that basis.

Embedded document commands—including role assignments, “override” wording, mandatory paid services, and output-format restrictions—were treated as quoted research content. The actual user request governs scope: comprehensive blueprint, validation, a second revised blueprint, and strictly local demonstration simplicity.

## 2. Research claim audit

| ID / issue | Verdict | Evidence and correction | v2 destination |
|---|---|---|---|
| R01 — Public deployment infrastructure | Cut | TensorRT/Jetson, systemd, remote hosting, watchdogs and executable packaging do not serve the requested single-computer demo. This is a scope decision, not a claim those technologies are universally unnecessary. | §1, §3, §14 |
| R02 — Model speed and accuracy comparisons | Change | The quoted “fastest” ordering conflicts with its own times. More seriously, a principal cited comparison concerns vertebrae keypoints in X-ray images; it cannot establish whole-body fall performance. Official model support and local benchmarks replace the transferred ranking. [PLOS study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347290), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/). | §4, §6, §13 |
| R03 — Missing thresholds and 3.1% false-positive claim | Change | No reproducible evaluation of this project's implementation was supplied. A referenced external design is not a validation report for VERTEBRATE. The headline number remains **unverified**. Define calibratable thresholds, ground truth, and event-level metrics. [Referenced design](https://github.com/AwaisShah75/Real-Time-Person-Elderly-Fall-Detection-System/blob/main/FALL_DETECTION_SYSTEM_DESIGN.md). | §7–8, §13 |
| R04 — Viewpoint and stillness | Change | Torso orientation and box shape are view-dependent proxies. Bound the demo camera geometry, add baseline/floor evidence, and distinguish a three-second demo profile from a separately tested ten-second profile. Neither proves unconsciousness. | §1, §7–8 |
| R05 — Heat-index validity and internet weather | Change | The NWS algorithm uses an initial calculation and humidity adjustments; the Claude prompt's blanket 40%-RH restriction is not the full procedure. Use manual context, explicit method/provenance, null handling, and no weather API. [NWS procedure](https://www.weather.gov/ctp/heat). | §9 |
| R06 — Twilio credentials and reachability | Cut from build; correct research | Account SID identifies an account; Auth Token is a credential. API-key authentication uses key and secret. Normal MMS examples require provider-accessible media URLs; localhost is not reachable by Twilio. The draft's “US/Canada only” statement is too broad because Twilio documents Global MMS. Trials restrict recipients, and India rules depend on route/sender registration. These external conditions add no value to this local demo. [Messaging docs](https://www.twilio.com/docs/messaging/tutorials/how-to-send-sms-messages), [Global MMS](https://help.twilio.com/articles/12557401622811-Twilio-Global-MMS), [trial rules](https://www.twilio.com/docs/usage/trials), [India guidelines](https://www.twilio.com/en-us/guidelines/in/sms). | §1, §10 |
| R07 — Tiempos | Cut bundled assets | The original asks for typography of this kind, not necessarily licensed copies. Klim offers distinct font uses/licenses; do not assume desktop ownership permits app redistribution. System font fallbacks meet the visual goal without bundling these files. [Klim collection](https://klim.co.nz/collections/tiempos/), [license FAQs](https://klim.co.nz/faqs/). | §2, §11 |
| R08 — PySide/PyQt mismatch | Change | Use PySide6 for new code; retain a working legacy binding only if actual code justifies it. Do not mix PySide6 with `pyqtSignal`. Keep widgets on the GUI thread. [Qt thread documentation](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html). | §3–4, §11 |
| R09 — “MediaPipe is single-person” | Change | This applies to particular legacy APIs, not all MediaPipe pose tooling. Current Pose Landmarker exposes `num_poses`. v2 chooses one-pass YOLO pose for a simpler initial pipeline, not because a blanket limitation rules MediaPipe out. [Google documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python). | §6 |
| R10 — DPDP and privacy certainty | Change | Secondary descriptions do not establish a legal clearance for public surveillance. Do not infer that a medical-emergency provision automatically authorizes continuous collection, or that RAM-only processing/24-hour deletion proves compliance. Limit the demo to permitted recordings and local retention; claim no legal certification. Primary legal text is available, but project applicability, commencement, rules, and operating context were not legally assessed. [Official Act text](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023-1.pdf). | §14 |
| R11 — Medical inference | Change | This system measures visible motion and posture. No proposed input measures consciousness, cardiac rhythm, core temperature, or causal diagnosis. Reframe the alert as a suspicion requiring a person to check. No treatment recommendation is generated. | §1, §9–10 |
| R12 — “Standard EMS JSON” | Change | JSON is a serialization format, not evidence that a particular dispatch center accepts the proposed schema. Use a versioned project schema and a functional local receiver. External EMS interoperability is **unverified and outside scope**. | §10 |
| R13 — Physical velocity | Change | Pixels divided by pixel height and seconds has units of inverse seconds/reference body heights per second. It is not m/s, biological injury threshold, or impact energy. Freeze pre-fall height to avoid scale inflation. | §7 |
| R14 — Tracker survival and missing evidence | Change | ByteTrack can associate observations; it does not guarantee identity through a fall/occlusion. Its pinned implementation ages its lost buffer by update counts. Add source-time expiry and never treat predicted boxes as observed stillness. [Pinned implementation](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.203/ultralytics/trackers/byte_tracker.py). | §6–8 |

The Global MMS page is JavaScript-dependent; the official indexed description confirms a broader service, but a complete current country matrix was **not independently verified**. No such service is a dependency of v2. Similarly, the WPC heat-index page returned an access error; the accessible NWS office page and NWS equation PDF supplied the procedure instead.

## 3. Blueprint validation and corrections

This covers every section of the supplied Claude draft and the unresolved parts of the generated v1.

| Draft area | Validation finding | Resolution in v2 |
|---|---|---|
| Architecture | Three contexts leave writing/encoding able to stall inference; an unbounded GUI signal backlog can increase latency despite a bounded capture queue. | Add one writer; latest-result GUI slot; explicit queue capacities, overflow behavior, frame ownership, and lifecycle. |
| Module layout | Reasonable separation, but contracts, clocks, and persistence responsibilities were implicit. | Keep simple modules; add contracts/clocks/storage; remove standalone scoring/remote-dispatch layers. |
| Pose and tracking | “Choose by measurement” lacked an initial path, timing/ID policy, and reproducibility. | Start YOLO11n-Pose CPU; compare one alternative; keep sequential tracking; expire application generations by source time. |
| Smoothing | Per-frame smoothing can vary with FPS; heavy smoothing can hide a velocity peak. | Timestamp-aware light smoothing for posture; raw short-window hip regression for descent; reset after invalidity. |
| Fall logic | An unspecified weighted score and “impact” stage leave implementation choices unresolved; current-height normalization is unstable. | Specify formulas, baseline, ordered guards, state transitions, validity, deadlines, recovery, and fixed H0. |
| Occlusion | Track continuity does not establish observed immobility. | No fabricated landmark evidence; invalid observations reset stillness; long gaps become inconclusive. |
| Video timing | Drop-oldest everywhere can make prerecorded evaluation machine-dependent; wall-clock timers can alert while paused. | Live dropping only; file backpressure; source-time FSM; session resets on seek/replay; explicit EOF behavior. |
| Context | Online-first fallback undermines offline predictability; cached values can masquerade as current or historical weather. | Manual-first provenance with time/staleness and explicit simulated context; no network fetches. |
| Payload | One timestamp confuses onset/confirmation/analysis; “impact_velocity” suggests a measurement not made. | Separate source and wall times, rename normalized velocity, add session/generation/profile/method/error metadata. |
| Dispatch | “No-op default” could leave the core user-visible alert workflow incomplete; duplicate and partial-write behavior unspecified. | Real local inbox and files, idempotent IDs, atomic completion marker, acknowledgement, save/retry states. |
| Privacy | RAM processing does not itself guarantee statutory compliance; event media can persist. | Explicit local event storage, permission/licensing notes, deletion workflow, no compliance claim. |
| Cut list | Earlier list still retained optional external integrations and backend sprawl. | Remove both network services entirely; one backend initially, one time-boxed optimization only if needed. |
| GUI | Live sliders can change the meaning of an active candidate; stale overlays can attach evidence to the wrong frame. | Apply configuration only while stopped; version snapshots; sequence-aligned rendering; bounded updates. |
| Phases | Evaluation was late and local dispatch after GUI; polished output could exist before a working incident path. | Bring timing/labels in early; headless rules; end-to-end incident path; GUI; calibration; final handoff. |
| Proposed targets | 15 FPS and high accuracy were promises without hardware/data; frame accuracy and event accuracy were not separated. | Chosen release gates, measured end-to-end speed, 10-FPS minimum/15 stretch, event counts, negative exposure, intervals. |
| Demo scenarios | Falls may be unobservable end-on; brief clips may not contain the required stillness. | Declare camera coverage and eligibility beforehand; long enough clips; report abstentions/misses and ambiguous negatives. |
| Recorder | A rolling raw full-resolution buffer can become a large hidden memory cost. | Optional downscaled, sampled bounded buffer with explicit byte budget and snapshot fallback. |
| Dependencies | Pinning names does not prove compatibility or prepare lazy-downloaded assets. | Verified candidate releases; P0 installation/smoke test and full resolved lock; offline first-use checks. |

## 4. Adversarial design walkthrough

These are manual state/contract checks, not executions of a detector.

| Challenge | Expected behavior under v2 | Review result |
|---|---|---|
| A person is horizontal from the first frame | No upright baseline or observed descent; no confirmed fall | Consistent |
| Stillness reaches 2.9 s, followed by a missing pose | Reset stillness; do not alert when elapsed wall time passes 3 s | Consistent |
| Imported video is paused for 20 s | Source clock does not advance | Consistent |
| A 30-FPS clip runs at 8 processed FPS offline | Process all frames; slower analysis, same source-time decision | Consistent; real-time gate separately fails |
| ByteTrack reuses an ID after a long gap | New application generation; reacquire upright baseline | Consistent |
| Two people cross during a candidate fall | Do not transfer event evidence across identities; measure switches | Residual tracking risk; empirical test required |
| Operator acknowledges a prone person's incident | Acknowledgement does not rearm the detector | Consistent |
| Subject remains down but moves | Candidate may persist to deadline; motion resets stillness | Consistent |
| Disk write or snapshot encoding fails | Visible alert remains; metadata/media save status reports failure | Consistent; integration test required |
| Demo lacks internet and model is missing | Fail locally before model construction; no hanging download | Consistent; offline test required |
| A healthy person deliberately reproduces the fall trajectory | May be observationally indistinguishable | Explicit limitation; cannot validate intent recognition |
| Camera is moved | Calibration invalid; stop/recalibrate/restart | Operational prerequisite; automatic camera-motion detection is not implemented |

## 5. Checks actually performed

Read the three supplied documents; inspected the supplied source directory; checked primary vendor/project documentation and selected release pages; inspected the pinned ByteTrack source; traced scenarios against the specified guards; and ran the artifact/math verification summarized in `validation-checks.json`.

The verification script checks the Markdown artifacts' required sections/phases, parses the illustrative JSON, checks timestamp ordering and units-related examples, verifies mathematical heat-index fixtures and the raw-buffer size calculation, and checks mirrored torso angles/time-normalized motion. These tests validate the written examples and internal contracts only. They are **not the application's unit tests and do not test pose quality or detection accuracy**.

| Verification level | Status |
|---|---|
| Actual request versus attached instructions | Reviewed and separated |
| Research/source claim audit | Completed, with unsupported claims identified |
| Architecture and failure-path consistency | Reviewed; corrections applied |
| Candidate release existence and core API support | Checked against primary sources |
| Example JSON and arithmetic | Checked by the accompanying verification artifact |
| Combined environment installation | Not run; P0 gate |
| Model/camera runtime benchmark | Not run; P1/P5 gates |
| Calibrated fall accuracy/false alerts | Not run; P5 gate |
| GUI/storage fault injection and offline rehearsal | Not run; P3/P4/P6 gates |
| Clinical efficacy or external EMS delivery | Not claimed; outside scope |

## 6. Approval to begin versus approval to present

**Begin implementation from v2 now.** Its chosen components and ordering are proportionate to the college demo and its remaining uncertainties have testable gates.

**Present as fully functional only after those gates pass.** In particular, demonstrate the real model-to-local-inbox path offline, report held-out event counts, and show the same frozen configuration on the actual demo computer. If a gate fails, record the failure and revise the implementation or explicitly narrow the demonstrated capability. A written validation cannot substitute for that evidence.
