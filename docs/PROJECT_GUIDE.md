# Vertebrate in plain language

## The project in 20 seconds

> “Vertebrate watches a webcam for fall-like posture. A pose model finds people and their joints. Our code follows each person, checks their posture across several frames, then looks at how they went down and whether they move afterwards. The window shows the urgency. A local siren and optional caregiver messages can draw attention to the event.”

### Follow one person from camera to alert

1. **See.** OpenCV reads a webcam image. YOLO11n-pose runs on the CPU and returns a rectangle around each person plus up to 17 joint locations. A joint location is called a keypoint.
2. **Check.** Vertebrate looks at the rectangle and at shoulders, hips and legs. It gives each person a posture label for this image.
3. **Confirm.** A tracker gives the person a temporary ID such as #2. Five fall-like observations for that ID confirm a fall. The count cannot be shared between people.
4. **Explain.** The app looks back at recent joint positions to estimate how #2 went down. It then watches for movement, getting up and people coming near.
5. **Respond.** The window shows the priority, the app logs the event, and configured alerts can reach a caregiver. The local siren has its own delay.

### What did we build ourselves?

“The pretrained model answers ‘Where are the people and their joints?’ Our Python code answers ‘Does this posture look like a fall, has it persisted for this person, and what should the caregiver see?’”

**Reading order:** posture and tracking on page 2, fall analysis on page 3, alerts on page 4, code and libraries on pages 5–6, and short practice answers on page 7.

## How five observations confirm a fall

### First, check one person in one image

**Too close.** Skip the fall rules if their rectangle covers more than 45% of the image, or if both its top and bottom are within 4 pixels of the image edges. A close-up of someone’s shoulders can otherwise look surprisingly wide.

**Wide body.** A vote counts if the rectangle’s width divided by height is greater than 1.2, and at least one hip plus one knee or ankle is visible. A 180 × 100 pixel box has a ratio of 1.8. It passes this rule only with the required joints and after the close-up check.

**Shoulders below hips.** The other way to earn a vote is for the average shoulder position to be more than 20 pixels below the average hip position. This uses both shoulders and both hips. Image coordinates increase downward. If neither rule matches, the code labels the pose “Standing”; sitting can receive this label too.

### Next, follow the same person

The tracker compares old and new rectangles. A pair qualifies if their overlap divided by their combined area (IoU) is at least 0.20, or their centres move by at most 35% of the larger rectangle’s diagonal. It assigns the best overlap first, using centre distance to break ties. IDs restart with the camera.

> **Count it aloud.** For #2, fall-like, fall-like, missed, fall-like, fall-like, fall-like gives 1, 2, 2, 3, 4, 5. The missed frame pauses the count. A matched standing or too-close observation resets it. Five observations have no fixed duration in seconds.

A confirmed track stays confirmed through three missed frames. The tracker retains its ID and count through 15 misses and drops them on miss 16. A nearby bystander who kneels with an upright torso can be labelled “Helping” and have their fall count suppressed. A helper who ends up lying down can still trigger a fall.

## How the app describes the fall

### How fast did they go down?

Each ID keeps up to eight seconds of joint positions, capped at 480 samples. The code follows the hips from their earlier standing level to their final level. It times the middle 10%–90% of the drop and divides that time by 0.65 to estimate the whole descent. Under 1.5 seconds is sudden; otherwise it is gradual. Estimates from 1 to 2 seconds also get a borderline flag.

If the person was already on the floor when first seen, the posture can still be confirmed. The descent is “unclear” because the camera missed the start. Missing hip points, a small drop or long gaps can also leave the type unclear.

### What pattern did the camera see?

**Sudden descent.** The code estimates which joints finished dropping first, whether the arms reached out, and the direction when visible. Hands first can suggest a trip; hips first can suggest a slip; knees or upper body first without an arm reaction can suggest a collapse. The output describes a possible pattern.

**Gradual descent.** The code checks, in order, for a hand held near the chest, signs of unsteadiness such as sideways hip sway, then a controlled descent such as lowering smoothly or pausing partway. Weak evidence leaves the pattern unclear.

### Did they move, get up or receive help?

**Movement.** The app compares averaged poses a second apart and uses the second-largest joint shift so one jittery joint has less influence. It measures shifts in torso lengths, using shoulder-to-hip distance as a ruler. The first 1.5 seconds after confirmation are treated as landing. Fifteen observed seconds of stillness can raise the priority.

**Recovery.** The code looks for at least one second standing with the hips raised by 0.8 torso lengths from confirmation. If it lacks a usable hip comparison, it falls back to the standing posture.

**Bystanders.** The app compares the bottom centres of people’s boxes. Within six torso lengths counts as nearby; within 2.5 for two seconds counts as being with them. After 20 seconds with someone nearby but nobody approaching or staying close, it can flag no response. These are visual estimates of proximity.

## What triggers each alert?

### How the window chooses urgency

For each person, the first matching rule below wins. For an event involving several people, the window uses the highest current priority among them.

**Recovered.** The person meets the getting-up rule.

**Critical.** The person has been still for 15 observed seconds, was previously critical for stillness and is currently without detected movement, shows a collapse pattern without current movement, or shows the hand-to-chest pattern.

**Check.** The descent looks controlled and its timing is outside the borderline range.

**Urgent.** The remaining confirmed cases, including unclear falls, receive this priority.

### The banner, Discord message and siren use different timing

**On confirmation:** the window starts the event and shows a fall. The first configured Discord message can wait up to about 1.5 seconds for the descent estimate to settle. Later updates can report prolonged stillness, movement returning, a helper arriving, no response, the siren or the end.

**While the event is open:** the siren uses elapsed event time. Its default delay is 30 seconds; the user can select 0 to disable it or up to 600 seconds. Critical priority alone does not immediately ring it. Silencing the siren leaves the event open.

**At the end:** after two seconds with nobody confirmed, the app closes the event using the first clear time as its end. Stopping the camera or closing the app can also end it. Overlapping falls share one event and one CSV row containing the affected IDs.

> **Example.** #2 is already lying down when monitoring starts. After five matching observations the app can show a confirmed fall with unclear type and Urgent priority. If it later observes 15 seconds of stillness, the priority becomes Critical. With the default setting, the siren rings when the event reaches 30 seconds.

## Which files do the work?

Start at **app.py**. The files below it in this list live in **app_modules/**. For a code walkthrough, follow the first six rows in order.

| File | What to say about it |
|---|---|
| app.py | Launches the Qt application, applies the theme and opens the window. |
| camerathread.py | Opens the webcam, loads local weights, runs the pose model, times inference and sends results to the window. Start at CameraThread.run(). |
| common.py | Contains the close-up guard, posture rules, drawing and shared settings. Read posture_state(), fallen_geometry() and annotate(). |
| tracking.py | Matches people across images, manages each ID’s streak and starts the post-fall watch. Read PersonTracker.update(). |
| fallanalysis.py | Keeps pose history and calculates descent speed, pattern, movement, bystanders and priority. Read PoseHistory, PostFallWatch and priority_of(). |
| mainwindow.py | Connects Start/Stop, images, reports, logging, alerts and the siren timer. Read show_frame() and check_alarm(). |
| falllog.py | Maintains the live event, merges overlapping IDs and writes a completed event to the device’s CSV. |
| falllogdialog.py | Shows live and past events. Clearing the log preserves a backup and keeps an ongoing event. |
| caregivers.py | Configures Discord, builds messages and sends them through a queue using a webhook or bot. |
| locationfeed.py | Prefers fresh external GPS, then system location, then an older receiver fix. The coordinates describe the device. |
| alarmsound.py | Plays or stops the local siren and provides a system-beep fallback. |

## Libraries, screen elements and storage

### Which library does what?

| Library | Job |
|---|---|
| OpenCV | Reads webcam images, draws boxes, joints and labels, and converts image colours. |
| Ultralytics + PyTorch | Load and run YOLO11n-pose on the CPU. The “n” means nano, the model size. |
| NumPy | Calculates rectangle overlap, joint distances, movement and timing from coordinates. |
| PySide6 / Qt | Runs the window, signals, timers, camera thread, positioning and sound. |
| Python 3.12 + standard library | Connects these pieces and handles CSV files, timestamps, queues and Discord HTTP requests. |

### Where do the remaining pieces live?

**Screen components.** videoview.py paints the live feed; urgencymeter.py draws the three-level Check / Urgent / Critical bar, which turns green on recovery. Before confirmation the sidebar can show a text count such as 3 of 5. stattile.py draws statistic cards; logomark.py draws the logo. __init__.py marks the package.

**Model and setup.** yolo11n-pose.pt contains the learned weights. The app looks beside app.py, then inside models/. models/manifest.json records provenance and the checksum. requirements.txt and requirements_macOS.txt list installation dependencies; requirements-win-cpu.lock preserves a historical Windows environment.

**What gets saved?** runtime/logs/falls/<device>.csv holds finished events, timings, IDs, priority, pattern, movement, help, location when available and alarm details. runtime/logs/performance/ holds session timing summaries. Webcam frames stay in memory. Configured Discord alerts send text and may include device location.

**Where is the evidence?** tests/test_fall_analysis.py checks synthetic poses. tests/test_fall_log.py checks log clearing and backups. tests/test_frame_signal.py checks that person-ID reports reach the window. README.md covers setup; CONTEXT.md explains the design; HANDOFF.md records verification gaps and known issues.

## Short answers to practise aloud

### Why use five observations?

“One odd camera frame can look like a fall. We require repeated evidence for the same person. Three votes from one person and two from another cannot trigger an alert.”

### What is the difference between confirmation and priority?

“Confirmation asks whether someone repeatedly looks fallen. Priority asks how urgently to check on them, using movement, the descent pattern and signs of getting up.”

### Why use separate threads?

“The camera thread runs the model and analysis. The main thread draws the window and handles controls. A third thread sends Discord messages. Only one display image can wait in the queue, so a slow window doesn’t build up old frames.”

### Can it work offline?

“Yes. With the weights installed, camera analysis, the event log and the siren work locally on the CPU. Discord needs internet access. Location is optional, and the operating system may use a network service for it.”

### Can it tell why someone fell?

“It estimates visible patterns. A hand near the chest can trigger Critical even if the person is holding a phone. A deliberate lie-down can resemble a fall, and a crowded view can cause ID swaps. We describe what the camera saw.”

### What accuracy can we claim?

“The repository has no validated real-world accuracy percentage. Its tests check specific behaviours. To measure accuracy, we would need labelled videos of falls and everyday activity, then count both missed falls and false alerts.”

**Verification for this guide.** Reviewed source revision 711abd4 against CONTEXT.md on 25 November 2026. All eight repository tests passed locally, including the Qt signal test. This review did not include a live camera or caregiver-delivery test.
