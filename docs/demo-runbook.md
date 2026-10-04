# VERTEBRATE physical rehearsal runbook

This is a procedure for upcoming human calibration, not evidence that rehearsals
or release gates have passed. Use blueprint §§1, 13 and 14 for the full criteria.

## Prepare

1. Obtain participant agreement and retain dataset permissions/notices. Use
   prerecorded falls or safe, supported acting with appropriate supervision.
   Never ask anyone to perform an uncontrolled fall.
2. Transfer the frozen environment, trusted weights/manifest and allowed clips.
   Follow README's offline install steps. Disconnect networking, run
   python -m vertebrate doctor --offline with the project virtual environment,
   and retain the actual output. Missing assets must fail locally.
3. Record hardware/camera/driver facts, code revision, canonical config hash,
   model/video/manifest hashes and room setup. Distinguish unknown hardware from
   measured facts; do not reuse historical performance as current evidence.
4. Fix the camera at a level oblique/side view of a clear marked floor. Confirm
   full-body visibility, including both feet, before and after planned motions.
   Stay within one or two people; check lighting and clutter.
5. Calibrate on development footage: upright baseline stability, stationary
   jitter, rapid sits/bends, slow lies, supported falls and recovery. Preserve
   source-time labels and group-separated dev/holdout partitions. Freeze the
   selected config before holdout; do not tune on headline holdout errors.
6. Prepare daily activity, recovery, supported fall, two-person, already-down,
   visibility-gap and early-EOF recordings. Include accepted forward/back/side
   views. Verify CFR timing and hashes. Do not append frozen frames.
7. Run doctor against the chosen config and verify writable outbox/free space.
   Inspect retained failed records before starting. A separate rehearsal outbox
   can keep records organized.
8. Enter context as manual_measured only after measurement, or manual_simulated
   for a demonstration. Missing, invalid or stale (30-minute) context must remain
   explicit and cannot prevent detection.
9. If clips are wanted, enable runtime.enable_optional_clip in a separate stopped
   config and advance its revision. Keep the 128 MiB cap. Check local MJPG output
   with a rehearsal incident; snapshot/JSON remains the supported fallback.

## Launch

1. Unset QT_QPA_PLATFORM before a visible desktop session. From the repository:

   ~~~powershell
   .\.venv\Scripts\python.exe -m vertebrate --config config/demo.json
   ~~~

2. Point out “Local demo — no external emergency dispatch.” Explain the ordered
   descent/posture/stillness evidence and the delay before confirmation. There
   is no diagnosis or external dispatch.
3. Select demo-3s. Play daily activity, then recovery before confirmation. Expect
   no confirmed incident in the supported calibrated setup; log discrepancies.
4. Play a supported fall. Observe fresh upright acquisition, “Possible fall —
   observing,” then one “Suspected fall with sustained stillness — check the
   person” alert after sufficient observed stillness.
5. Open the incident. Inspect identity/generation, source onset/down/stillness/
   confirmation times, analysis UTC versus recording UTC, frozen config,
   context provenance/staleness, snapshot and saved status. Acknowledge and
   verify that the existing record updates without duplication.
6. With optional clips, verify JSON/snapshot arrives without waiting for post-roll.
   Allow three more source seconds, then inspect clip_status and open clip.avi
   locally. Check clip-timestamps.json for requested/available bounds and
   truncation. Playback cannot establish missing detection evidence.
7. Show the two-person case: only the correct person's evidence may confirm.
   Note the outside-tested-limit warning if more than two people appear.
8. Pause a file for 20 wall-clock seconds: source time and stillness credit must
   not advance. Resume, then replay; replay requires a fresh session/baseline.
9. Repeat the complete recorded workflow with networking disconnected. Try a
   planned webcam scenario with the same frozen config only after the recorded
   sequence succeeds.
10. Measure processing FPS separately from source FPS and elapsed source time.
    Complete three rehearsals plus blueprint §13 live, delivery, lifecycle and
    soak measurements before claiming release gates pass. If hardware misses
    live gates, accurately label the demonstration prerecorded analysis.

## Demonstrate failure honestly

1. Obscure required landmarks or end a clip before stillness completes. Expect
   reset/inconclusive evidence, not confirmation from missing poses or EOF
   continuation. Brief invalid gaps erase stillness credit.
2. Begin already down: show acquisition/informational status, not an unarmed fall.
3. Include safe rapid deliberate lying down as an ambiguity stress case. If it
   alerts, record it separately; do not claim intent recognition or hide the
   outcome by retuning holdout thresholds.
4. Show missing or simulated context with its proper label. Context cannot prove
   that heat caused a fall.
5. Stop before optional post-roll ends: preserve available footage and mark
   truncation. Explain codec failure as snapshot plus JSON. If metadata storage
   actually fails, show the visible fault/retained alert, repair disk access and
   retry without pretending delivery already succeeded.
6. Select observation-10s only while stopped and use a separate sufficiently long
   clip. Never append a frozen last frame to achieve the longer duration.
7. Do not damage the sole model or incident copies for fault demonstrations. Use
   a separate deliberately missing-asset config/outbox and restore the approved
   configuration afterward.

## Afterward

1. Stop capture; wait for cooperative owner shutdown and queued critical writes.
   A driver timeout is a recorded failure, not successful resource release.
2. Inspect complete incident.json records, snapshots, optional clips/timestamps,
   acknowledgements and failure reasons. Interrupted post-roll becomes explicitly
   unavailable on restart. Temporary/incomplete files are not successful records.
3. Save actual rehearsal observations with the frozen revision. Keep synthetic
   tests, injected failures, human calibration and unseen holdout results distinct.
4. Delete rehearsal incidents/private recordings when no longer needed under the
   agreed retention policy. Verify cleanup targets; preserve consented evidence
   that is still required.
5. Do not commit participant videos, private outbox records or secrets. Keep
   permissions/notices. Local storage alone is not compliance certification.

