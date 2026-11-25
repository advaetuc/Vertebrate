# VERTEBRATE — Handoff State

**As of:** 25 Nov 2026, 23:10 IST · **Branch:** `main` · **Last commit before this
file:** `30308cf` ("Add Clear log with confirmation to the fall log")

All planned features are built and committed. The fall analysis reaches the
window only since the 25 Nov hand-off fix, and it has been checked on synthetic
stick-figure falls only — the next person's first job is to verify it on real
hardware. Architecture and rules: [`CONTEXT.md`](CONTEXT.md). Setup and features
for users: [`README.md`](README.md).

## Status

| Area | State |
|---|---|
| Fall confirmation (5 frames, same person), close-range guard, tracking | Working on a real camera (7 falls logged on 25 Nov at 15 fps, MacBook Air) |
| Location watermark, fall log, emergency alarm, Discord alerts (fixed channel) | Working |
| Fall analysis: movement after a fall, people around, sudden vs gradual, patterns, priority | Built; reaches the window since 25 Nov; needs a camera retest |
| Urgency meter, Clear log | Built 25 Nov; needs a look on a real screen |
| Automated tests (`tests/`, 8 tests) | Analysis and fall-log tests pass with Qt stubbed; not yet run with the real PySide6 |

## Do these first

1. Run the tests: `.venv/bin/python -m unittest discover -s tests -v` (macOS) or
   `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` (Windows).
   All 8 should pass.
2. Camera retest: person A stands fully in view, then falls (onto a mat); person
   B stands about two body lengths away. Check A's label
   (`FALL CONFIRMED (<pattern>)`), B's label (`- near #<A>`), the Urgency card,
   the banner, the Fall log row (Priority, Type, After the fall, Help) and the
   Discord message. Keep the terminal's `[analysis] #…` line.
3. Remove the unused files and commit:
   `git rm app_modules/evidencemeter.py app_modules/debounce.py`.
4. Push `main` to GitHub (not checked when this file was written).

## Recent changes (newest first)

| When | Change |
|---|---|
| 25 Nov | Clear log with confirmation; old log moved to `runtime/logs/falls/cleared/` |
| 25 Nov | Urgency meter replaces the evidence meter |
| 25 Nov | **Fix:** `frame_ready`'s analysis argument `dict` → `object` — as a QVariantMap the int-keyed reports arrived empty, so fall type, priority and people nearby never reached the window, log or Discord |
| 25 Nov | Bystanders found near someone never seen standing; labels explain an unclear type; `[analysis]` console line; `tests/` added |
| 25 Nov | Phase 5B (`2c98c8f`): gradual patterns — hand to chest, unsteady, controlled |
| 25 Nov | Phase 5A (`498243e`): sudden patterns — collapse, slip, trip |
| 25 Nov | Phase 3 (`ba352d3`): sudden vs gradual, priority |
| 25 Nov | Phase 2 (`1ee7994`): people around a fallen person |
| 25 Nov | Phase 1 (`61b54be`): movement after a fall |
| 25 Nov | Phase 0 (`f69ddad`): pose history, real frame rate |

## Open issues

Full list with evidence: *Current known limitations* in `CONTEXT.md`.

- **Analysis untested on real footage.** Expect wrong or "unclear" labels.
- **Slip read as trip** in synthetic tests: hips land one frame before the
  hands, and the 0.1 s tie rule lets the hands win (`TIE_S`, `fall_pattern()`).
- **Phone or cup at chest height** triggers "Hand to chest" (Critical).
- **Tracker ID changes** during fast falls make the type "unclear".
- **Open decision:** should "no movement for 15 s" or a collapse ring the siren
  at once instead of waiting for the alarm delay?

## Next steps

1. Replay script: run the 100 URFD clips in `data/videos/urfd/` through the same
   pipeline and write one CSV row per clip (confirmed, type, pattern, priority,
   timing).
2. With that data, fix the slip/trip tie rule (tie only within one frame; hands
   win only when the arms clearly reached out) and re-run every test.
3. Record a few safe staged falls of each kind and check the labels.
4. Before any demo, check `runtime/logs/performance/`: the slowest frame rate
   should stay above about 8 fps.

## Secrets and local files

- `runtime/config/discord_webhook.txt` holds the Discord webhook URL. `runtime/`
  is Git-ignored. Never commit, print or share it.
- Logs: `runtime/logs/falls/` (fall log + `cleared/` backups),
  `runtime/logs/performance/`.
- Weights `yolo11n-pose.pt` (root or `models/`) and the URFD clips are local
  only.

## People

Commits so far come from the student team (GitHub: `samarthdombe`,
`rohitmanglani807-ai`, `maithiliashtaputre-lgtm`, `shivansh-vijayvergiya`);
original code © Advaet Chintakunta (MIT). Repository:
<https://github.com/advaetuc/Vertebrate>.
