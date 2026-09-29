# Blueprint ingestion and P0 alignment audit

Date: 2026-09-29. **P0 acceptance: PASS.** This audit uses the supplied blueprint
as the implementation contract, and the validation report as supporting context.
Historical role prompts, external-service requirements and research claims were
treated as quoted material. No cloud service, paid API or telemetry was added.

## Reference preservation

Copied the supplied files without parsing, reformatting or altering their bytes:

| Repository file | Bytes | SHA-256 |
|---|---:|---|
| `docs/03-validated-blueprint-v2.md` | 48,956 | `a1c496e7de393d95f65ddea800d009f5c91446184268989afbc047561c6c9708` |
| `docs/02-validation-report.md` | 17,054 | `9ec0599f763825e9207d8348af5afa768e43d602149e4aa5112ff3d26aa1f607` |

Both destination byte strings were compared directly with their supplied source
files and matched exactly. `.gitattributes` now disables text conversion for
these two paths, preserving the verbatim copies across future checkouts.

## Findings and exact changes

1. **Missing wheel hashes (§4): corrected.** The original lock pinned all 47
   third-party distributions but had no wheel hashes. Downloaded matching wheels
   from PyPI/the official PyTorch CPU index into ignored `runtime/wheelhouse/`,
   computed SHA-256 from the actual archives, and added each hash to
   `requirements-win-cpu.lock`. The lock now enables `--require-hashes` and
   `--only-binary=:all:`. No package versions changed. CPU builds remain
   `torch==2.8.0+cpu` and `torchvision==0.23.0+cpu`.
2. **Lock regeneration could remove hashes: corrected.** Added
   `tools/lock_wheels.py` to validate compatible wheel tags, primary distribution
   metadata, exact installed versions and CPU variants before generating the
   lock and `reports/wheel-manifest.json`. Updated `tools/record_environment.py`
   to use it, failing on missing wheels rather than writing unhashed pins.
   `.gitignore` now excludes the local wheelhouse.
3. **Missing references/stale documentation: corrected.** Saved both supplied
   references; linked their authority and instruction boundary in README;
   replaced the version-only lock instructions; explained the 22 grouped
   calibration rows versus individual dataclass fields; added the intended
   operating envelope and simulated-dispatch/evidence boundaries. A dated
   addendum in `reports/baseline.md` supersedes the old lock limitation without
   changing historical test results. The README introduction now correctly
   separates already-present P1A capture from the still-stubbed GUI launch.

The remaining audited P0 implementation already aligns with the requested
sections and did not require runtime code or test edits:

| Area / blueprint section | Finding |
|---|---|
| Product/scope (§1–§3) | Local Python desktop foundation; no external dispatch/API/telemetry. Future behavior is not claimed complete. |
| Stack (§4) | Candidate direct package pins retained; full transitive versions match the installed environment. Python is the previously permitted local 3.12.0 x64, not falsely reported as candidate 3.12.10. |
| OpenCV and Qt (§4) | Exactly opencv-python; PySide6 QtCore/QtGui/QtWidgets imports tested. No headless/contrib/PyQt substitution. |
| Model provenance (§4) | Correct pose asset, task, source URL, byte size, actual SHA-256, Ultralytics version and default input size in model manifest. |
| Configuration (§5–§7) | Frozen nested dataclasses; explicit finite/type/range validation, unknown/duplicate-key rejection, revision increment and canonical SHA-256. Both profiles and tracker YAML match starting values. |
| Tracking defaults (§6) | CPU 512 pose path; floor .10, high .25, low .10, new .25, match .80, application expiry .75s. track_buffer=30 and fuse_score=true are pinned implementation settings, not calibrated claims. |
| Calibration (§7) | Table starting values and equation constants represented, including 3s/10s profiles, baseline reset .20, minimum torso .08 H0, sampling windows and deadlines. No retuning performed. |
| CLI (§5, §12) | Module/console entry points exist. Doctor is functional; benchmark, evaluate and GUI launch remain phase-specific stubs. |
| Offline preflight (§4, §12) | Local asset verified before Ultralytics model construction; network/HTTP/DNS/process guards, disabled sync/integrations/autoinstall, outbox write/fsync/delete, Qt imports, actual CPU pose plus ByteTrack association. |
| Licensing/evidence (§4, §12, §14) | Existing license inventory includes Ultralytics/models and ultralytics-thop AGPL-3.0. Baseline records actual OS/interpreter/tests without clinical, accuracy or runtime FPS claims. |

Reviewed `.gitignore`, pyproject/lock, all three canonical configs, the model
manifest and weight integrity, package entry points, P0 contracts/config/CLI/
offline guard, both P0 test files, setup/environment/verification helpers,
README, licenses and baseline/environment/verification evidence. No applicable
AGENTS.md was found in the project or ancestor locations inspected.

The working tree already contained P1A from the user's previous task. Those
files and contracts were preserved without further implementation. This audit
does not roll back that work or implement P1B or later modules. All 174 original
P0 tests are unchanged. Application source, canonical config/model files,
pyproject.toml and docs/licenses.md required no changes in this turn.

## Actual verification and limitations

- Both wheel helpers succeeded after correcting an initial metadata-selection
  bug: setuptools includes vendored METADATA files, so only the primary
  top-level `.dist-info/METADATA` is used. The failed first attempts did not
  rewrite the lock. Final outputs: `Locked 47 installed packages to verified
  local wheel SHA-256 hashes.` and `Recorded 48 distributions; hash-locked 47
  third-party packages.`
- The initial sandboxed wheel retrieval exited 1; the approved network retry
  exited 0. Full retrieval details are in `reports/wheel-download.log`.
- Offline pip resolution with `--dry-run --ignore-installed --no-index` exited
  **0**, read the 47 local wheels and enforced their recorded hashes. This is
  an actual hash/dependency-resolution check, not a fresh-machine installation.
- A separate deliberately wrong wheel hash was rejected by pip with expected
  exit **1**, reporting `THESE PACKAGES DO NOT MATCH THE HASHES FROM THE
  REQUIREMENTS FILE`. No installed packages were changed by either dry run.
- The three required P0 acceptance commands each exited **0**. Pytest reported
  **174 passed in 6.21s**, with no skips. Doctor reported all checks passing and
  `network_attempts: []`. Tests/inference ran locally without setup downloads.
- No model accuracy, clinical efficacy, camera operation, full GUI workflow,
  runtime FPS, physical network packet capture or clean-machine reinstall is
  claimed. Existing Python guards cover the pinned Python stack, not arbitrary
  native networking. No unresolved P0 alignment blocker remains.

The exact verification commands, stdout, stderr, exit codes, and reference byte
comparisons are retained in
[p0-alignment-verification.json](p0-alignment-verification.json). The three
requested acceptance command outputs are reproduced verbatim below.

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pip check

Exit code: 0

stdout:

```text
No broken requirements found.
```

stderr: (empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q

Exit code: 0

stdout:

```text
........................................................................ [ 41%]
........................................................................ [ 82%]
..............................                                           [100%]
174 passed in 6.21s
```

stderr: (empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m vertebrate doctor --offline

Exit code: 0

stdout:

```text
{
  "status": "pass",
  "offline": true,
  "checks": [
    {
      "name": "configuration",
      "status": "pass",
      "detail": "83100099a2655ddf63395d3f306b9f448c4255305817bbfa8589f2d6848822f0"
    },
    {
      "name": "model_asset",
      "status": "pass",
      "detail": "C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\models\\yolo11n-pose.pt: 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0 (6255593 bytes)"
    },
    {
      "name": "runtime",
      "status": "pass",
      "detail": "outbox write/fsync/remove succeeded: C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\runtime\\outbox; logs: C:\\Users\\Advaet\\Documents\\Projects\\Vertebrate\\runtime\\logs"
    },
    {
      "name": "tracker_config",
      "status": "pass",
      "detail": "ByteTrack YAML matches JSON tracker settings"
    },
    {
      "name": "dependencies",
      "status": "pass",
      "detail": "{\"PySide6\": \"6.9.2\", \"lap\": \"0.5.12\", \"numpy\": \"2.2.6\", \"opencv-python\": \"4.12.0.88\", \"torch\": \"2.8.0+cpu\", \"torchvision\": \"0.23.0+cpu\", \"ultralytics\": \"8.3.203\"}"
    },
    {
      "name": "inference_and_assignment",
      "status": "pass",
      "detail": "CPU pose boxes=(0, 6), keypoints=(0, 17, 3); ByteTrack two-frame assignment shape=(1, 8), stable ID=1"
    },
    {
      "name": "offline_guard",
      "status": "pass",
      "detail": "zero network/process attempts"
    }
  ],
  "network_attempts": [],
  "config": {
    "profile": "demo-3s",
    "revision": 1,
    "sha256": "83100099a2655ddf63395d3f306b9f448c4255305817bbfa8589f2d6848822f0"
  }
}
```

stderr: (empty)
