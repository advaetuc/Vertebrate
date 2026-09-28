# VERTEBRATE Phase P0 baseline

Recorded 2026-09-28. **P0 acceptance: PASS.** No benchmarks or detection-accuracy
claims were produced. All verification used the project virtual environment.

## 1. Repository and environment inspection

The root `C:\Users\Advaet\Documents\Projects\Vertebrate` was empty, including
hidden entries. It had no Git repository, Python code, virtual environment,
tests, models, or instructions. Parent directory listings were inspected before
writing files. Adjacent projects were DeformX, Mirage, Research and TULYA; the
parent also contained `mirage-1.0.0.zip`. None was modified.

No AGENTS.md existed in the project or its ancestor chain through C:\.
TULYA's AGENTS.md is outside this project's scope. Searches of the project
parent, user Downloads and `.cache`, including hidden/ignored files, found no
`yolo11n-pose.pt` or `03-validated-blueprint-v2.md`. They did find unrelated
TULYA virtual environments/tests and a cached Hugging Face `ehm_model_stage1.pt`;
these were preserved. The explicit P0 prompt supplied the configuration contract.
This was a targeted cache search, not a claim to have searched every disk.

| Fact | Observed value |
|---|---|
| OS | Windows 11, kernel 10.0.26200, AMD64; Python reports Windows-11-10.0.26200-SP0 |
| Python | 3.12.0 (tags/v3.12.0:0fb18b0, Oct 2 2023, 13:03:39), MSC v.1935, 64-bit AMD64 |
| Base interpreter | C:\Program Files\Python312\python.exe |
| Virtual environment | C:\Users\Advaet\Documents\Projects\Vertebrate\.venv |
| Other registered Python versions | 3.11 and 3.9; unqualified system `python` was 3.9 |
| Processor reported by Python | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| Logical processors reported by os.cpu_count | 32 |
| pip | 23.2.1 |
| Installed distributions | 48 including editable VERTEBRATE; 47 pinned third-party packages |
| Network during setup | Available after approved sandbox escalation; downloaded packages and official model release |
| Network during successful doctor | Zero attempted calls recorded by Python network/process guards |

`Get-ComputerInfo` exposed legacy registry labels Windows 10 Pro / 2009 and empty
OS detail fields. The exact kernel facts above come from Python's Windows API
and platform queries. CIM processor/memory queries returned `Access denied`;
no RAM measurement or hardware performance result is claimed.

The first sandboxed CPU package install exited 1 with WinError 10013 (socket
access forbidden), followed by pip's misleading no-matching-distribution text.
The same install succeeded after network permission. This was not an air-gapped
machine and did not require changing the requested dependency versions.

## 2. Resolved stack and local asset

| Package | Verified installed version |
|---|---|
| ultralytics | 8.3.203 |
| torch | 2.8.0+cpu |
| torchvision | 0.23.0+cpu |
| numpy | 2.2.6 |
| opencv-python | 4.12.0.88 (only OpenCV distribution) |
| PySide6 / Addons / Essentials / shiboken6 | 6.9.2 |
| lap | 0.5.12 |
| pytest | 8.4.2 |
| PyYAML | 6.0.2 (explicit tracker YAML parser dependency) |
| setuptools / wheel | 80.9.0 / 0.45.1 |

The complete resolved transitive stack and license metadata are in
[environment.json](environment.json); the installable version lock is
[requirements-win-cpu.lock](../requirements-win-cpu.lock). CPU build suffixes
are intentional, not version substitutions. This is a version lock, not a
hash-locked artifact repository. Fresh-machine/air-gapped reinstallation was
not performed; the lock was checked against this installed environment using
pip's offline dry-run, which exited 0 with every requirement already satisfied.

Actual downloaded asset:

- Path: `models/yolo11n-pose.pt`
- Size: **6,255,593 bytes**
- SHA-256: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`
- Source: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
- License: AGPL-3.0

The hash was measured from the actual file and checked again by doctor. It is
not presented as an independently signed upstream checksum.

## 3. Setup commands actually executed

Commands below ran from the project root; the initial `py` command creates the
environment and all subsequent Python work uses its executable.

| Exact command | Exit code / observed result |
|---|---|
| `py -3.12 -m venv .venv` | 0; no output |
| `.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu` | First 1 (sandbox WinError 10013); approved retry 0 |
| `.\.venv\Scripts\python.exe -m pip install ultralytics==8.3.203 numpy==2.2.6 opencv-python==4.12.0.88 PySide6==6.9.2 lap==0.5.12 pytest==8.4.2 PyYAML==6.0.2 setuptools==80.9.0 wheel==0.45.1 --log reports/pip-setup.log` | 0; full pip setup log retained |
| `.\.venv\Scripts\python.exe tools/provision_model.py` | 0; printed the manifest, including the exact size/hash above |
| `.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .` | 0; `Successfully installed vertebrate-0.1.0` |
| `.\.venv\Scripts\python.exe tools/record_environment.py` | 0; `Recorded 48 distributions; locked 47 third-party packages.` |
| `.\.venv\Scripts\python.exe -m pip install --dry-run --no-index --no-deps -r requirements-win-cpu.lock` | 0; all 47 pins already satisfied |
| `.\.venv\Scripts\python.exe tools/verify_p0.py` | First 1, then 0; exact child command outputs below and in dated JSON records |

The first PyTorch resolution temporarily selected NumPy 2.5.2; installation of
the requested stack replaced it with the required 2.2.6 before testing. Pip
suggested an update from 23.2.1; pip was intentionally kept and locked at the
verified version. Download progress is not reproduced here; the retained
[pip setup log](pip-setup.log) contains detailed package installation output.

## 4. Acceptance gates and scope

| Requirement | Status | Evidence |
|---|---|---|
| Package layout / both CLI entry points | PASS | Editable installation; module and console doctor exit 0 |
| Local Python >=3.12 / CPU stack / lock | PASS | Exact environment recorded; CPU builds and pin consistency checked |
| Frozen validated config and deterministic digest/revisions | PASS | Unit tests for both profiles, all 34 calibration fields, invalid numbers/keys/types/relationships |
| Canonical JSON and ByteTrack YAML | PASS | Both profiles loaded; YAML checked against JSON parameters |
| Model integrity manifest and license attribution | PASS | Actual file hash/size verified; docs/licenses.md and metadata inventory |
| Runtime directory creation / outbox writability | PASS | Real write, flush, fsync, removal; negative filesystem and permission-denial cases |
| Required imports and one OpenCV distribution | PASS | QtCore/QtGui/QtWidgets, NumPy, OpenCV, torch, torchvision, lap, Ultralytics |
| Offline local pose + ByteTrack smoke | PASS | Real model.track CPU inference plus real two-frame BYTETracker update; no success mocks |
| Missing/corrupt model fails before model loading | PASS | Exceptions, nonzero CLI result, no dependency/model load, zero guarded network calls |
| `python -m pip check` | PASS | Exit 0, `No broken requirements found.` |
| Required pytest command | PASS | Exit 0, **174 passed**, zero skipped |
| `python -m vertebrate doctor --offline` | PASS | Exit 0, all checks pass, empty network_attempts |
| Benchmark / evaluate / GUI stubs | PASS (P0 contracts only) | Exit 2 with explicit phase-specific not_implemented messages |

The blank-frame model output was boxes `(0, 6)` and keypoints `(0, 17, 3)`.
This is a valid empty detection result, not evidence of person/fall detection.
Separately, controlled detection boxes were passed through actual ByteTrack on
two frames; assignment shape was `(1, 8)` and identity 1 persisted. The success
integration test starts a fresh Python process using real weights and libraries.
The injected PermissionError test is explicitly a diagnostic test, not proof of
a real Windows ACL change; the separate outbox-as-file test uses the real filesystem.

## 5. Deviations, warnings and limits

- Python 3.12.0 x64 was used instead of candidate 3.12.10, as explicitly allowed.
- All requested package versions were retained; PyYAML/build tools and all
  transitive dependencies are pinned in addition.
- The prompt says 22 calibration values but enumerates 34. All 34 were implemented.
- Ultralytics 8.3.203 checks for `YOLO_OFFLINE=true`, not `1`. Doctor sets the
  requested `1` flags, temporarily uses `true` during guarded library work,
  and restores `1` afterward. Its dedicated settings file disables sync and
  integrations before the first import; ONLINE and AUTOINSTALL are disabled.
- Python 3.12.0 Windows platform discovery runs local `ver`. That result is cached
  before subprocess denial. Guard counters refer to the guarded preflight body;
  they are not OS-level packet capture or protection against arbitrary native code.
- The first verification run failed due to two test decorator syntax errors and
  blocking that local platform lookup. Both were fixed. Its failed outputs are
  preserved in `verification-20260928T165353Z.json`; they are not counted as passes.
- A source inspection attempted before the package had finished installing failed
  with file-not-found, then succeeded after installation; no package code was edited.
- No GUI window, real camera/video, detection accuracy, production packaging, or
  throughput benchmark was tested. Those are outside P0. No remaining P0 blocker.

## 6. Files created

All listed project files are new; no existing work was modified. `.venv` contains
the installed environment, and ignored runtime directories contain generated
local settings. Individual third-party environment files are not source deliverables.

| File | Purpose |
|---|---|
| `.gitignore` | Excludes requested runtime/media/environment/cache paths and build metadata |
| `pyproject.toml` | Package layout, pinned direct dependencies, test extra, console entry point |
| `requirements-win-cpu.lock` | 47 exact resolved third-party version pins and CPU index |
| `README.md` | Setup, verification, configuration and command contracts |
| `docs/licenses.md` | Third-party/model attribution and license references |
| `reports/baseline.md` | This structured inspection and acceptance report |
| `reports/environment.json` | Exact interpreter/platform and installed distribution/license facts |
| `reports/pip-setup.log` | Detailed dependency installation transcript |
| `reports/verification-20260928T165353Z.json` | Initial failed verification, exact stdout/stderr/exit codes |
| `reports/verification-20260928T165431Z.json` | Successful verification, exact stdout/stderr/exit codes |
| `config/demo.json` | Complete canonical 3-second stillness profile |
| `config/observation-10s.json` | Complete canonical 10-second stillness profile |
| `config/bytetrack.yaml` | Pinned tracker thresholds, buffer, matching and fuse_score |
| `models/manifest.json` | Actual local model metadata, hash and size |
| `models/yolo11n-pose.pt` | Official downloaded YOLO11n pose weights |
| `src/vertebrate/__init__.py` | Package identity/version |
| `src/vertebrate/__main__.py` | Module entry point |
| `src/vertebrate/contracts.py` | Immutable status/config/doctor/asset records and exceptions |
| `src/vertebrate/config.py` | Frozen dataclasses, strict parsing, validation, hashing and revisions |
| `src/vertebrate/cli.py` | CLI, asset verification, preflight, inference and assignment smoke |
| `src/vertebrate/offline.py` | Network/HTTP/DNS/subprocess guards and offline environment |
| `tests/unit/test_config.py` | Canonical configuration and invalid-input tests |
| `tests/integration/test_offline_assets.py` | Asset/runtime/offline failure tests and real smoke integration |
| `tools/provision_model.py` | Explicit online setup download; never invoked by runtime |
| `tools/record_environment.py` | Reproduce metadata inventory and resolved version lock |
| `tools/verify_p0.py` | Run and archive exact acceptance commands/results |
| `data/manifests/.gitkeep` | Retain the manifest directory in source control |

Directories created: `runtime/outbox/`, `runtime/logs/`, `data/manifests/`,
`data/videos/`. Outbox/logs are also created on demand by doctor.

## 7. Successful verification: exact commands, exit codes and verbatim outputs

The following is rendered from the retained successful JSON transcript. Empty
stderr is explicitly recorded; newline formatting is displayed as text.

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pip check

Exit code: 0

stdout:

```text
No broken requirements found.
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q

Exit code: 0

stdout:

```text
........................................................................ [ 41%]
........................................................................ [ 82%]
..............................                                           [100%]
174 passed in 5.98s
```

stderr:

(empty)

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

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\vertebrate.exe doctor --offline

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

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m vertebrate

Exit code: 2

stdout:

```text
{"status": "not_implemented", "message": "GUI launch not implemented until P4", "profile": "demo-3s", "config_sha256": "83100099a2655ddf63395d3f306b9f448c4255305817bbfa8589f2d6848822f0"}
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m vertebrate benchmark

Exit code: 2

stdout:

```text
{"status": "not_implemented", "message": "Not implemented until P1B"}
```

stderr:

(empty)

### C:\Users\Advaet\Documents\Projects\Vertebrate\.venv\Scripts\python.exe -m vertebrate evaluate --fail-on-gates

Exit code: 2

stdout:

```text
{"status": "not_implemented", "message": "Not implemented until P5"}
```

stderr:

(empty)
