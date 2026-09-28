# Project VERTEBRATE — P0

A single local Python desktop application foundation. P0 implements immutable
configuration and offline preflight. Capture/GUI, benchmarking, and evaluation
remain explicit stubs for P4, P1B, and P5 respectively. No cloud services or paid
APIs are used. No fall-detection performance or hardware throughput is claimed.

## Environment and setup (Windows x64)

Run from this repository root. Python 3.12+ is required; the verified local
interpreter and resolved dependency versions are in `reports/baseline.md`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

Package installation is an explicit online setup operation. The lock pins the
entire resolved Windows CPU environment, including test/build tools. To install
air-gapped, first prepare a compatible wheelhouse, then use `--no-index
--find-links <wheelhouse>` with the lock. The lock is platform-specific and is
not a cryptographically hash-locked wheelhouse.

The local weight is `models/yolo11n-pose.pt`. If provisioning a new checkout
without this file, run the explicit online setup command:

```powershell
.\.venv\Scripts\python.exe tools/provision_model.py
```

This refuses to overwrite an existing model and records the downloaded file's
actual SHA-256 and size in `models/manifest.json`. The application never invokes
this script. Preserve/review the known manifest when transferring assets; a hash
recorded at download detects subsequent corruption, not independent provenance.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py tests/integration/test_offline_assets.py -q
.\.venv\Scripts\python.exe -m vertebrate doctor --offline
.\.venv\Scripts\vertebrate.exe doctor --offline
```

`tools/verify_p0.py` runs these gates and the three stubs, retaining stdout,
stderr, and actual exit codes in dated JSON reports. Integration tests require
real local weights and installed dependencies; they fail when absent.

Doctor always runs offline, even without the flag. It checks config and asset
integrity before importing the model library; probes outbox write/fsync/delete;
validates ByteTrack YAML against JSON; checks Qt, numerical and model imports;
rejects OpenCV conflicts; and performs real CPU pose inference and two-frame
ByteTrack assignment. Empty detections on the blank 512-square frame are valid;
controlled detection boxes separately exercise actual tracker association.
This smoke pass is not a detection-accuracy benchmark.

Network/HTTP/DNS and subprocess guards cover the pinned Python stack during
doctor. `YOLO_OFFLINE=1`, `ULTRALYTICS_OFFLINE=1`, and `YOLO_AUTOINSTALL=false`
are set before imports, and Ultralytics sync/integrations are disabled. Guards
temporarily use `YOLO_OFFLINE=true` for Ultralytics 8.3.203 because its source
recognizes only that spelling, then restore `1`; both public offline flags
remain set after doctor. The Python Windows-version lookup is cached before
subprocess denial because Python 3.12.0 calls the local `ver` command.
These guards are not a sandbox for arbitrary native code. Settings are local under
`runtime/logs/ultralytics`, avoiding user-global Ultralytics settings.

## Configuration and command contracts

`config/demo.json` uses 3-second stillness; `config/observation-10s.json` uses
10 seconds. Both contain all 34 explicitly enumerated calibration fields in the
P0 request (despite its heading saying 22), plus model, tracker and runtime
settings. No full blueprint document was present; the prompt is the source.

Frozen dataclasses reject unknown fields, duplicate JSON keys, invalid types,
non-finite numbers, invalid ranges and inconsistent time/threshold relationships.
Omitted known fields use declared defaults. Canonical hashes use sorted UTF-8 JSON
with compact separators and `allow_nan=False`. `increment_revision` accepts
section dataclasses or partial section dictionaries and increments exactly once.

All relative paths resolve from the working directory (run from project root).
The asset manifest path must remain inside that root; absolute runtime paths are
allowed for local storage. Outbox/logs are created on demand. The setup includes
`data/manifests` and `data/videos`; videos/outbox/logs are ignored by Git.

```powershell
.\.venv\Scripts\python.exe -m vertebrate --source 0 --config config/demo.json
.\.venv\Scripts\python.exe -m vertebrate doctor --offline --config config/observation-10s.json
.\.venv\Scripts\python.exe -m vertebrate benchmark --manifest data/manifests/dev.json --device cpu --output reports/benchmark.json
.\.venv\Scripts\python.exe -m vertebrate evaluate --manifest data/manifests/holdout.json --config config/demo.json --output reports/evaluation.json --fail-on-gates
```

Doctor returns 0 only for full success, 1 for preflight failure. Stubs return 2
with `not_implemented` and create no misleading results. A missing/corrupt weight
requires explicit local repair; it never causes a runtime download. Read
`docs/licenses.md` before redistributing code, dependencies, or weights.
