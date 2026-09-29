"""Run P1A acceptance and retain exact commands/stdout/stderr/exit codes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
arguments = [
    ['tools/generate_dev_fixtures.py'],
    ['-m', 'pip', 'check'],
    ['-m', 'pytest', 'tests/unit/test_clocks.py', '-q'],
    ['-m', 'pytest', 'tests/unit/test_config.py', 'tests/integration/test_offline_assets.py',
     'tests/unit/test_clocks.py', '-q'],
    ['-m', 'vertebrate', 'doctor', '--offline'],
]
records = []
for args in arguments:
    command = [sys.executable, *args]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
    record = dict(command=subprocess.list2cmdline(command), exit_code=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    records.append(record)
    print(json.dumps(record, indent=2), flush=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
path = root / 'reports' / f'p1a-verification-{stamp}.json'
path.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
raise SystemExit(0 if all(r['exit_code'] == 0 for r in records) else 1)
