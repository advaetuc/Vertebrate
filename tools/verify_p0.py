"""Execute acceptance commands and retain verbatim stdout/stderr + exit codes."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, '-m', 'pip', 'check'],
    [sys.executable, '-m', 'pytest', 'tests/unit/test_config.py', 'tests/integration/test_offline_assets.py', '-q'],
    [sys.executable, '-m', 'vertebrate', 'doctor', '--offline'],
    [str(Path(sys.executable).with_name('vertebrate.exe')), 'doctor', '--offline'],
    [sys.executable, '-m', 'vertebrate'],
    [sys.executable, '-m', 'vertebrate', 'benchmark'],
    [sys.executable, '-m', 'vertebrate', 'evaluate', '--fail-on-gates'],
]
records = []
for command in commands:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
    item = {'command': subprocess.list2cmdline(command), 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}
    records.append(item)
    print(json.dumps(item, indent=2), flush=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
target = root / 'reports' / f'verification-{stamp}.json'
target.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
expected = [0, 0, 0, 0, 2, 0, 2]  # Benchmark implemented in P1B; GUI/evaluate remain stubs.
raise SystemExit(0 if [r['exit_code'] for r in records] == expected else 1)
