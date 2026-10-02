"""Run the five P1B acceptance commands, preserving exact process outputs."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
arguments = [
    ['-m', 'pip', 'check'],
    ['-m', 'pytest', 'tests/unit/test_clocks.py', 'tests/integration/test_capture_tracking.py', '-q'],
    ['-m', 'vertebrate', 'benchmark', '--manifest', 'data/manifests/dev.json',
     '--device', 'cpu', '--output', 'reports/benchmark.json'],
    ['-m', 'pytest', '-q'],
    ['-m', 'vertebrate', 'doctor', '--offline'],
]
records = []
for args in arguments:
    result = subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=1800)
    record = {'command': '.\\.venv\\Scripts\\python.exe ' + subprocess.list2cmdline(args),
              'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    records.append(record)
    (root / 'reports/p1b-verification.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)
raise SystemExit(0 if all(r['exit_code'] == 0 for r in records) else 1)
