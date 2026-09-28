"""Record the installed environment without importing the model libraries."""
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys

root = Path(__file__).resolve().parents[1]
distributions = sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower())
packages = []
for dist in distributions:
    packages.append({'name': dist.metadata['Name'], 'version': dist.version,
                     'license_expression': dist.metadata.get('License-Expression'),
                     'license': dist.metadata.get('License'),
                     'license_classifiers': [c for c in dist.metadata.get_all('Classifier', []) if c.startswith('License')],
                     'license_files': dist.metadata.get_all('License-File', []),
                     'project_urls': dist.metadata.get_all('Project-URL', []),
                     'home_page': dist.metadata.get('Home-page')})
environment = {'python': sys.version, 'executable': sys.executable, 'base_executable': sys._base_executable,
               'platform': platform.platform(), 'uname': platform.uname()._asdict(),
               'windows_version': list(sys.getwindowsversion()), 'pointer_bits': struct.calcsize('P') * 8,
               'logical_processors_reported_by_os': os.cpu_count(), 'packages': packages}
(root / 'reports/environment.json').write_text(json.dumps(environment, indent=2) + '\n', encoding='utf-8')
lock = ['# Resolved on Windows AMD64 / CPython 3.12.0. Includes build and test tools.',
        '# CPU PyTorch builds are intentional. Install project separately with --no-deps.',
        '--extra-index-url https://download.pytorch.org/whl/cpu', '']
lock += [f"{p['name']}=={p['version']}" for p in packages if p['name'] != 'vertebrate']
(root / 'requirements-win-cpu.lock').write_text('\n'.join(lock) + '\n', encoding='utf-8')
print(f'Recorded {len(packages)} distributions; locked {len(packages) - 1} third-party packages.')
