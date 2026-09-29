"""Record the installed environment without importing the model libraries."""
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys

from lock_wheels import write_lock

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
count = write_lock(root, root / 'runtime/wheelhouse')
(root / 'reports/environment.json').write_text(json.dumps(environment, indent=2) + '\n', encoding='utf-8')
print(f'Recorded {len(packages)} distributions; hash-locked {count} third-party packages.')
