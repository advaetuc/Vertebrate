"""Explicit, online P0 setup only. Never imported by the application."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt'
target = ROOT / 'models' / 'yolo11n-pose.pt'
target.parent.mkdir(exist_ok=True)
if target.exists():
    raise SystemExit('Model already exists; refusing to replace it.')
temporary = target.with_suffix('.pt.download')
try:
    with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if temporary.stat().st_size == 0:
        raise RuntimeError('Downloaded file is empty')
    temporary.replace(target)
finally:
    temporary.unlink(missing_ok=True)
with target.open('rb') as stream:
    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = {'schema_version': '1.0', 'models': [{
    'name': target.name, 'relative_path': 'models/yolo11n-pose.pt', 'task': 'pose',
    'sha256': sha, 'size_bytes': target.stat().st_size, 'source_url': URL,
    'ultralytics_version': '8.3.203', 'default_imgsz': 512, 'license': 'AGPL-3.0',
}]}
(ROOT / 'models' / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest, indent=2))
