"""Negative-path tests plus genuine local CPU inference/assignment integration."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

import pytest

from vertebrate import cli
from vertebrate.config import AppConfig, config_to_dict
from vertebrate.contracts import AssetVerificationError, OfflineNetworkError, Status
from vertebrate.offline import offline_guard

ROOT = Path(__file__).resolve().parents[2]


def fixture_root(tmp_path, content=b'local test asset'):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/demo.json').write_text(json.dumps(config_to_dict(AppConfig())))
    (tmp_path / 'config/bytetrack.yaml').write_bytes((ROOT / 'config/bytetrack.yaml').read_bytes())
    model = tmp_path / 'models/yolo11n-pose.pt'
    if content is not None:
        model.write_bytes(content)
    entry = {'name': model.name, 'relative_path': 'models/yolo11n-pose.pt', 'task': 'pose',
             'sha256': hashlib.sha256(content or b'absent').hexdigest(), 'size_bytes': len(content or b'')}
    manifest = tmp_path / 'models/manifest.json'
    manifest.write_text(json.dumps({'schema_version': '1.0', 'models': [entry]}))
    return model, manifest


@pytest.mark.parametrize('content,reason', [(None, 'missing local model'), (b'', 'empty'),
                                         (b'changed', 'SHA-256 mismatch')])
def test_invalid_asset_fails_before_import_or_load(tmp_path, monkeypatch, capsys, content, reason):
    model, manifest = fixture_root(tmp_path, content)
    if content == b'changed':
        data = json.loads(manifest.read_text())
        data['models'][0]['sha256'] = '0' * 64
        manifest.write_text(json.dumps(data))
    with pytest.raises(AssetVerificationError, match=reason):
        cli.verify_model_asset(manifest, model.name, tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Dependencies/model must never be loaded for an invalid asset')
    monkeypatch.setattr(cli, '_check_dependencies', forbidden)
    monkeypatch.setattr(cli, '_smoke', forbidden)
    monkeypatch.chdir(tmp_path)
    assert cli.main(['doctor', '--offline']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['network_attempts'] == []
    assert reason in str(result['checks'])
    assert [c['name'] for c in result['checks']] == ['configuration', 'model_asset', 'offline_guard']


@pytest.mark.parametrize('field,value,reason', [
    ('sha256', 'UNAVAILABLE', 'placeholder'), ('sha256', 'A' * 64, 'invalid'),
    ('size_bytes', 999, 'size mismatch'), ('size_bytes', -1, 'nonnegative'),
    ('relative_path', '../escape.pt', 'escapes'), ('task', 'detect', 'task'),
])
def test_manifest_contract(tmp_path, field, value, reason):
    model, manifest = fixture_root(tmp_path)
    data = json.loads(manifest.read_text())
    data['models'][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(AssetVerificationError, match=reason):
        cli.verify_model_asset(manifest, model.name, tmp_path)


@pytest.mark.parametrize('payload', ['{oops', '[]', '{}', '{"schema_version":"1.0","models":[1]}'])
def test_malformed_manifest(tmp_path, payload):
    _, manifest = fixture_root(tmp_path)
    manifest.write_text(payload)
    with pytest.raises(AssetVerificationError):
        cli.verify_model_asset(manifest, 'yolo11n-pose.pt', tmp_path)


def test_unwritable_outbox_real_filesystem(tmp_path, monkeypatch, capsys):
    fixture_root(tmp_path)
    (tmp_path / 'runtime').mkdir()
    # A file where the directory must be is unwriteable as an outbox on Windows
    # and POSIX, without pretending chmod is an ACL on Windows.
    (tmp_path / 'runtime/outbox').write_text('blocks directory creation')
    monkeypatch.chdir(tmp_path)
    assert cli.main(['doctor', '--offline']) == 1
    result = json.loads(capsys.readouterr().out)
    assert 'writability failed' in str(result)
    assert result['network_attempts'] == []


def test_outbox_permission_denial_diagnostic(tmp_path, monkeypatch):
    fixture_root(tmp_path)
    def denied(*args, **kwargs):
        raise PermissionError('test-injected access denied')
    monkeypatch.setattr(cli.tempfile, 'NamedTemporaryFile', denied)
    result = cli.run_doctor(root_dir=tmp_path)
    assert result.status == Status.FAIL
    assert 'access denied' in str(result)


def test_network_guard_blocks_before_real_network(tmp_path, monkeypatch):
    reached = []
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: reached.append('socket'))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: reached.append('http'))
    with offline_guard(tmp_path) as attempts:
        for operation in (lambda: socket.create_connection(('example.com', 443)),
                          lambda: urllib.request.urlopen('https://example.com')):
            with pytest.raises(OfflineNetworkError):
                operation()
    assert reached == []
    assert len(attempts) == 2


def test_opencv_conflict(monkeypatch):
    class Dist:
        def __init__(self, name):
            self.metadata = {'Name': name}
    monkeypatch.setattr(cli.metadata, 'distributions', lambda: [Dist('opencv-python'), Dist('opencv-python-headless')])
    with pytest.raises(RuntimeError, match='conflicting'):
        cli._check_dependencies()


def test_tracker_yaml_mismatch(tmp_path):
    fixture_root(tmp_path)
    path = tmp_path / 'config/bytetrack.yaml'
    path.write_text(path.read_text().replace('0.25', '0.35'))
    result = cli.run_doctor(root_dir=tmp_path)
    assert result.status == Status.FAIL
    assert 'must match' in str(result)


@pytest.mark.integration
def test_real_local_asset_and_offline_doctor():
    # This is intentionally a failure, not a skip, if setup did not provision assets.
    record = cli.verify_model_asset(ROOT / 'models/manifest.json', 'yolo11n-pose.pt', ROOT)
    assert record.size_bytes > 0
    result = subprocess.run([sys.executable, '-m', 'vertebrate', 'doctor', '--offline'],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'pass'
    assert report['network_attempts'] == []
    assert any(c['name'] == 'inference_and_assignment' and c['status'] == 'pass' for c in report['checks'])
