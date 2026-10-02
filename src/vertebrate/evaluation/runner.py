"""Local manifest replay using the same pipeline/FSM as future GUI consumers.

This P2 path emits candidate decisions, not P5 event matching/release metrics.
Usage: python -m vertebrate.evaluation.runner --manifest data/manifests/dev.json
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from ..config import config_sha256, load_config
from ..pipeline import HeadlessPipeline
from ..pose import PoseAdapter, guarded_runtime
from .labels import load_dataset_manifest


def run_manifest(manifest_path, config_path='config/demo.json', *, root_dir=None):
    root = Path(root_dir or Path.cwd()).resolve()
    attempts = []
    with guarded_runtime(root, attempts):
        config = load_config(root / config_path)
        manifest = load_dataset_manifest(root / manifest_path, verify_files=True, root_dir=root)
        from ..cli import _check_tracker
        _check_tracker(config, root)
        pose = PoseAdapter(config, root, headless=True)
        pose.warmup()
        clips = []
        for clip in manifest.clips:
            pipeline = HeadlessPipeline(config, root, pose=pose, frame_rate=clip.frame_timing.fps)
            # Stable replay identity makes incident IDs reproducible for the same
            # verified clip/config. Interactive capture retains fresh UUID sessions.
            session_id = hashlib.sha256((clip.clip_id + clip.sha256 + config_sha256(config)).encode()).hexdigest()
            result = pipeline.run(root / clip.relative_path, recorded_start_utc=clip.recorded_start_utc,
                                  session_id_fn=lambda value=session_id: value)
            if result.frames != clip.frame_timing.frame_count or result.dropped_frames:
                raise RuntimeError(f'{clip.clip_id}: lossless frame count failed')
            clips.append({'clip_id': clip.clip_id, 'sha256': clip.sha256, **asdict(result)})
    return {'schema_version': '1.0', 'partition': manifest.partition,
            'config_sha256': config_sha256(config), 'model_sha256': pose.asset.sha256,
            'clips': clips, 'network_attempts': attempts,
            'note': 'Source-time candidate decisions only; no accuracy or P5 release-gate claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/dev.json')
    parser.add_argument('--config', default='config/demo.json')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            report = run_manifest(args.manifest, args.config)
        text = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + '\n', encoding='utf-8')
        print(text)
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'fail', 'error': f'{type(exc).__name__}: {exc}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
