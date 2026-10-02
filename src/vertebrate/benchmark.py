"""Measured, serial CPU Capture -> Pose -> Tracking on verified local clips."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import monotonic_ns, perf_counter

import numpy as np

from .capture import CaptureSessionManager
from .config import config_sha256, load_config
from .contracts import EndOfStreamEvent, FramePacket
from .evaluation.labels import load_dataset_manifest
from .pose import PoseAdapter, guarded_runtime
from .tracking import ByteTrackManager


def timing_summary(values):
    if not values:
        raise ValueError('Cannot benchmark an empty frame sequence')
    return {'mean_ms': float(np.mean(values)), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(np.max(values))}


def run_benchmark(manifest_path='data/manifests/dev.json', device='cpu',
                  output='reports/benchmark.json', config_path='config/demo.json',
                  image_sizes=(416, 512, 640), root_dir=None):
    root = Path(root_dir or Path.cwd()).resolve()
    if device != 'cpu':
        raise ValueError('P1B benchmark supports --device cpu only')
    if not image_sizes or any(type(n) is not int or n <= 0 or n % 32 for n in image_sizes):
        raise ValueError('Benchmark image sizes must be positive multiples of 32')
    cfg = load_config(root / config_path)
    attempts = []
    started_run = perf_counter()
    with guarded_runtime(root, attempts):
        manifest = load_dataset_manifest(root / manifest_path, verify_files=True, root_dir=root)
        if manifest.partition != 'dev' or not manifest.clips:
            raise ValueError('Benchmark requires a nonempty development manifest')
        from .cli import _check_tracker
        _check_tracker(cfg, root)
        runs = []
        for imgsz in image_sizes:
            run_cfg = replace(cfg, model=replace(cfg.model, imgsz=imgsz, device=device))
            started = perf_counter()
            pose = PoseAdapter(run_cfg, root)
            initialization_ms = (perf_counter() - started) * 1000
            pose.warmup()
            timings = {name: [] for name in ('capture', 'pose', 'tracking', 'total', 'frame_age_at_completion')}
            keys, identities = set(), set()
            detections = valid_keypoints = matched_samples = dropped = 0
            clip_reports = []
            for clip in manifest.clips:
                capture = CaptureSessionManager(root / clip.relative_path,
                                                recorded_start_utc=clip.recorded_start_utc)
                tracker = ByteTrackManager(cfg.tracker, root, clip.frame_timing.fps)
                count = 0
                clip_detections = clip_keypoints = clip_samples = 0
                clip_start = perf_counter()
                try:
                    capture.start()
                    if not np.isclose(capture.clock.verified_fps, clip.frame_timing.fps, rtol=0, atol=1e-6):
                        raise ValueError(f'Manifest FPS differs from verified file FPS: {clip.clip_id}')
                    tracker.handle_event(capture.queue.get())
                    capture_startup_ms = (perf_counter() - clip_start) * 1000
                    while True:
                        begin = perf_counter()
                        capture.pump()
                        item = capture.queue.get()
                        captured = perf_counter()
                        if isinstance(item, EndOfStreamEvent):
                            tracker.handle_event(item)
                            break
                        if not isinstance(item, FramePacket):
                            raise RuntimeError('Unexpected capture lifecycle event')
                        observation = pose.infer(item)
                        inferred = perf_counter()
                        samples = tracker.update(observation)
                        tracked = perf_counter()
                        timings['frame_age_at_completion'].append(
                            (monotonic_ns() - item.acquired_monotonic_ns) / 1e6)
                        for name, elapsed in (
                            ('capture', captured - begin), ('pose', inferred - captured),
                            ('tracking', tracked - inferred), ('total', tracked - begin)):
                            timings[name].append(elapsed * 1000)
                        count += 1
                        detections += len(observation.boxes_xyxy)
                        valid_keypoints += int(observation.keypoint_valid_mask.sum())
                        matched_samples += len(samples)
                        clip_detections += len(observation.boxes_xyxy)
                        clip_keypoints += int(observation.keypoint_valid_mask.sum())
                        clip_samples += len(samples)
                        for sample in samples:
                            key = sample.person_key
                            identities.add((key.session_id, key.tracker_id))
                            keys.add((key.session_id, key.tracker_id, key.generation))
                    if count != clip.frame_timing.frame_count or capture.queue.dropped_frames_count:
                        raise RuntimeError(f'Lossless frame count failed for {clip.clip_id}')
                    dropped += capture.queue.dropped_frames_count
                    clip_reports.append({'clip_id': clip.clip_id, 'sha256': clip.sha256,
                                         'frames': count, 'capture_startup_ms': capture_startup_ms,
                                         'detections': clip_detections, 'valid_keypoints': clip_keypoints,
                                         'matched_samples': clip_samples,
                                         'last_source_t_s': item.last_source_t_s,
                                         'recorded_start_utc': clip.recorded_start_utc})
                finally:
                    capture.stop()
                attempts.extend(tracker.network_attempts)
            attempts.extend(pose.network_attempts)
            runs.append({'imgsz': imgsz, 'config_sha256': config_sha256(run_cfg),
                         'initialization_ms': initialization_ms,
                         'model_load_ms': pose.model_load_ms, 'warmup_ms': pose.warmup_ms,
                         'frames': len(timings['total']), 'clips': clip_reports,
                         'stage_timings': {name: timing_summary(v) for name, v in timings.items()},
                         'processed_fps': 1000 * len(timings['total']) / sum(timings['total']),
                         'dropped_frames': dropped, 'detections': detections,
                         'keypoint_availability': {'valid': valid_keypoints, 'possible': 17 * detections,
                             'fraction': valid_keypoints / (17 * detections) if detections else None},
                         'matched_samples': matched_samples, 'track_count': len(identities),
                         'generation_count': len(keys), 'model_sha256': pose.asset.sha256})
        import torch
        report = {'schema_version': '1.0', 'status': 'pass',
                  'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  'device': device, 'precision': 'float32', 'batch': 1,
                  # platform.platform() repeats a Windows `ver` subprocess.
                  # Use uname fields cached by offline_guard and local env only.
                  'environment': {'python': platform.python_version(),
                                  'platform': ' '.join((platform.system(), platform.release(),
                                                        platform.version(), platform.machine())),
                                  'processor': os.environ.get('PROCESSOR_IDENTIFIER'),
                                  'torch_threads': torch.get_num_threads(),
                                  'versions': {n: importlib.metadata.version(n) for n in
                                               ('numpy', 'opencv-python', 'torch', 'torchvision', 'ultralytics', 'lap')}},
                  'manifest_sha256': hashlib.sha256((root / manifest_path).read_bytes()).hexdigest(),
                  'methodology': 'Serial lossless file capture, pose, association. One warmup frame per size; '
                      'startup includes imports/load and per-clip full CFR verification. Frame timings include '
                      'guard and immutable-copy overhead; exclude startup/warmup/EOF/report serialization. '
                      'All measured frames included, no outlier removal. Model loads after the first share '
                      'process/OS caches. Synthetic drawings measure pipeline throughput, not accuracy. '
                      'Frame age is local acquisition-to-completion, not historical video age or camera latency. '
                      'Null keypoint fraction means no detections; no observed track is not an accuracy result.',
                  'runs': runs, 'network_attempts': attempts,
                  'elapsed_ms': (perf_counter() - started_run) * 1000}
    if attempts:
        raise RuntimeError(f'Benchmark made unexpected offline attempts: {attempts}')
    destination = root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
