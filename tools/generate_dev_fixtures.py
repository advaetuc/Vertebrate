r"""Generate deterministic local synthetic clips; these are NOT accuracy data.

    .\.venv\Scripts\python.exe tools/generate_dev_fixtures.py

AVI/MJPG output is deterministic on the pinned environment. Other encoders may
produce different bytes; regenerated manifest hashes always describe real files.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from vertebrate.capture import OpenCVFrameSource
from vertebrate.evaluation.labels import (ClipManifestRow, DatasetManifest, FrameTimingSpec,
                                         PersonAnnotation, TimeInterval, save_dataset_manifest)


def generate(root: Path) -> DatasetManifest:
    folder = root / 'tests/fixtures/videos'
    folder.mkdir(parents=True, exist_ok=True)
    rows, reports = [], []
    fps, frame_count = 25.0, 100
    names = ('dev_empty_scene', 'dev_single_person_fall_sim', 'dev_two_person_adl_sim')
    for clip_id in names:
        path = folder / (clip_id + '.avi')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (640, 360))
        if not writer.isOpened():
            raise RuntimeError('Local OpenCV AVI/MJPG encoder unavailable')
        try:
            for i in range(frame_count):
                image = np.full((360, 640, 3), 24, dtype=np.uint8)
                cv2.line(image, (0, 315), (639, 315), (100, 100, 100), 2)
                if clip_id == names[1]:
                    # Upright 0..1s, descent 1..1.6s, down 1.6..4s.
                    p = min(1.0, max(0.0, (i - 25) / 15))
                    angle = p * np.pi / 2
                    hip = (280 + int(35 * p), 235 + int(65 * p))
                    head = (hip[0] + int(85 * np.sin(angle)), hip[1] - int(85 * np.cos(angle)))
                    cv2.circle(image, head, 14, (40, 210, 230), -1)
                    cv2.line(image, head, hip, (40, 210, 230), 10)
                    cv2.line(image, hip, (hip[0] - 30, 312), (40, 210, 230), 8)
                    cv2.line(image, hip, (hip[0] + 20, 312), (40, 210, 230), 8)
                elif clip_id == names[2]:
                    for x, color in ((150 + i, (100, 220, 50)), (470 - i, (230, 130, 60))):
                        cv2.circle(image, (x, 120), 14, color, -1)
                        cv2.line(image, (x, 138), (x, 245), color, 9)
                        cv2.line(image, (x, 245), (x - 18, 310), color, 7)
                        cv2.line(image, (x, 245), (x + 18, 310), color, 7)
                writer.write(image)
        finally:
            writer.release()
        reader = OpenCVFrameSource(path)
        try:
            reader.open()  # full actual decoded PTS/count verification
            timing = FrameTimingSpec(reader.fps, reader.frame_count, reader.frame_count / reader.fps, True)
            modes = sorted(reader.timing_modes)
        finally:
            reader.release()
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        people = ()
        if clip_id == names[1]:
            people = (PersonAnnotation('synthetic-person-1', True,
                       fall_onset_interval_s=TimeInterval(1, 1.2),
                       down_posture_interval_s=TimeInterval(1.6, 4),
                       stillness_eligible_interval_s=TimeInterval(1.6, 4),
                       initial_bbox_xyxy=(240., 130., 315., 315.)),)
        elif clip_id == names[2]:
            people = (PersonAnnotation('synthetic-person-1', False),
                      PersonAnnotation('synthetic-person-2', False))
        rows.append(ClipManifestRow(clip_id, path.relative_to(root).as_posix(), sha,
                    'synthetic-no-human-subject', clip_id, 'opencv-drawing-fixtures',
                    'synthetic-side-view', 'ambiguity_stress_test',
                    'Synthetic pipeline fixture; not human fall-detection accuracy evidence',
                    len(people), timing, None, people))
        reports.append({'clip_id': clip_id, 'sha256': sha, 'size_bytes': path.stat().st_size,
                        'fps': timing.fps, 'frame_count': timing.frame_count,
                        'duration_s': timing.duration_s, 'timing_modes': modes,
                        'codec': 'MJPG', 'dimensions': [640, 360]})
    manifest = DatasetManifest('1.0', 'dev', tuple(rows))
    save_dataset_manifest(manifest, root / 'data/manifests/dev.json')
    print(json.dumps(reports, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    generate(parser.parse_args().root.resolve())
