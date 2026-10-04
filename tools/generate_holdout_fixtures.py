"""Deterministic synthetic scoring fixtures, never human release evidence.

No frames are frozen/appended to real footage: every frame is locally drawn.
Annotations describe the programmed geometry independently of model predictions.
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from vertebrate.capture import OpenCVFrameSource
from vertebrate.evaluation.labels import (ClipManifestRow, DatasetManifest, FrameTimingSpec,
    PersonAnnotation, TimeInterval, save_dataset_manifest)


def generate(root):
    root = Path(root).resolve()
    folder = root/'tests/fixtures/holdout'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ('empty', 'adl_two', 'fall', 'excluded_fall'):
        name = 'holdout_'+kind
        path = folder/(name+'.avi')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25., (640, 360))
        if not writer.isOpened(): raise RuntimeError('Local MJPG encoder unavailable')
        try:
            for index in range(200):
                t = index/25
                image = np.full((360, 640, 3), 35, np.uint8)
                cv2.line(image, (0, 330), (639, 330), (130, 130, 130), 2)
                if kind == 'adl_two':
                    for x, color in ((150+int(12*np.sin(t)), (180, 210, 70)), (460, (70, 200, 230))):
                        cv2.circle(image, (x, 140), 15, color, -1)
                        cv2.line(image, (x, 156), (x, 260), color, 12)
                        for dx in (-24, 24): cv2.line(image, (x, 260), (x+dx, 326), color, 8)
                elif 'fall' in kind:
                    p = min(1., max(0., (t-1.2)/.6))
                    hip = (290, 260+int(55*p))
                    head = (290-int(100*p), 150+int(165*p))
                    color = (80, 220, 210) if kind == 'fall' else (210, 90, 160)
                    cv2.circle(image, head, 15, color, -1)
                    cv2.line(image, head, hip, color, 12)
                    for dx in (-24, 24): cv2.line(image, hip, (290+dx, 326), color, 8)
                writer.write(image)
        finally:
            writer.release()
        reader = OpenCVFrameSource(path)
        try:
            reader.open()
            timing = FrameTimingSpec(reader.fps, reader.frame_count, reader.frame_count/reader.fps, True)
        finally:
            reader.release()
        if 'fall' in kind:
            people = (PersonAnnotation('drawn-person', True,
                fall_onset_interval_s=TimeInterval(1.2, 1.24), down_posture_interval_s=TimeInterval(1.8, 8.),
                stillness_eligible_interval_s=TimeInterval(1.8, 4.8), initial_bbox_xyxy=(260., 130., 320., 330.)),)
        elif kind == 'adl_two':
            people = tuple(PersonAnnotation(label, False, initial_bbox_xyxy=box) for label, box in (
                ('drawn-left', (120., 120., 180., 330.)), ('drawn-right', (430., 120., 490., 330.))))
        else:
            people = ()
        rows.append(ClipManifestRow(name, path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest(),
            'synthetic-holdout-'+kind, name, 'holdout-drawing-v1-'+kind, 'synthetic-side-view',
            'excluded_out_of_envelope' if kind == 'excluded_fall' else 'supported',
            'Synthetic regression only; excluded geometry' if kind == 'excluded_fall' else 'Synthetic scoring fixture; not human accuracy evidence',
            len(people), timing, None, people))
    manifest = DatasetManifest('1.0', 'holdout', tuple(rows))
    save_dataset_manifest(manifest, root/'data/manifests/holdout.json')
    for row in rows: print(f'{row.clip_id}: {row.sha256}, {row.frame_timing.frame_count} frames')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    generate(parser.parse_args().root)
