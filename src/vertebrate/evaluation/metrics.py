"""Locked source-time event scoring. Synthetic fixtures never prove release accuracy.

Intervals are two-sided exact Clopper-Pearson (95%). False-alert exposure counts
whole supported negative clips once, not person-hours. Its Poisson bound is
one-sided 95%. Latency uses the annotated onset interval's end as reference and
also reports the full onset uncertainty range and stillness-completion delay.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

from scipy.stats import beta, chi2
import numpy as np

from ..contracts import FramePacket
from ..validation import integer, number, text
from .labels import DatasetManifest, load_dataset_manifest


@dataclass(frozen=True)
class MatchingPolicy:
    onset_tolerance_s: float = .5
    confirmation_early_s: float = .25
    confirmation_late_s: float = 2.
    initial_box_iou: float = .5


POLICY = MatchingPolicy()


@dataclass(frozen=True)
class Prediction:
    clip_id: str
    person_label: str | None
    onset_s: float
    confirmed_s: float
    incident_id: str

    def __post_init__(self):
        text(self.clip_id, 'clip_id')
        text(self.incident_id, 'incident_id')
        if self.person_label is not None: text(self.person_label, 'person_label')
        number(self.onset_s, 'onset_s')
        number(self.confirmed_s, 'confirmed_s')
        if self.confirmed_s < self.onset_s:
            raise ValueError('Confirmation cannot precede onset')


def binomial_interval(successes, trials):
    integer(trials, 'trials')
    integer(successes, 'successes')
    if successes > trials: raise ValueError('successes exceeds trials')
    if trials == 0: return None
    return [0. if successes == 0 else float(beta.ppf(.025, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(.975, successes+1, trials-successes))]


def poisson_upper_95(count, hours):
    integer(count, 'count')
    number(hours, 'hours')
    if hours == 0: return None
    return float(chi2.ppf(.95, 2*(count+1)) / (2*hours))


def _ratio(a, b): return a/b if b else None


def _summary(values):
    return {'values_s': values, 'mean_s': float(np.mean(values)) if values else None,
            'median_s': float(np.median(values)) if values else None,
            'p95_s': float(np.percentile(values, 95)) if values else None}


def compute_metrics(manifest: DatasetManifest, predictions, *, valid_core_observations=0,
                    eligible_core_observations=0, inconclusive_episodes=0, candidate_episodes=0):
    """Chronological first qualifying prediction wins; all later duplicates are FP.

person_label is an explicit annotation association, never a numeric tracker ID.
Unassociated predictions remain FP; absent/abstained eligible events remain FN.
"""
    if not isinstance(manifest, DatasetManifest): raise ValueError('Expected DatasetManifest')
    for name, value in [('valid_core_observations', valid_core_observations),
                        ('eligible_core_observations', eligible_core_observations),
                        ('inconclusive_episodes', inconclusive_episodes), ('candidate_episodes', candidate_episodes)]:
        integer(value, name)
    if valid_core_observations > eligible_core_observations or inconclusive_episodes > candidate_episodes:
        raise ValueError('Numerator exceeds denominator')
    rows = {c.clip_id: c for c in manifest.clips}
    predictions = tuple(predictions)
    if any(not isinstance(p, Prediction) or p.clip_id not in rows for p in predictions):
        raise ValueError('Prediction must reference a known clip')
    events = {(c.clip_id, p.person_label): p for c in manifest.clips if c.eligibility == 'supported'
              for p in c.person_annotations if p.fall_expected}
    negatives = {c.clip_id for c in manifest.clips if c.eligibility == 'supported'
                 and not any(p.fall_expected for p in c.person_annotations)}
    matched, outcomes, latency, delay, uncertainty = set(), [], [], [], []
    duplicate_count = negative_fp = fp = 0
    separate = {kind: [] for kind in ('excluded_out_of_envelope', 'ambiguity_stress_test')}
    for c in manifest.clips:
        if c.eligibility in separate:
            separate[c.eligibility].append({'clip_id': c.clip_id, 'reason': c.exclusion_reason,
                'annotated_events': sum(p.fall_expected for p in c.person_annotations),
                'predictions': [asdict(p) for p in predictions if p.clip_id == c.clip_id]})
    # Python's stable sort preserves emission order on equal confirmation times.
    for prediction in sorted(predictions, key=lambda p: p.confirmed_s):
        if rows[prediction.clip_id].eligibility != 'supported': continue
        key = (prediction.clip_id, prediction.person_label)
        event = events.get(key)
        compatible = event is not None and (
            event.fall_onset_interval_s.start_s-POLICY.onset_tolerance_s <= prediction.onset_s <= event.fall_onset_interval_s.end_s+POLICY.onset_tolerance_s
            and event.stillness_eligible_interval_s.end_s-POLICY.confirmation_early_s <= prediction.confirmed_s <= event.stillness_eligible_interval_s.end_s+POLICY.confirmation_late_s)
        if compatible and key not in matched:
            matched.add(key)
            outcome = 'TP'
            latency.append(prediction.confirmed_s-event.fall_onset_interval_s.end_s)
            uncertainty.append([prediction.confirmed_s-event.fall_onset_interval_s.end_s,
                                prediction.confirmed_s-event.fall_onset_interval_s.start_s])
            delay.append(prediction.confirmed_s-event.stillness_eligible_interval_s.end_s)
        else:
            outcome = 'duplicate_FP' if compatible and key in matched else 'FP'
            duplicate_count += outcome == 'duplicate_FP'
            fp += 1
            negative_fp += prediction.clip_id in negatives
        outcomes.append({**asdict(prediction), 'outcome': outcome})
    tp, fn = len(matched), len(events)-len(matched)
    hours = sum(rows[key].frame_timing.duration_s for key in negatives)/3600
    return {'TP': tp, 'FP': fp, 'FN': fn, 'recall': _ratio(tp, tp+fn), 'precision': _ratio(tp, tp+fp),
        'recall_ci_95': binomial_interval(tp, tp+fn), 'precision_ci_95': binomial_interval(tp, tp+fp),
        'binomial_interval_method': 'clopper_pearson_exact_two_sided',
        'negative_hours': hours, 'negative_false_alerts': negative_fp,
        'false_alerts_per_hour': _ratio(negative_fp, hours),
        'false_alerts_per_hour_poisson_upper_95': poisson_upper_95(negative_fp, hours),
        'duplicate_count': duplicate_count, 'confirmation_latency': _summary(latency),
        'latency_reference': 'annotated_onset_interval_end', 'latency_onset_uncertainty_s': uncertainty,
        'stillness_completion_delay': _summary(delay),
        'valid_core_observations': valid_core_observations, 'eligible_core_observations': eligible_core_observations,
        'pose_valid_coverage': _ratio(valid_core_observations, eligible_core_observations),
        'inconclusive_episodes': inconclusive_episodes, 'candidate_episodes': candidate_episodes,
        'inconclusive_fraction': _ratio(inconclusive_episodes, candidate_episodes),
        'predictions': outcomes, 'missed_events': [{'clip_id': c, 'person_label': p} for c, p in sorted(events.keys()-matched)],
        **separate}


def _iou(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.


class ClipMeasurements:
    """Conservative first-frame box association; no guessing after ID expiry.

Initial boxes annotate frame zero only. Ambiguous/no matches remain unassociated.
Coverage denominator counts every person-frame from onset through eligible
completion, including visibility gaps. Predictions do not define eligibility.
"""
    def __init__(self, row, confidence):
        self.row, self.confidence = row, confidence
        self.associations = {}
        self.valid = self.eligible = 0
        self.intervals = []
        self.last_t = None

    def observe(self, frame, samples):
        if not isinstance(frame, FramePacket): return
        if frame.sequence == 0:
            candidates = [(sample.person_key, person.person_label) for sample in samples
                if sample.observation_valid and sample.matched_detection and not sample.predicted_only
                for person in self.row.person_annotations if person.initial_bbox_xyxy is not None
                and _iou(sample.bbox_xyxy, person.initial_bbox_xyxy) >= POLICY.initial_box_iou]
            for key, label in candidates:
                if sum(k == key for k, _ in candidates) == 1 and sum(p == label for _, p in candidates) == 1:
                    self.associations[key] = label
        if self.last_t is not None: self.intervals.append(frame.source_t_s-self.last_t)
        self.last_t = frame.source_t_s
        if self.row.eligibility != 'supported': return
        by_label = {self.associations[s.person_key]: s for s in samples if s.person_key in self.associations}
        for person in self.row.person_annotations:
            if not person.fall_expected: continue
            if person.fall_onset_interval_s.start_s <= frame.source_t_s <= person.stillness_eligible_interval_s.end_s:
                self.eligible += 1
                s = by_label.get(person.person_label)
                if s is None or not s.observation_valid or not s.matched_detection or s.predicted_only: continue
                mask = s.keypoint_valid_mask & (s.keypoint_confidences >= self.confidence)
                self.valid += bool(all(mask[i] for i in (5, 6, 11, 12)) and sum(mask[i] for i in (13, 14, 15, 16)) >= 2)


def check_partition_leakage(dev, holdout):
    """Reject shared source groups, people, sessions, clip paths or exact bytes."""
    if dev.partition != 'dev' or holdout.partition != 'holdout': raise ValueError('Expected dev and holdout partitions')
    overlap = {}
    for field in ('subject_id', 'session_group', 'source_group', 'sha256', 'relative_path'):
        common = {getattr(c, field) for c in dev.clips} & {getattr(c, field) for c in holdout.clips}
        if common: overlap[field] = sorted(common)
    if overlap: raise ValueError(f'Development/holdout leakage: {overlap}')


def release_gates(metrics, runs, manifest):
    gates = {}
    def gate(name, value, threshold, passed):
        gates[name] = {'value': value, 'required': threshold, 'status': 'pass' if passed else 'fail'}
    for name in ('recall', 'precision', 'pose_valid_coverage'):
        value = metrics[name]
        gate(name, value, '>=0.90', value is not None and value >= .9)
    gate('negative_exposure', metrics['negative_hours'], '>=1 hour and zero false alerts',
         metrics['negative_hours'] >= 1 and metrics['negative_false_alerts'] == 0)
    gate('duplicates', metrics['duplicate_count'], '0', metrics['duplicate_count'] == 0)
    for n in (1, 2):
        group = [r for r in runs if next(c for c in manifest.clips if c.clip_id == r['clip_id']).people_count == n]
        elapsed = sum(r['elapsed_s'] for r in group)
        fps = sum(r['frames'] for r in group)/elapsed if elapsed > 0 else None
        gate(f'cpu_fps_{n}_people', fps, '>=10 end-to-end frames/s', fps is not None and fps >= 10)
    intervals = [t for r in runs for t in r['processed_source_intervals_s']]
    p95 = float(np.percentile(intervals, 95)) if intervals else None
    gate('source_continuity_p95_s', p95, '<=0.150', p95 is not None and p95 <= .15)
    synthetic = any('synthetic' in (c.subject_id+' '+c.camera_view+' '+str(c.exclusion_reason)).lower() for c in manifest.clips)
    falls = sum(p.fall_expected for c in manifest.clips if c.eligibility == 'supported' for p in c.person_annotations)
    two_person_falls = sum(p.fall_expected for c in manifest.clips if c.eligibility == 'supported' and c.people_count == 2
                           for p in c.person_annotations)
    gate('representative_holdout', {'supported_falls': falls, 'two_person_falls': two_person_falls,
                                   'synthetic': synthetic}, '>=30 supported falls, >=6 with two people; non-synthetic',
         not synthetic and falls >= 30 and two_person_falls >= 6)
    gates['representative_holdout']['reason'] = ('Synthetic regression fixtures are not release evidence' if synthetic else
        'Provenance requires independent review; direction and room quotas are reported as unmeasured')
    return gates


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', encoding='utf-8', delete=False) as stream:
            name = stream.name
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists(): Path(name).unlink()


def _code_sha256(root):
    digest = hashlib.sha256()
    for path in sorted((root/'src/vertebrate').rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def evaluate_manifest(manifest_path='data/manifests/holdout.json', config_path='config/demo.json',
                      output='reports/evaluation.json', *, root_dir=None):
    from ..config import config_sha256, config_to_dict, load_config
    from ..cli import verify_model_asset
    from .runner import run_manifest
    root = Path(root_dir or Path.cwd()).resolve()
    manifest_file, config_file, target = (root / p for p in (manifest_path, config_path, output))
    manifest = load_dataset_manifest(manifest_file, verify_files=True, root_dir=root)
    if manifest.partition != 'holdout' or not manifest.clips: raise ValueError('Evaluation requires a nonempty holdout manifest')
    dev = load_dataset_manifest(root / 'data/manifests/dev.json', verify_files=True, root_dir=root)
    check_partition_leakage(dev, manifest)
    config = load_config(config_file)
    asset = verify_model_asset(root / config.runtime.manifest_path, config.model.model_name, root)
    protected = [manifest_file, config_file, asset.path, root / 'data/manifests/dev.json'] + [root/c.relative_path for c in (*manifest.clips, *dev.clips)]
    if target.suffix != '.json' or target.resolve() in {p.resolve() for p in protected}:
        raise ValueError('Output must be a distinct JSON report, not an input asset')
    lock = {'config_sha256': config_sha256(config), 'config': config_to_dict(config),
            'manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
            'development_manifest_sha256': hashlib.sha256((root/'data/manifests/dev.json').read_bytes()).hexdigest(),
            'model_sha256': asset.sha256, 'code_sha256': _code_sha256(root),
            'matching_policy': asdict(POLICY), 'calibration': 'Starting thresholds unchanged; no human-data tuning claimed'}
    lock_path = target.with_name(target.stem+'.lock.json')
    if lock_path.resolve() in {p.resolve() for p in protected}: raise ValueError('Lock path collides with input')
    _atomic_json(lock_path, lock)  # freeze before inference, not after inspecting errors
    replay = run_manifest(manifest_file, config_file, root_dir=root)
    if (replay['config_sha256'] != lock['config_sha256'] or replay['model_sha256'] != lock['model_sha256']
            or hashlib.sha256(manifest_file.read_bytes()).hexdigest() != lock['manifest_sha256']
            or _code_sha256(root) != lock['code_sha256']):
        raise ValueError('Frozen inputs changed during evaluation; discard run')
    predictions = [Prediction(**p) for r in replay['clips'] for p in r['associated_predictions']]
    supported = [r for r in replay['clips'] if next(c for c in manifest.clips if c.clip_id == r['clip_id']).eligibility == 'supported']
    metrics = compute_metrics(manifest, predictions,
        valid_core_observations=sum(r['valid_core_observations'] for r in supported),
        eligible_core_observations=sum(r['eligible_core_observations'] for r in supported),
        inconclusive_episodes=sum(r['inconclusive_episodes'] for r in supported),
        candidate_episodes=sum(r['candidate_episodes'] for r in supported))
    gates = release_gates(metrics, replay['clips'], manifest)
    report = {'schema_version': '1.0', 'status': 'completed', 'partition': 'holdout', 'lock': lock,
        'metrics': metrics, 'gates': gates, 'gates_passed': all(g['status'] == 'pass' for g in gates.values()),
        'clips': replay['clips'], 'network_attempts': replay['network_attempts'],
        'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': config.model.device},
        'identity_policy': 'Unique frame-zero annotated-box IoU >=0.5; no numeric-ID assumptions or reassociation after generation expiry',
        'coverage_policy': 'Valid required core / all annotated onset-to-stillness-completion person-frames, including gaps',
        'performance_policy': 'Actual elapsed capture preflight, capture, inference, tracking, features, FSM and shutdown; model load/warmup excluded',
        'unmeasured_release_gates': ['direction_quotas', '30_minutes_in_demo_room', 'scripted_rehearsals', 'identity_switch_review', 'live_freshness', 'confirmation_delivery',
                                    'five_minute_gui_responsiveness', 'hardware_lifecycle_timing', '30_minute_soak'],
        'headline_release_evidence': False,
        'note': 'Regression evaluation; no headline accuracy claim. Synthetic data and repeated runs are not unseen human holdout or calibrated accuracy.'}
    _atomic_json(target, report)
    return report
