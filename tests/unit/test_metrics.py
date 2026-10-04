"""Exact metric arithmetic and controlled predictions; no detector accuracy claim."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vertebrate.contracts import FramePacket, PersonTrackKey, SourceKind, TrackSample
from vertebrate.evaluation.labels import (ClipManifestRow, DatasetManifest, FrameTimingSpec,
    PersonAnnotation, TimeInterval, load_dataset_manifest)
from vertebrate.evaluation.metrics import (ClipMeasurements, Prediction, binomial_interval,
    check_partition_leakage, compute_metrics, evaluate_manifest, poisson_upper_95, release_gates)

ROOT = Path(__file__).resolve().parents[2]


def row(clip_id='fall', people=None, eligibility='supported', duration=8.):
    if people is None:
        people = (PersonAnnotation('alice', True, fall_onset_interval_s=TimeInterval(1., 1.2),
            down_posture_interval_s=TimeInterval(1.6, duration), stillness_eligible_interval_s=TimeInterval(1.6, 4.6),
            initial_bbox_xyxy=(0., 0., 100., 200.)),)
    return ClipManifestRow(clip_id, f'{clip_id}.avi', hashlib.sha256(clip_id.encode()).hexdigest(),
        'subject-'+clip_id, 'session-'+clip_id, 'recording-'+clip_id, 'side', eligibility,
        'excluded fixture' if eligibility != 'supported' else None, len(people),
        FrameTimingSpec(25., round(duration*25), duration, True), None, people)


def manifest(*rows): return DatasetManifest('1.0', 'holdout', rows or (row(),))
def prediction(onset=1.1, confirmed=4.6, person='alice', clip='fall', incident='id1'):
    return Prediction(clip, person, onset, confirmed, incident)


@pytest.mark.parametrize('onset,confirmation', [(.5, 4.35), (1.7, 6.6), (1.1, 4.6)])
def test_inclusive_tolerances(onset, confirmation):
    m = compute_metrics(manifest(), [prediction(onset, confirmation)])
    assert (m['TP'], m['FP'], m['FN']) == (1, 0, 0)


@pytest.mark.parametrize('onset,confirmation', [(.499999, 4.6), (1.700001, 4.6), (1.1, 4.349999), (1.1, 6.600001)])
def test_outside_tolerances_is_fp_and_missed_event(onset, confirmation):
    m = compute_metrics(manifest(), [prediction(onset, confirmation)])
    assert (m['TP'], m['FP'], m['FN']) == (0, 1, 1)


def test_abstention_is_false_negative_and_precision_is_null():
    m = compute_metrics(manifest(), [])
    assert m['recall'] == 0 and m['FN'] == 1
    assert m['precision'] is None and m['precision_ci_95'] is None
    assert m['recall_ci_95'] == pytest.approx([0., .975])


def test_empty_denominators_never_serialize_as_perfect():
    m = compute_metrics(DatasetManifest('1.0', 'holdout', ()), [])
    for k in ('recall', 'precision', 'pose_valid_coverage', 'inconclusive_fraction',
              'false_alerts_per_hour', 'false_alerts_per_hour_poisson_upper_95'):
        assert m[k] is None
    assert 'NaN' not in json.dumps(m, allow_nan=False)


def test_first_chronological_match_then_duplicate_fp_even_same_incident_id():
    m = compute_metrics(manifest(), [prediction(confirmed=5.), prediction(confirmed=4.5)])
    assert (m['TP'], m['FP'], m['FN'], m['duplicate_count']) == (1, 1, 0, 1)
    assert m['predictions'][0]['confirmed_s'] == 4.5
    assert m['precision'] == .5 and m['recall'] == 1


def test_equal_confirmation_times_preserve_emission_order():
    m = compute_metrics(manifest(), [prediction(onset=1.2, incident='first'), prediction(onset=1., incident='second')])
    assert m['predictions'][0]['incident_id'] == 'first'
    assert m['predictions'][0]['outcome'] == 'TP'


def test_generator_reproduces_actual_bytes_in_a_clean_local_root(tmp_path):
    import runpy
    generate = runpy.run_path(str(ROOT/'tools/generate_holdout_fixtures.py'))['generate']
    expected = load_dataset_manifest(ROOT/'data/manifests/holdout.json', True, ROOT)
    created = generate(tmp_path)
    assert [c.sha256 for c in created.clips] == [c.sha256 for c in expected.clips]
    assert load_dataset_manifest(tmp_path/'data/manifests/holdout.json', True, tmp_path) == created


@pytest.mark.parametrize('label', ['bob', '1', None])
def test_wrong_or_unassociated_person_never_matches(label):
    m = compute_metrics(manifest(), [prediction(person=label)])
    assert (m['TP'], m['FP'], m['FN']) == (0, 1, 1)


def test_single_prediction_cannot_match_two_people():
    alice = row().person_annotations[0]
    two = row(people=(alice, replace(alice, person_label='bob')))
    m = compute_metrics(manifest(two), [prediction()])
    assert m['TP'] == 1 and m['FN'] == 1
    assert m['missed_events'] == [{'clip_id': 'fall', 'person_label': 'bob'}]


def test_different_clip_cannot_match_same_label():
    m = compute_metrics(manifest(row(), row('second')), [prediction()])
    assert m['TP'] == 1 and m['FN'] == 1


def test_counts_ratios_intervals_coverage_and_inconclusive_exact():
    m = compute_metrics(manifest(row(), row('second')), [prediction(), prediction(person=None)],
        valid_core_observations=90, eligible_core_observations=100, inconclusive_episodes=1, candidate_episodes=4)
    assert (m['TP'], m['FP'], m['FN']) == (1, 1, 1)
    assert m['precision'] == m['recall'] == .5
    assert m['recall_ci_95'] == pytest.approx([.01257911709342505, .9874208829065749])
    assert m['pose_valid_coverage'] == .9 and m['inconclusive_fraction'] == .25
    assert m['confirmation_latency']['mean_s'] == pytest.approx(3.4)
    assert m['latency_onset_uncertainty_s'][0] == pytest.approx([3.4, 3.6])
    assert m['stillness_completion_delay']['mean_s'] == 0


@pytest.mark.parametrize('success,trials,expected', [
    (0, 10, [0., 1-.025**.1]), (10, 10, [.025**.1, 1.]),
    (27, 30, [.7347115495257919, .9788828629702774])])
def test_exact_clopper_pearson_reference_values(success, trials, expected):
    assert binomial_interval(success, trials) == pytest.approx(expected)


@pytest.mark.parametrize('args', [(-1, 2), (3, 2), (True, 2), (1, -1), (1., 2)])
def test_invalid_binomial_counts(args):
    with pytest.raises(ValueError): binomial_interval(*args)


def test_negative_video_hours_count_once_with_two_people_and_exact_poisson_bound():
    negative = row('negative', (PersonAnnotation('a', False), PersonAnnotation('b', False)), duration=3600)
    m = compute_metrics(manifest(negative), [])
    assert m['negative_hours'] == 1
    assert m['false_alerts_per_hour'] == 0
    assert m['false_alerts_per_hour_poisson_upper_95'] == pytest.approx(-math.log(.05))
    assert poisson_upper_95(1, 1) == pytest.approx(4.743864518390577)


def test_only_false_alerts_on_negative_clips_enter_false_alert_rate():
    negative = row('negative', (), duration=1800)
    m = compute_metrics(manifest(row(), negative), [prediction(person=None), prediction(clip='negative', person=None)])
    assert m['FP'] == 2 and m['negative_false_alerts'] == 1
    assert m['false_alerts_per_hour'] == 2


@pytest.mark.parametrize('eligibility', ['excluded_out_of_envelope', 'ambiguity_stress_test'])
def test_excluded_and_ambiguous_reported_separately(eligibility):
    m = compute_metrics(manifest(row(eligibility=eligibility)), [prediction()])
    assert (m['TP'], m['FP'], m['FN']) == (0, 0, 0)
    assert m['recall'] is None
    assert m[eligibility][0]['annotated_events'] == 1
    assert len(m[eligibility][0]['predictions']) == 1


@pytest.mark.parametrize('change', [{'onset_s': math.nan}, {'confirmed_s': math.inf},
    {'onset_s': -1}, {'onset_s': 5, 'confirmed_s': 4}, {'incident_id': ''}])
def test_prediction_finite_and_order_validation(change):
    with pytest.raises(ValueError): replace(prediction(), **change)


def test_unknown_clip_and_impossible_coverage_rejected():
    with pytest.raises(ValueError): compute_metrics(manifest(), [prediction(clip='unknown')])
    with pytest.raises(ValueError): compute_metrics(manifest(), [], valid_core_observations=2, eligible_core_observations=1)
    with pytest.raises(ValueError): compute_metrics(manifest(), [], inconclusive_episodes=1)


def sample(sequence=0, generation=1, key=99):
    return TrackSample(PersonTrackKey('session', key, generation), sequence, sequence/25,
        np.array([0., 0., 100., 200.]), .9, np.ones((17, 2)), np.ones(17), np.ones(17, bool), True)


def frame(sequence=0):
    return FramePacket('session', sequence, SourceKind.VIDEO, 'local', 100, 200,
        np.zeros((200, 100, 3), np.uint8), sequence/25, sequence+1)


def test_box_association_never_assumes_numeric_tracker_id():
    observer = ClipMeasurements(row(), .35)
    observer.observe(frame(), [sample()])
    assert observer.associations == {PersonTrackKey('session', 99, 1): 'alice'}
    observer.observe(frame(25), [sample(25)])
    observer.observe(frame(26), [sample(26, generation=2)])
    observer.observe(frame(27), [])
    assert observer.eligible == 3 and observer.valid == 1


def test_ambiguous_initial_boxes_abstain_and_do_not_match_later_frames():
    annotation = row().person_annotations[0]
    observer = ClipMeasurements(row(people=(annotation, replace(annotation, person_label='bob'))), .35)
    observer.observe(frame(), [sample()])
    assert not observer.associations
    observer.observe(frame(25), [sample(25)])
    assert observer.eligible == 2 and observer.valid == 0


@pytest.mark.parametrize('invalid', ['core', 'legs', 'confidence', 'predicted'])
def test_coverage_requires_real_common_core(invalid):
    observer = ClipMeasurements(row(), .35)
    observer.observe(frame(), [sample()])
    s = sample(25)
    if invalid == 'predicted': s = replace(s, predicted_only=True, matched_detection=False, observation_valid=False)
    elif invalid == 'confidence': s = replace(s, keypoint_confidences=np.zeros(17))
    else:
        mask = np.ones(17, bool)
        mask[[5] if invalid == 'core' else [13, 14, 15]] = False
        s = replace(s, keypoint_valid_mask=mask)
    observer.observe(frame(25), [s])
    assert observer.valid == 0 and observer.eligible == 1


@pytest.mark.parametrize('field', ['subject_id', 'session_group', 'source_group', 'sha256', 'relative_path'])
def test_partition_leakage_rejected(field):
    dev = DatasetManifest('1.0', 'dev', (row('dev'),))
    holdout = manifest(replace(row(), **{field: getattr(dev.clips[0], field)}))
    with pytest.raises(ValueError, match='leakage'): check_partition_leakage(dev, holdout)


def test_actual_holdout_files_have_valid_hashes_cfr_and_no_dev_overlap():
    holdout = load_dataset_manifest(ROOT/'data/manifests/holdout.json', True, ROOT)
    dev = load_dataset_manifest(ROOT/'data/manifests/dev.json', True, ROOT)
    check_partition_leakage(dev, holdout)
    assert len(holdout.clips) == 4
    assert all(c.frame_timing.cfr_verified and c.frame_timing.frame_count == 200 for c in holdout.clips)
    assert any(not c.person_annotations for c in holdout.clips)
    assert any(p.fall_expected for c in holdout.clips for p in c.person_annotations)


def test_gate_boundaries_and_missing_evidence_do_not_silently_pass():
    data = manifest(row(), row('two', (PersonAnnotation('a', False), PersonAnnotation('b', False))))
    m = compute_metrics(data, [prediction()], valid_core_observations=9, eligible_core_observations=10)
    runs = [{'clip_id': c.clip_id, 'frames': 100, 'elapsed_s': 10., 'processed_source_intervals_s': [.15]} for c in data.clips]
    gates = release_gates(m, runs, data)
    assert gates['recall']['status'] == gates['precision']['status'] == gates['pose_valid_coverage']['status'] == 'pass'
    assert gates['cpu_fps_1_people']['status'] == gates['cpu_fps_2_people']['status'] == 'pass'
    assert gates['source_continuity_p95_s']['status'] == 'pass'
    assert gates['negative_exposure']['status'] == gates['representative_holdout']['status'] == 'fail'


def test_cli_gate_failure_exit_contract_with_explicit_stub_report(monkeypatch, tmp_path):
    # CLI branching only. Actual inference is tested by the required evaluate run.
    from vertebrate.cli import main
    monkeypatch.setattr('vertebrate.evaluation.metrics.evaluate_manifest', lambda *a: {
        'status': 'completed', 'gates_passed': False, 'network_attempts': [], 'metrics': {'TP': 0, 'FP': 0, 'FN': 1}})
    assert main(['evaluate', '--output', str(tmp_path/'r.json')]) == 0
    assert main(['evaluate', '--fail-on-gates']) == 2


def test_missing_holdout_fails_before_inference(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('Inference must follow asset verification')
    monkeypatch.setattr('vertebrate.evaluation.runner.run_manifest', forbidden)
    with pytest.raises(ValueError): evaluate_manifest(tmp_path/'missing.json', root_dir=ROOT)
