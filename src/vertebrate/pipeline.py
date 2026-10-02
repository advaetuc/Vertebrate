"""Headless sequential vision processing with an owned bounded capture worker.

PoseAdapter and ByteTrackManager are the existing pose-estimator/session-tracker
implementations. process_observation supports deterministic recorded-pose replay;
it is a component path, not a claim of model accuracy on synthetic trajectories.
"""
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Protocol

from .capture import CaptureWorker
from .config import AppConfig
from .contracts import (CaptureError, EndOfStreamEvent, FramePacket, PersonTrackKey,
                        PoseObservation, SessionResetEvent, TrackSample)
from .features import FeatureExtractor
from .fsm import CandidateAlert, Decision, PersonFSM


class PoseEstimator(Protocol):
    def infer(self, frame: FramePacket) -> PoseObservation: ...


class SessionTracker(Protocol):
    def update(self, observation: PoseObservation) -> tuple[TrackSample, ...]: ...
    def handle_event(self, event: SessionResetEvent | EndOfStreamEvent) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    frames: int
    dropped_frames: int
    alerts: tuple[CandidateAlert, ...]
    transitions: tuple[Decision, ...]
    network_attempts: tuple[str, ...]


class HeadlessPipeline:
    def __init__(self, config: AppConfig | None = None, root_dir: Path | None = None,
                 *, pose: PoseEstimator | None = None, tracker: SessionTracker | None = None,
                 frame_rate: float = 30):
        self.config = config or AppConfig()
        self.root = Path(root_dir or Path.cwd()).resolve()
        self.pose = pose  # lazy: recorded-pose replay does not require loading YOLO
        if tracker is None:
            from .tracking import ByteTrackManager
            tracker = ByteTrackManager(self.config.tracker, self.root, frame_rate)
        self.tracker = tracker
        self.session_id = None
        self.closed = False
        self._retired_sessions = set()
        self._people = {}
        self._last_match = {}
        self._retired_keys = set()
        self._last_sequence = self._last_t = None
        self.frames = 0
        self.alerts = []
        self.transitions = []

    def _record(self, fsm, decision, previous):
        if decision.alert is not None:
            self.alerts.append(decision.alert)
        if previous != decision.state or decision.inconclusive or decision.alert or fsm.closed:
            self.transitions.append(decision)

    def _close_people(self, reason):
        decisions = []
        for _, fsm in self._people.values():
            previous = fsm.state
            decision = fsm.close(reason)
            self._record(fsm, decision, previous)
            decisions.append(decision)
        self._people.clear()
        self._last_match.clear()
        return tuple(decisions)

    def handle_event(self, event):
        if isinstance(event, SessionResetEvent):
            if event.new_session_id == self.session_id or event.new_session_id in self._retired_sessions:
                raise ValueError('Session reset requires a fresh session ID')
            decisions = self._close_people('source_reset')
            if self.session_id is not None:
                self._retired_sessions.add(self.session_id)
            self.tracker.handle_event(event)
            self.session_id, self.closed = event.new_session_id, False
            self._last_sequence = self._last_t = None
            self._retired_keys.clear()
            return decisions
        if isinstance(event, EndOfStreamEvent):
            if self.closed or event.session_id != self.session_id:
                raise ValueError('EOF does not belong to an open session')
            self.tracker.handle_event(event)
            self.closed = True
            return self._close_people('end_of_stream')
        raise TypeError('Expected source reset or EOF')

    def source_error(self):
        self.closed = True
        return self._close_people('source_error')

    def _check_frame(self, session, sequence, source_t):
        if self.closed or session != self.session_id:
            raise ValueError('Frame does not belong to the open source session')
        if self._last_sequence is not None and (sequence <= self._last_sequence or source_t <= self._last_t):
            self.source_error()
            raise ValueError('Pipeline requires increasing source sequence and time')

    def process(self, item):
        if not isinstance(item, FramePacket):
            return self.handle_event(item)
        self._check_frame(item.session_id, item.sequence, item.source_t_s)
        if self.pose is None:
            from .pose import PoseAdapter
            self.pose = PoseAdapter(self.config, self.root, headless=True)
        try:
            observation = self.pose.infer(item)
            if (observation.session_id, observation.sequence, observation.source_t_s, observation.width, observation.height) != (
                    item.session_id, item.sequence, item.source_t_s, item.width, item.height):
                raise ValueError('Pose result is not aligned with its source frame')
            return self.process_observation(observation)
        except Exception:
            self.source_error()
            raise

    def process_observation(self, observation: PoseObservation):
        self._check_frame(observation.session_id, observation.sequence, observation.source_t_s)
        try:
            return self.process_tracks(self.tracker.update(observation), sequence=observation.sequence,
                                       source_t_s=observation.source_t_s, frame_height=observation.height)
        except Exception:
            self.source_error()
            raise

    def process_tracks(self, samples, *, sequence, source_t_s, frame_height=720):
        """Process already-associated immutable tracks; absent people get no credit."""
        self._check_frame(self.session_id, sequence, source_t_s)
        samples = tuple(samples)
        keys = [s.person_key for s in samples]
        if len(set(keys)) != len(keys) or len({k.tracker_id for k in keys}) != len(keys):
            raise ValueError('Duplicate person/numeric tracker ID in one frame')
        for sample in samples:
            if (sample.sequence != sequence or sample.source_t_s != source_t_s
                    or sample.person_key.session_id != self.session_id or sample.person_key in self._retired_keys):
                raise ValueError('Stale/misaligned track observation')
        self._last_sequence, self._last_t = sequence, source_t_s
        self.frames += 1
        decisions = []
        for key in list(self._people):
            if key in keys:
                continue
            extractor, fsm = self._people[key]
            previous = fsm.state
            replaced = any(k.tracker_id == key.tracker_id for k in keys)
            expired = source_t_s - self._last_match[key] > self.config.tracker.application_identity_expiry_s
            if replaced or expired:
                decision = fsm.close('identity_expired')
                self._retired_keys.add(key)
                del self._people[key]
                del self._last_match[key]
            else:
                extractor.invalidate(preserve_baseline=fsm.freeze_baseline)
                decision = fsm.missing(sequence, source_t_s)
                if decision.reset_baseline:
                    extractor.reset()
            self._record(fsm, decision, previous)
            decisions.append(decision)
        for sample in samples:
            key = sample.person_key
            if key not in self._people:
                self._people[key] = (FeatureExtractor(self.config.calibration, frame_height),
                                     PersonFSM(key, self.config.calibration))
            extractor, fsm = self._people[key]
            previous = fsm.state
            features = extractor.update(sample, candidate_active=fsm.freeze_baseline)
            decision = fsm.update(features, recovery_valid=all(
                sample.keypoint_valid_mask[i] and sample.keypoint_confidences[i] >= self.config.calibration.keypoint_confidence
                for i in (15, 16)))
            if decision.reset_baseline:
                extractor.reset()
            if sample.observation_valid and sample.matched_detection and not sample.predicted_only:
                self._last_match[key] = source_t_s
            else:
                self._last_match.setdefault(key, source_t_s)
            self._record(fsm, decision, previous)
            decisions.append(decision)
        return tuple(decisions)

    def run(self, source, *, capture_factory=CaptureWorker, on_decisions=None, **capture_options):
        """Consume the bounded capture queue; callbacks may delay processing only."""
        from .pose import PoseAdapter, guarded_runtime
        attempts = []
        worker = None
        with guarded_runtime(self.root, attempts):
            try:
                if self.pose is None:
                    self.pose = PoseAdapter(self.config, self.root, headless=True)
                worker = capture_factory(source, **capture_options)
                worker.start()
                while True:
                    if worker.error is not None:
                        raise worker.error
                    try:
                        item = worker.queue.get(timeout_s=.1)
                    except Empty:
                        if worker.state in ('error', 'stopped'):
                            raise CaptureError('Capture stopped without EOF')
                        continue
                    decisions = self.process(item)
                    if on_decisions is not None:
                        on_decisions(item, decisions)
                    if isinstance(item, EndOfStreamEvent):
                        break
            except Exception:
                self.source_error()
                raise
            finally:
                if worker is not None:
                    worker.stop()
        attempts.extend(getattr(self.pose, 'network_attempts', ()))
        attempts.extend(getattr(self.tracker, 'network_attempts', ()))
        if attempts:
            raise RuntimeError(f'Unexpected offline attempts: {attempts}')
        return PipelineResult(self.frames, worker.queue.dropped_frames_count,
                              tuple(self.alerts), tuple(self.transitions), tuple(attempts))
