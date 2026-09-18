"""Shared backend selection; importing this module never imports MediaPipe."""
from abc import ABC, abstractmethod
from pathlib import Path

BACKENDS = ('mediapipe-legacy', 'mediapipe-tasks')


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frames, *, fps, width, height, progress=True, extras=None, on_frame=None):
        """Return a Pose; call on_frame() after each completed frame."""


def create_estimator(backend='mediapipe-legacy', device='cpu', model=None,
                     additional_config=None, pose_workers=1):
    if device not in ('cpu', 'gpu'):
        raise ValueError(f'Unknown device: {device}')
    config = dict(additional_config or {})
    if backend == 'mediapipe-legacy':
        from .mediapipe_legacy import LegacyEstimator
        return LegacyEstimator(device, model, config, pose_workers)
    if backend == 'mediapipe-tasks':
        from .mediapipe_tasks import TasksEstimator
        return TasksEstimator(device, model, config, pose_workers)
    raise ValueError(f'Unknown backend: {backend}')


def add_estimator_arguments(parser):
    parser.add_argument('--backend', choices=BACKENDS, default=BACKENDS[0],
                        help='Estimator implementation (default: unchanged legacy Holistic)')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        help='Inference delegate; unavailable GPU causes an error, never a CPU retry')
    parser.add_argument('--model', type=Path,
                        help='Tasks Holistic .task model; otherwise search ancestors for models/mediapipe/holistic_landmarker.task')
