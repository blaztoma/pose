"""Compatibility adapter for the existing mp.solutions.holistic CPU pipeline."""
from .base import PoseEstimator


class LegacyEstimator(PoseEstimator):
    def __init__(self, device, model, config, pose_workers):
        if device != 'cpu':
            raise ValueError('mediapipe-legacy only supports --device cpu. Use a GPU-capable Tasks installation for GPU.')
        if model is not None:
            raise ValueError('--model is only supported by mediapipe-tasks')
        if pose_workers < 0:
            raise ValueError('--workers must be nonnegative')
        self.config = config
        self.pose_workers = pose_workers

    def estimate(self, frames, *, fps, width, height, progress=True, extras=None, on_frame=None):
        # New Tasks releases no longer ship mp.solutions. Keep this import lazy.
        from pose_format.utils.holistic import load_holistic
        return load_holistic(frames, fps=fps, width=width, height=height,
                             progress=progress, additional_holistic_config=self.config,
                             pose_workers=self.pose_workers, on_frame=on_frame)
