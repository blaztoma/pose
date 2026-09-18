"""Holistic Tasks VIDEO backend and conversion to the existing .pose schema."""
import json
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from .base import PoseEstimator

THRESHOLDS = {
    'min_face_detection_confidence', 'min_face_suppression_threshold',
    'min_face_landmarks_confidence', 'min_pose_detection_confidence',
    'min_pose_suppression_threshold', 'min_pose_landmarks_confidence',
    'min_hand_landmarks_confidence',
}
COMPONENTS = (
    ('pose_landmarks', 33, True), ('face_landmarks', 478, False),
    ('left_hand_landmarks', 21, False), ('right_hand_landmarks', 21, False),
    ('pose_world_landmarks', 33, True),
)


def validate_runtime():
    """Reject legacy Tasks bindings before their native empty-packet abort."""
    try:
        installed = version('mediapipe')
    except PackageNotFoundError:
        installed = 'not installed'
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?', installed)
    release = tuple(map(int, match.groups())) if match else None
    if release is None or not (1, 0, 1) <= release < (2, 0, 0):
        raise ValueError(
            f'mediapipe-tasks requires MediaPipe >=1.0.1,<2; found {installed}. '
            f'Python: {sys.executable}. Activate the separate .venv-tasks environment '
            'or run its Scripts/videos_to_poses.exe directly (bin/videos_to_poses on Linux). '
            'Keep the legacy .venv unchanged. Older Holistic Tasks bindings can abort '
            'with "The packet is empty" when landmarks are absent.')


def holistic_header(width, height):
    # Exported from the existing holistic_components(additional_face_points=10).
    # Keeping topology as data avoids importing the removed mp.solutions package.
    schema = json.loads(Path(__file__).with_name('holistic_schema.json').read_text(encoding='utf-8'))
    return PoseHeader(version=0.2, dimensions=PoseHeaderDimensions(width, height, 0),
                      components=[PoseHeaderComponent(**item) for item in schema])


def result_arrays(result, width, height):
    data, confidence = [], []
    for name, count, use_visibility in COMPONENTS:
        landmarks = getattr(result, name, None) or []
        if not landmarks:
            data.append(np.zeros((count, 3), dtype=np.float32))
            confidence.append(np.zeros(count, dtype=np.float32))
            continue
        if len(landmarks) != count:
            raise ValueError(f'{name}: expected {count} landmarks, got {len(landmarks)}')
        # Preserve legacy world-coordinate storage too: animation divides world
        # x/y by width/height. Changing it here would silently distort the rig.
        xyz = np.array([[p.x * width, p.y * height, p.z] for p in landmarks], dtype=np.float32)
        scores = np.array([(p.visibility if p.visibility is not None else 0.0)
                           if use_visibility else 1.0 for p in landmarks], dtype=np.float32)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(scores)
        xyz[~valid] = 0
        scores[~valid] = 0
        data.append(xyz)
        confidence.append(np.clip(scores, 0, 1))
    return np.concatenate(data), np.concatenate(confidence)


def result_extras(result, index, timestamp_ms):
    def world(name):
        # Native metric coordinates; each hand has its own local origin.
        return [[float(p.x), float(p.y), float(p.z)] for p in (getattr(result, name, None) or [])]
    return {
        'frame_index': index, 'timestamp_ms': timestamp_ms,
        'face_blendshapes': {c.category_name: float(c.score) for c in (result.face_blendshapes or [])},
        'left_hand_world_landmarks': world('left_hand_world_landmarks'),
        'right_hand_world_landmarks': world('right_hand_world_landmarks'),
    }


def resolve_model(model):
    if model is not None:
        path = Path(model).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f'Tasks model not found: {path}')
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (base, *base.parents):
            path = parent / 'models' / 'mediapipe' / 'holistic_landmarker.task'
            if path.is_file():
                return path
    raise FileNotFoundError('Holistic Tasks model not found. Download the official holistic_landmarker.task and pass --model PATH.')


class TasksEstimator(PoseEstimator):
    def __init__(self, device, model, config, pose_workers):
        if device == 'gpu' and platform.system() == 'Windows':
            raise ValueError('MediaPipe Tasks Windows wheels have GPU processing disabled. Use --device cpu here; GPU requires a supported Linux build and graphics runtime.')
        if pose_workers != 1:
            raise ValueError('Tasks VIDEO tracking requires --workers 1. Parallelize separate videos with videos_to_poses --num-workers.')
        unknown = config.keys() - THRESHOLDS - {'output_face_blendshapes'}
        if unknown:
            raise ValueError(f'Unsupported Tasks options: {sorted(unknown)}. Legacy model_complexity/refine_face_landmarks/smooth_landmarks do not apply to the Tasks model bundle.')
        for name in config.keys() & THRESHOLDS:
            if not isinstance(config[name], (int, float)) or not 0 <= config[name] <= 1:
                raise ValueError(f'{name} must be a number between 0 and 1')
        if 'output_face_blendshapes' in config and not isinstance(config['output_face_blendshapes'], bool):
            raise ValueError('output_face_blendshapes must be true or false')
        validate_runtime()
        self.device = device
        self.model = resolve_model(model)
        self.config = {'output_face_blendshapes': True, **config}

    def estimate(self, frames, *, fps, width, height, progress=True, extras=None, on_frame=None):
        if not np.isfinite(fps) or not 0 < fps <= 1000 or width <= 0 or height <= 0:
            raise ValueError('Tasks requires positive image dimensions and FPS in (0, 1000]')
        import mediapipe as mp
        vision = mp.tasks.vision
        delegate = mp.tasks.BaseOptions.Delegate.GPU if self.device == 'gpu' else mp.tasks.BaseOptions.Delegate.CPU
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO, **self.config)
        try:
            detector = vision.HolisticLandmarker.create_from_options(options)
        except (RuntimeError, ValueError, NotImplementedError) as error:
            raise RuntimeError(f'Cannot initialize Tasks {self.device} delegate: {error}. No CPU fallback was attempted.') from error
        data, confidence = [], []
        with detector:
            for index, frame in enumerate(tqdm(frames, disable=not progress, desc='Holistic Tasks')):
                timestamp = round(index * 1000 / fps)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                result = detector.detect_for_video(image, timestamp)
                xyz, scores = result_arrays(result, width, height)
                data.append(xyz)
                confidence.append(scores)
                if extras is not None:
                    extras(result_extras(result, index, timestamp))
                if on_frame is not None:
                    on_frame()
        if not data:
            raise ValueError('Video contains no decodable frames')
        return Pose(holistic_header(width, height), NumPyPoseBody(
            fps=fps, data=np.array(data)[:, None], confidence=np.array(confidence)[:, None]))
