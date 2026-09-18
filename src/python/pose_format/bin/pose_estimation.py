#!/usr/bin/env python
import argparse
import os
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm

from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact
from pose_format.estimation.base import add_estimator_arguments, create_estimator


def pose_video(input_path: str, output_path: str, format: str, additional_config=None,
               progress: bool = True, pose_workers: int = 1, *,
               backend='mediapipe-legacy', device='cpu', model=None,
               progress_callback=None, progress_position=0, progress_label=None, verbose=True):
    """Estimate a video; progress_callback(completed, total) can replace local tqdm."""
    if format != 'mediapipe':
        raise NotImplementedError('Pose format not supported')
    if additional_config is None:
        additional_config = {'model_complexity': 1} if backend == 'mediapipe-legacy' else {}
    estimator = create_estimator(backend, device, model, additional_config, pose_workers)
    # Load video metadata
    if verbose:
        tqdm.write('Loading video ...')
    metadata = video_metadata(input_path)
    width = metadata.width
    height = metadata.height
    fps = metadata.fps
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        tqdm.write(f'Estimating pose: {backend}, device={device} ...')
    total = getattr(metadata, 'nb_frames', None)
    total = int(total) if total is not None and total > 0 else None
    # Only publish a completed .pose; interrupted runs remain retryable.
    with TemporaryDirectory(prefix='.pose-', dir=output.parent) as temporary, tqdm(
            total=total, desc=progress_label or Path(input_path).name, unit='frame',
            position=progress_position, dynamic_ncols=True,
            disable=not progress or progress_callback is not None) as frame_bar:
        completed = 0
        if progress and progress_callback is not None:
            progress_callback(0, total)

        def on_frame():
            nonlocal completed
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
            else:
                frame_bar.update(1)

        staging = Path(temporary)
        extra_path = staging / (output.name + '.extras.jsonl')
        frames = read_frames_exact(input_path)
        try:
            if backend == 'mediapipe-tasks':
                with extra_path.open('w', encoding='utf-8') as extra_file:
                    def write_extras(record):
                        extra_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    pose = estimator.estimate(frames, fps=fps, width=width, height=height,
                                              progress=False, extras=write_extras,
                                              on_frame=on_frame if progress else None)
            else:
                pose = estimator.estimate(frames, fps=fps, width=width, height=height, progress=False,
                                          on_frame=on_frame if progress else None)
        finally:
            if hasattr(frames, 'close'):
                frames.close()
        pose_path = staging / output.name
        with pose_path.open('wb') as stream:
            pose.write(stream)
        report = {'backend': backend, 'device': device, 'mediapipe_version': version('mediapipe'),
                  'source': str(Path(input_path).resolve()), 'fps': float(fps),
                  'frames': len(pose.body.data), 'width': width, 'height': height,
                  'config': estimator.config, 'pose_workers': pose_workers}
        if backend == 'mediapipe-tasks':
            report.update(model=str(estimator.model),
                          model_sha256=hashlib.sha256(estimator.model.read_bytes()).hexdigest(),
                          extras_file=extra_path.name,
                          extras_schema_version=1,
                          timestamps='round(frame_index * 1000 / fps); CFR timeline',
                          pose_coordinates='Legacy schema: all x * width, y * height; z unchanged, including POSE_WORLD_LANDMARKS',
                          extras_hand_coordinates='Meters, separate local origin for each hand')
        meta_path = staging / (output.name + '.meta.json')
        meta_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        if verbose:
            tqdm.write('Saving to disk ...')
        if extra_path.exists():
            extra_path.replace(output.with_name(extra_path.name))
        meta_path.replace(output.with_name(meta_path.name))
        pose_path.replace(output)
    return pose


def parse_additional_config(config: str):
    if not config:
        return {}
    config = config.split(',')

    def parse_value(value):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return value

    return {k: parse_value(v) for k, v in [c.split('=') for c in config]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input video file')
    parser.add_argument('-o', required=True, type=str, help='path to output pose file')
    parser.add_argument('--format',
                        choices=['mediapipe'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument('--additional-config', type=str, help='additional configuration for the pose estimator')
    parser.add_argument('--workers', type=int, default=1, help='number of parallel holistic instances (0 = all CPUs)')
    parser.add_argument('--no-progress', action='store_true', help='Disable frame progress bars')
    add_estimator_arguments(parser)

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Video file {args.i} not found")

    additional_config = parse_additional_config(args.additional_config)
    try:
        pose_video(args.i, args.o, args.format, additional_config, pose_workers=args.workers,
                   backend=args.backend, device=args.device, model=args.model, progress=not args.no_progress)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')

    # pip install . && video_to_pose -i como.mp4 -o como1.pose --format mediapipe
    # pip install . && video_to_pose -i como.mp4 -o como2.pose --format mediapipe --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
    # pip install . && video_to_pose -i sparen.mp4 -o sparen.pose --format mediapipe --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"


if __name__ == '__main__':
    main()
