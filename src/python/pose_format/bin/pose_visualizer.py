#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info

from pose_format.utils.generic import normalize_pose_size


VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}


def find_background_video(pose_path: Path) -> Path:
    # Also support files produced with videos_to_poses --keep-video-suffixes.
    original = pose_path.with_suffix('')
    if original.suffix.lower() in VIDEO_SUFFIXES and original.is_file():
        return original
    matches = sorted(path for path in pose_path.parent.iterdir()
                     if path.is_file() and path.stem == pose_path.stem
                     and path.suffix.lower() in VIDEO_SUFFIXES)
    if not matches:
        # Estimation outputs may live in a separate --output-directory. The
        # sidecar records the original path; filtered poses use the same source.
        unfiltered = (pose_path.with_name(pose_path.stem[:-9] + '.pose')
                      if pose_path.stem.endswith('_filtered') else pose_path)
        metadata = unfiltered.with_name(unfiltered.name + '.meta.json')
        if metadata.is_file():
            try:
                source = Path(json.loads(metadata.read_text(encoding='utf-8'))['source'])
                if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
                    return source
            except (OSError, ValueError, KeyError, TypeError):
                pass
        raise ValueError(f'No matching original video for {pose_path}')
    if len(matches) > 1:
        raise ValueError(f'Ambiguous original video for {pose_path}: {matches}')
    return matches[0]


def visualize_pose(pose_path: str, video_path: str, normalize=False, background_video=None):
    if normalize and background_video is not None:
        raise ValueError('Overlay cannot be combined with normalization')
    if background_video is not None and Path(video_path).resolve() == Path(background_video).resolve():
        raise ValueError('Output video must not overwrite the original video')
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    if normalize:
        pose = pose.normalize(pose_normalization_info(pose.header))
        normalize_pose_size(pose)

    v = PoseVisualizer(pose)

    frames = v.draw_on_video(str(background_video)) if background_video is not None else v.draw()
    v.save_video(video_path, frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=Path, help='input pose file, or directory with --recursive')
    parser.add_argument('-o', type=Path, help='output video file, or output directory with --recursive')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Find .pose files in all subdirectories; skip existing output videos')
    parser.add_argument('--normalize', action='store_true', help='Normalize pose before visualization')
    parser.add_argument('--overlay', action='store_true',
                        help='Draw on the matching original video beside each .pose file')

    args = parser.parse_args()
    if args.overlay and args.normalize:
        parser.error('--overlay cannot be combined with --normalize (coordinates would not align)')

    if args.recursive:
        if not args.i.is_dir():
            parser.error('--recursive requires an input directory (-i)')
        if args.o is not None and args.o.exists() and not args.o.is_dir():
            parser.error('with --recursive, -o must be an output directory')

        pose_files = sorted(path for path in args.i.rglob('*')
                            if path.is_file() and path.suffix.lower() == '.pose')
        print(f'Found {len(pose_files)} pose files.')
        for pose_path in pose_files:
            relative_path = pose_path.relative_to(args.i)
            output_dir = args.o / relative_path.parent if args.o is not None else pose_path.parent
            suffix = 'overlay' if args.overlay else 'skeleton'
            video_path = output_dir / f'{pose_path.stem}_{suffix}.mp4'
            if video_path.exists():
                print(f'Skipping existing video: {video_path}')
                continue
            background_video = None
            if args.overlay:
                try:
                    background_video = find_background_video(pose_path)
                except ValueError as error:
                    print(f'Skipping: {error}')
                    continue
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f'Visualizing {pose_path} -> {video_path}')
            visualize_pose(str(pose_path), str(video_path), args.normalize, background_video)
        return

    if not args.i.is_file():
        parser.error('-i must be a pose file; use --recursive for a directory')
    if args.o is None:
        parser.error('-o is required when visualizing a single pose file')
    background_video = None
    if args.overlay:
        try:
            background_video = find_background_video(args.i)
        except ValueError as error:
            parser.error(str(error))
    visualize_pose(str(args.i), str(args.o), args.normalize, background_video)
