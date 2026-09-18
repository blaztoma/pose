"""Export MediaPipe .pose arrays for Blender's bundled Python (no extra packages)."""
import argparse
import json
from pathlib import Path

import numpy as np
from pose_format import Pose
if __package__:
    from .face import find_extras, load_face
else:  # Keep the standalone prepare_pose.py command usable.
    from face import find_extras, load_face


def prepare_pose(source: Path, output: Path, face_animation='auto'):
    pose = Pose.read(source.read_bytes())
    frames, people, _, dimensions = pose.body.data.shape
    if frames == 0 or people != 1 or dimensions != 3:
        raise ValueError('Expected a nonempty 3D pose with exactly one person')
    if not np.isfinite(pose.body.fps) or pose.body.fps <= 0:
        raise ValueError('Pose FPS must be positive and finite')
    if pose.header.dimensions.width <= 0 or pose.header.dimensions.height <= 0:
        raise ValueError('Pose image width and height must be positive')
    required = {'POSE_WORLD_LANDMARKS', 'LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS'}
    missing = required - {c.name for c in pose.header.components}
    if missing:
        raise ValueError(f'Missing required MediaPipe components: {sorted(missing)}')
    arrays = {'fps': np.array(float(pose.body.fps)),
              'width': np.array(pose.header.dimensions.width),
              'height': np.array(pose.header.dimensions.height)}
    summary = {'input': str(source.resolve()), 'frames': len(pose.body.data),
               'fps': float(pose.body.fps), 'components': {}}
    offset = 0
    for component in pose.header.components:
        stop = offset + len(component.points)
        name = component.name
        arrays[name] = pose.body.data.data[:, 0, offset:stop].copy()
        arrays[name + '_confidence'] = pose.body.confidence[:, 0, offset:stop].copy()
        arrays[name + '_names'] = np.array(component.points)
        if name in required and not np.isfinite(arrays[name]).all():
            raise ValueError(f'Non-finite coordinates in {name}')
        if name.endswith('_HAND_LANDMARKS'):
            expected = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP']
            expected += [f'{finger}_{joint}' for finger in ('INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY')
                         for joint in ('MCP', 'PIP', 'DIP', 'TIP')]
            if list(component.points) != expected:
                raise ValueError(f'Unexpected MediaPipe hand landmark order in {name}')
        detected = arrays[name + '_confidence'].max(axis=1) > 0
        ids = np.flatnonzero(detected)
        groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1)
        summary['components'][name] = {
            'detected_frames': int(detected.sum()),
            'detected_ranges_zero_based': [[int(g[0]), int(g[-1])] for g in groups if len(g)]}
        offset = stop
    extras = find_extras(source) if face_animation != 'off' else None
    summary['face'] = {'status': 'disabled' if face_animation == 'off' else 'missing_sidecar'}
    if extras is not None:
        face_arrays, summary['face'] = load_face(extras, frames, float(pose.body.fps))
        arrays.update(face_arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', required=True, type=Path)
    parser.add_argument('-o', required=True, type=Path)
    args = parser.parse_args()
    summary = prepare_pose(args.i, args.o)
    print(f'Exported {summary["frames"]} frames to {args.o}')


if __name__ == '__main__':
    main()
