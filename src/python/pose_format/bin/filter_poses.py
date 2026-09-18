"""Mark lower-body MediaPipe landmarks as unreliable, preserving originals."""

import argparse
from pathlib import Path

import numpy.ma as ma

from pose_format.pose import Pose


def filter_lower_body(pose: Pose, include_hips: bool = False) -> int:
    """Set confidence to zero and mask leg points in every frame/person."""
    parts = {'KNEE', 'ANKLE', 'HEEL', 'FOOT_INDEX'}
    if include_hips:
        parts.add('HIP')
    names = {f'{side}_{part}' for side in ('LEFT', 'RIGHT') for part in parts}
    indices = []
    offset = 0
    for component in pose.header.components:
        if component.name in ('POSE_LANDMARKS', 'POSE_WORLD_LANDMARKS'):
            indices.extend(offset + index for index, name in enumerate(component.points) if name in names)
        offset += len(component.points)
    if not indices:
        raise ValueError('No supported MediaPipe lower-body landmarks found')

    pose.body.confidence[..., indices] = 0
    mask = ma.getmaskarray(pose.body.data).copy()
    mask[..., indices, :] = True
    pose.body.data = ma.array(pose.body.data.data, mask=mask, copy=False)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', required=True, type=Path, help='Input .pose file or directory')
    parser.add_argument('-r', '--recursive', action='store_true', help='Include subdirectories')
    parser.add_argument('--include-hips', action='store_true', help='Also mark hip points as unreliable')
    args = parser.parse_args()

    if args.i.is_dir():
        paths = args.i.rglob('*') if args.recursive else args.i.iterdir()
        sources = sorted(path for path in paths if path.is_file() and path.suffix.lower() == '.pose')
    elif args.i.is_file() and args.i.suffix.lower() == '.pose':
        sources = [args.i]
    else:
        parser.error('-i must be an existing .pose file or directory')

    sources = [path for path in sources if not path.stem.lower().endswith('_filtered')]
    print(f'Found {len(sources)} original pose files.')
    completed = 0
    for source in sources:
        destination = source.with_name(f'{source.stem}_filtered.pose')
        if destination.exists():
            print(f'Skipping existing file: {destination}')
            continue
        pose = Pose.read(source.read_bytes())
        try:
            count = filter_lower_body(pose, args.include_hips)
        except ValueError as error:
            print(f'Skipping {source}: {error}')
            continue
        with destination.open('wb') as stream:
            pose.write(stream)
        completed += 1
        print(f'{source} -> {destination} ({count} landmarks masked)')
    print(f'Created {completed} filtered pose files.')


if __name__ == '__main__':
    main()
