"""Compare exported FBX animation against the saved Blender scene."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progress import report as report_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    job = json.loads(args.job.read_text(encoding='utf-8'))
    report = json.loads(Path(job['report']).read_text(encoding='utf-8'))
    count = report['frames']
    frames = sorted({1, count, *[1 + round((count-1)*t/4) for t in range(5)]})
    animated = report['animated_bones']
    bpy.ops.wm.open_mainfile(filepath=job['blend'])
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    static = [b.name for b in rig.data.bones if b.name not in animated]
    names = animated + static

    checked = 0
    report_progress('Checking FBX', 0, 3 * len(frames), 'sample')

    def sample(armature, local=False):
        nonlocal checked
        result = {}
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            result[frame] = {name: np.array(armature.pose.bones[name].matrix_basis if local else
                                          armature.matrix_world @ armature.pose.bones[name].matrix)
                             for name in names}
            checked += 1
            report_progress('Checking FBX', checked, 3 * len(frames), 'sample')
        return result

    expected = sample(rig)
    local_expected = sample(rig, local=True)
    for name in static:
        for frame in frames:
            if not np.allclose(local_expected[frame][name], local_expected[1][name], atol=1e-5):
                raise ValueError(f'Unexpected local animation in bone: {name}')
    # Face/neck/clavicles may now inherit head/torso movement, but legs must stay fixed in world space.
    lower = [name for name in names if name in ('Bip01 Pelvis', 'Bip01 Spine')
             or any(part in name for part in ('Thigh', 'Calf', 'Foot', 'Toe'))]
    for name in lower:
        for frame in frames:
            if not np.allclose(expected[frame][name], expected[1][name], atol=1e-5):
                raise ValueError(f'Unexpected lower-body motion: {name}')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=job['fbx'], anim_offset=0)
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    if not rig.animation_data or not rig.animation_data.action:
        raise ValueError('FBX has no animation action')
    actual = sample(rig)
    max_error = max(float(np.max(np.abs(actual[f][name] - expected[f][name])))
                    for f in frames for name in names)
    if max_error > 0.002:
        raise ValueError(f'FBX bone transforms differ from Blender scene (max error {max_error})')
    action_range = list(rig.animation_data.action.frame_range)
    if not np.allclose(action_range, [1, count], atol=0.01):
        raise ValueError(f'Unexpected FBX action range: {action_range}')
    moving = [name for name in animated if any(
        not np.allclose(expected[f][name], expected[1][name], atol=1e-5) for f in frames)]
    result = {'fbx_reimport': 'passed', 'action_range': action_range,
              'unanimated_bones_locally_static': True, 'lower_body_world_static': True,
              'moving_bones_at_sampled_frames': moving,
              'max_transform_error': max_error, 'samples': frames,
              'note': 'Structural check; does not certify gesture accuracy or MetaHuman retargeting.'}
    Path(job['validation']).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('FBX validation passed')


if __name__ == '__main__':
    main()
