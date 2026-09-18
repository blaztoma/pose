"""Compare exported FBX animation against the saved Blender scene."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progress import report as report_progress
from face import sample_face
from mouth_calibration import mesh_mouth_ratio


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
    face_mapping = report.get('face', {}).get('mapping', [])
    mouth = report.get('face', {}).get('mouth_calibration', {})
    mouth_samples = mouth.get('samples', []) if mouth.get('status') == 'calibrated' else []
    face_frames = sorted(set(frames) | {item[k] for item in face_mapping for k in ('peak_frame', 'min_frame')})
    face_frames = sorted(set(face_frames) | {item['frame'] for item in mouth_samples})
    face_expected = sample_face(bpy.context.scene.objects, face_mapping, face_frames, bpy.context.scene)
    # Confirm mapped keys deform real geometry, rather than empty placeholder shapes.
    shape_deltas = {}
    for item in face_mapping:
        keys = bpy.context.scene.objects[item['mesh']].data.shape_keys
        key = keys.key_blocks[item['key']]
        base = key.relative_key
        a = np.empty(len(key.data) * 3, dtype=np.float32)
        b = np.empty_like(a)
        key.data.foreach_get('co', a)
        base.data.foreach_get('co', b)
        shape_deltas[item['key']] = float(np.max(np.abs(a - b)))
        if shape_deltas[item['key']] < 1e-8:
            raise ValueError(f'Face shape has no vertex deformation: {item["key"]}')
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
    face_actual = sample_face(bpy.context.scene.objects, face_mapping, face_frames, bpy.context.scene)
    face_error = float(np.max(np.abs(face_actual - face_expected))) if face_mapping else 0.
    if face_error > .001:
        raise ValueError(f'FBX facial animation differs from Blender (max coefficient error {face_error})')
    mouth_geometry_error = 0.
    for item in mouth_samples:
        bpy.context.scene.frame_set(item['frame'])
        mesh = bpy.context.scene.objects[mouth['mesh']]
        ratio = mesh_mouth_ratio(mesh, mouth['anchor_vertices_upper_lower_left_right'])
        mouth_geometry_error = max(mouth_geometry_error, abs(ratio - item['after_ratio']))
    if mouth_geometry_error > .001:
        raise ValueError(f'FBX mouth geometry differs from calibration (ratio error {mouth_geometry_error})')
    ik_error = 0.0
    ik_checks = 0
    if report.get('arm_ik'):
        ik = np.load(report['arm_ik']['targets_file'])
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            for side, letter in enumerate(('L', 'R')):
                if ik['valid'][frame - 1, side]:
                    position = rig.matrix_world @ rig.pose.bones[f'Bip01 {letter} Hand'].matrix.translation
                    error = float(np.linalg.norm(np.asarray(position) - ik['targets_world'][frame - 1, side]))
                    ik_error = max(ik_error, error)
                    ik_checks += 1
        if ik_error > .002:
            raise ValueError(f'Exported FBX wrists miss IK targets (world error {ik_error})')
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
              'ik_wrist_samples_checked': ik_checks, 'max_ik_wrist_error_world_units': ik_error,
              'face': {'status': 'passed' if face_mapping else 'inactive', 'channels_checked': len(face_mapping),
                       'sample_frames': face_frames if face_mapping else [], 'max_coefficient_error': face_error,
                       'shape_vertex_deltas': shape_deltas},
              'mouth_geometry': {'samples_checked': len(mouth_samples), 'max_ratio_error': mouth_geometry_error},
              'note': 'Structural check; does not certify gesture accuracy or MetaHuman retargeting.'}
    Path(job['validation']).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('FBX validation passed')


if __name__ == '__main__':
    main()
