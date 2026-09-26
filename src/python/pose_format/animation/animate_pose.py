"""Blender experiment: MediaPipe upper-body landmarks -> Rocketbox bone animation.

Run with Blender --background --python animate_pose.py -- --job job.json.
This is an approximate monocular reconstruction, not a calibrated mocap solver.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

# Blender's Python does not have the host pose_format environment installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion import head_motion, torso_motion, smooth_valid, mp_to_rig
from progress import report
from arm_ik import calibrate_arms, solve_two_bone
from face import animate_face
from hand_contacts_blender import bake_contacts
from handshape_blender import bake_handshape

IDENTITY = Quaternion()


def basis(primary, secondary):
    x = Vector(primary).normalized()
    z = x.cross(Vector(secondary))
    if x.length < 0.5 or z.length < 1e-7:
        return None
    z.normalize()
    y = z.cross(x).normalized()
    return Matrix((x, y, z)).transposed()


def aim_object(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    job = json.loads(args.job.read_text(encoding='utf-8'))
    model = Path(job['model'])
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    report('Loading model', 0, 1, 'step')
    bpy.ops.import_scene.fbx(filepath=str(model), use_anim=False)
    report('Loading model', 1, 1, 'step')
    rigs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(rigs) != 1:
        raise ValueError('Expected one Rocketbox armature in the reference FBX')
    rig = rigs[0]
    rig.animation_data_clear()
    rig.show_in_front = True
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise ValueError('No mesh in the reference FBX')
    required_bones = {'Bip01 Pelvis', 'Bip01 L Thigh', 'Bip01 R Thigh',
                      'Bip01 Spine', 'Bip01 Spine1', 'Bip01 Spine2', 'Bip01 Head'}
    for side in ('L', 'R'):
        required_bones.update(f'Bip01 {side} {part}' for part in ('UpperArm', 'Forearm', 'Hand'))
        required_bones.update(f'Bip01 {side} Finger{finger}{suffix}'
                              for finger in range(5) for suffix in ('', '1', '2'))
    missing = required_bones - set(rig.data.bones.keys())
    if missing:
        raise ValueError(f'Unsupported reference skeleton; missing Rocketbox bones: {sorted(missing)}')
    for mesh in meshes:
        mesh.animation_data_clear()
        if mesh.data.shape_keys:
            mesh.data.shape_keys.animation_data_clear()
            for key in mesh.data.shape_keys.key_blocks:
                key.value = 0

    # Resolve the source FBX's obsolete absolute texture paths locally.
    for image in bpy.data.images:
        filename = image.filepath.replace('\\', '/').split('/')[-1]
        for directory in (model.parent.parent / 'Textures', model.parent / 'Textures', model.parent):
            path = directory / filename
            if path.is_file():
                image.filepath = str(path)
                image.reload()
                break
    materials = {material for mesh in meshes for material in mesh.data.materials if material}
    for material in materials:
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Roughness'].default_value = 0.65
            # Workbench uses the active image texture in each material.
            for node in material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image and '_color' in node.image.name:
                    material.node_tree.nodes.active = node
                    break

    data = np.load(job['data'], allow_pickle=False)
    frames = len(data['POSE_WORLD_LANDMARKS'])
    body_names = list(data['POSE_WORLD_LANDMARKS_names'])
    body = data['POSE_WORLD_LANDMARKS'].copy()
    # This project's exporter scales WORLD x/y by image width/height.
    body[..., 0] /= float(data['width'])
    body[..., 1] /= float(data['height'])
    body = mp_to_rig(body)
    body_conf = data['POSE_WORLD_LANDMARKS_confidence']
    torso_rotations, torso_valid, torso_stats = torso_motion(body, body_conf, body_names, float(data['fps']))
    head_rotations, head_valid, head_stats = head_motion(data)
    world = {}
    for side in ('LEFT', 'RIGHT'):
        ids = [body_names.index(f'{side}_{part}') for part in ('SHOULDER','ELBOW','WRIST')]
        valid = (body_conf[:, ids] > 0.35).all(axis=1)
        world[side] = smooth_valid(body[:, ids], valid)

    hands = {}
    for side in ('LEFT', 'RIGHT'):
        key = side + '_HAND_LANDMARKS'
        points = data[key].copy()
        # Hand z is wrist-relative, normalized approximately like image x.
        points[..., 2] *= float(data['width'])
        points -= points[:, :1]
        points = mp_to_rig(points)
        valid = (data[key + '_confidence'] > 0).all(axis=1)
        hands[side] = smooth_valid(points, valid)

    rest_rot = {b.name: b.matrix_local.to_quaternion() for b in rig.data.bones}
    rest_head = {b.name: b.head_local.copy() for b in rig.data.bones}
    relative = {b.name: (rest_rot[b.parent.name].inverted() @ rest_rot[b.name]
                         if b.parent else rest_rot[b.name]) for b in rig.data.bones}
    last_local = {}
    mapping = {}
    stats = {'frames': frames, 'fps': float(data['fps']), 'animated_bones': [],
             'torso_motion': torso_stats, 'head_motion': head_stats,
             'note': 'Head rotation and shoulder-based torso yaw/side lean, relative to initial neutral frames. No forward torso bend. Fixed lower body. Missing head/torso samples hold local rotations. Short hand gaps interpolated; long gaps hold local hand/finger rotations. Unreliable arms ease to a neutral hanging pose.'}

    def segment(a, b):
        return rest_head[b] - rest_head[a]

    arm_solver = job.get('arm_solver', 'ik')
    avatar = {'shoulder_width': segment('Bip01 R UpperArm', 'Bip01 L UpperArm').length}
    for side, letter in [('LEFT', 'L'), ('RIGHT', 'R')]:
        avatar[side] = (segment(f'Bip01 {letter} UpperArm', f'Bip01 {letter} Forearm').length,
                        segment(f'Bip01 {letter} Forearm', f'Bip01 {letter} Hand').length)
    report('Calibrating arms', 0, 1, 'step')
    targets, calibration = calibrate_arms(data, body, body_conf, body_names, avatar)
    stats['arm_solver'] = arm_solver
    stats['arm_calibration'] = calibration
    if arm_solver == 'ik':
        stats['note'] += ' IK uses image-space gesture position, estimated world depth and fixed avatar link lengths.'
    report('Calibrating arms', 1, 1, 'step')
    ik_world = np.zeros((frames, 2, 3))
    ik_valid = np.zeros((frames, 2), dtype=bool)
    ik_errors, clamp_errors = [], []
    previous_bend = {}

    scene = bpy.context.scene
    scene.render.fps = round(float(data['fps']))
    scene.render.fps_base = scene.render.fps / float(data['fps'])
    scene.frame_start = 1
    scene.frame_end = frames
    report('Animating', 0, frames)
    for frame in range(frames):
        scene.frame_set(frame+1)
        global_rot = {}

        def inherited(name):
            if name in global_rot:
                return global_rot[name]
            bone = rig.data.bones[name]
            parent = inherited(bone.parent.name) if bone.parent else IDENTITY
            return parent @ relative[name]

        def posed_head(name):
            bone = rig.data.bones[name]
            if bone.parent is None:
                return rest_head[name].copy()
            parent = bone.parent.name
            offset = rest_rot[parent].inverted() @ (rest_head[name] - rest_head[parent])
            return posed_head(parent) + inherited(parent) @ offset

        def apply(name, goal=None, blend=0.6):
            bone = rig.data.bones[name]
            parent = inherited(bone.parent.name) if bone.parent else IDENTITY
            base = parent @ relative[name]
            local = base.inverted() @ goal if goal is not None else last_local.get(name, IDENTITY).copy()
            previous = last_local.get(name)
            if previous is not None:
                if previous.dot(local) < 0:
                    local.negate()
                # Ease into new observations after missing detections.
                local = previous.slerp(local, blend)
            local.normalize()
            pb = rig.pose.bones[name]
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = local
            pb.keyframe_insert('rotation_quaternion', frame=frame+1, group=name)
            last_local[name] = local.copy()
            global_rot[name] = base @ local

        # Legs are children of Spine in Rocketbox: leave Spine/Pelvis fixed.
        # Distribute torso rotation over Spine1/Spine2, using cumulative world goals.
        torso_delta = Matrix(torso_rotations[frame].tolist()).to_quaternion()
        for name, weight in [('Bip01 Spine1', 0.45), ('Bip01 Spine2', 1.0)]:
            goal = IDENTITY.slerp(torso_delta, weight) @ rest_rot[name] if torso_valid[frame] else None
            apply(name, goal)
            mapping[name] = 'WORLD shoulders: relative yaw and side lean; hips ignored'
        # Head is an absolute orientation goal, so torso rotation is not added twice.
        # Keep Neck local rotation unchanged: Rocketbox clavicles are its children.
        goal = (Matrix(head_rotations[frame].tolist()).to_quaternion() @ rest_rot['Bip01 Head']
                if head_valid[frame] else None)
        apply('Bip01 Head', goal)
        mapping['Bip01 Head'] = 'FACE upper-face rigid fit; independent of torso orientation'

        shoulder_center = (posed_head('Bip01 L UpperArm') + posed_head('Bip01 R UpperArm')) / 2

        for side, letter in [('LEFT','L'), ('RIGHT','R')]:
            upper = f'Bip01 {letter} UpperArm'
            fore = f'Bip01 {letter} Forearm'
            hand = f'Bip01 {letter} Hand'
            rest_upper, rest_fore = segment(upper, fore), segment(fore, hand)
            points, valid = world[side]
            offsets, usable = targets[side]
            use_ik = arm_solver == 'ik' and usable[frame]
            if use_ik or valid[frame]:
                if use_ik:
                    shoulder = posed_head(upper)
                    pole, target = np.asarray(shoulder_center) + offsets[frame]
                    elbow, wrist, bend, clamped = solve_two_bone(
                        shoulder, target, pole, *avatar[side], previous_bend.get(side))
                    previous_bend[side] = bend
                    clamp_errors.append(clamped)
                    elbow, wrist = Vector(elbow), Vector(wrist)
                else:
                    shoulder, elbow, wrist = map(Vector, points[frame])
                u, f = elbow-shoulder, wrist-elbow
                for name, direction, secondary, ref, ref_secondary in [
                    (upper,u,f,rest_upper,rest_fore), (fore,f,u,rest_fore,rest_upper)]:
                    target_basis, rest_basis = basis(direction, secondary), basis(ref, ref_secondary)
                    if target_basis is not None and rest_basis is not None:
                        goal = (target_basis @ rest_basis.transposed()).to_quaternion() @ rest_rot[name]
                    else:
                        goal = ref.normalized().rotation_difference(direction.normalized()) @ rest_rot[name]
                    # Targets are already smoothed. Smoothing solved rotations
                    # again would move the wrist away from the IK target.
                    apply(name, goal, blend=1.0 if use_ik else .6)
                if use_ik:
                    side_index = 0 if side == 'LEFT' else 1
                    ik_world[frame, side_index] = rig.matrix_world @ wrist
                    ik_valid[frame, side_index] = True
                    ik_errors.append((posed_head(hand) - wrist).length)
            else:
                # Unobserved arms ease to a declared neutral fallback, not A-pose.
                sign = 1 if side == 'LEFT' else -1
                u, f = Vector((0, 0.15*sign, -1)), Vector((0.12, 0.05*sign, -1))
                for name, direction, secondary, ref, ref_secondary in [
                    (upper,u,f,rest_upper,rest_fore), (fore,f,u,rest_fore,rest_upper)]:
                    delta = basis(direction, secondary) @ basis(ref, ref_secondary).transposed()
                    apply(name, delta.to_quaternion() @ rest_rot[name], blend=0.15)
            mapping[upper] = (f'{side}: calibrated wrist target + elbow pole (IK), world-rotation fallback'
                              if arm_solver == 'ik' else f'{side}_SHOULDER -> {side}_ELBOW (world)')
            mapping[fore] = (f'{side}: fixed-length analytical IK; baked rotation keyframes'
                             if arm_solver == 'ik' else f'{side}_ELBOW -> {side}_WRIST (world)')

            hp, hv = hands[side]
            h = hp[frame]
            palm_rest = basis(segment(hand, f'Bip01 {letter} Finger2'),
                              rest_head[f'Bip01 {letter} Finger1'] - rest_head[f'Bip01 {letter} Finger4'])
            palm_target = basis(h[9]-h[0], h[5]-h[17]) if hv[frame] else None
            goal = ((palm_target @ palm_rest.transposed()).to_quaternion() @ rest_rot[hand]
                    if palm_target is not None else None)
            apply(hand, goal)
            mapping[hand] = f'{side}_HAND: wrist, middle MCP, index MCP, little MCP'

            for finger in range(5):
                start = 1 + 4*finger
                chain = [f'Bip01 {letter} Finger{finger}{suffix}' for suffix in ('','1','2')]
                for joint, name in enumerate(chain):
                    goal = None
                    if hv[frame]:
                        target = Vector(h[start+joint+1]-h[start+joint])
                        rest_dir = (segment(name, chain[joint+1]) if joint < 2
                                    else segment(chain[joint-1], name))
                        parent_name = rig.data.bones[name].parent.name
                        base = inherited(parent_name) @ relative[name]
                        current_direction = base @ (rest_rot[name].inverted() @ rest_dir)
                        if target.length > 1e-6 and current_direction.length > 1e-6:
                            goal = current_direction.normalized().rotation_difference(target.normalized()) @ base
                    apply(name, goal)
                    mapping[name] = f'{side}_HAND landmarks {start+joint} -> {start+joint+1}'

        report('Animating', frame + 1, frames)

    if job.get('handshape_profile'):
        report('Transferring handshape', 0, 1, 'step')
        stats['handshape'] = bake_handshape(rig, job['handshape_profile'], float(data['fps']), frames)
        report('Transferring handshape', 1, 1, 'step')
    if job.get('hand_contacts'):
        report('Fitting hand contacts', 0, 1, 'step')
        stats['hand_contacts'] = bake_contacts(rig, meshes, job['hand_contacts'], float(data['fps']), frames)
        # Contact correction deliberately refines the inferred right wrist targets.
        for sample in stats['hand_contacts']['samples']:
            frame = sample['frame'] - 1
            scene.frame_set(frame+1)
            bpy.context.view_layer.update()
            ik_world[frame, 1] = rig.matrix_world @ rig.pose.bones['Bip01 R Hand'].matrix.translation
        report('Fitting hand contacts', 1, 1, 'step')
    stats['face'] = animate_face(meshes, data, report, job.get('mouth_calibration', 'auto'))
    stats['face']['input'] = job.get('face_input', {})
    rig.animation_data.action.name = job['name'] + '_upper_body_from_pose'
    stats['animated_bones'] = sorted(last_local)
    stats['mapping'] = mapping
    stats['arm_ik'] = {'solved_frames_left': int(ik_valid[:, 0].sum()),
                       'solved_frames_right': int(ik_valid[:, 1].sum()),
                       'clamped_targets': int(np.count_nonzero(np.asarray(clamp_errors) > 1e-4)),
                       'max_target_clamp_rig_units': max(clamp_errors, default=0.),
                       'max_solver_error_rig_units': max(ik_errors, default=0.),
                       'targets_file': str(Path(job['report']).with_name('ik_targets.npz'))}
    if max(ik_errors, default=0.) > avatar['shoulder_width'] * 1e-4:
        raise ValueError(f'IK forward-kinematic validation failed: {max(ik_errors)}')
    np.savez_compressed(stats['arm_ik']['targets_file'], targets_world=ik_world, valid=ik_valid)
    for side in hands:
        stats[side.lower() + '_hand_frames_after_short_gap_fill'] = int(hands[side][1].sum())
    # Confirm no lower-body animation curves were introduced.
    assert not any(any(s in name for s in ('Thigh','Calf','Foot','Toe','Pelvis')) for name in last_local)
    Path(job['report']).write_text(json.dumps(stats, indent=2), encoding='utf-8')

    # Export only the character and baked animation, not preview camera/lights.
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig
    scene.frame_set(1)
    report('Exporting FBX', 0, 1, 'step')
    bpy.ops.export_scene.fbx(filepath=job['fbx'], use_selection=True,
                            object_types={'ARMATURE','MESH'}, add_leaf_bones=False,
                            bake_anim=True, bake_anim_use_all_actions=False,
                            bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0,
                            path_mode='COPY' if job['embed_textures'] else 'RELATIVE',
                            embed_textures=job['embed_textures'])
    report('Exporting FBX', 1, 1, 'step')

    bpy.ops.object.camera_add(location=(0, -4.0, 1.50))
    camera = bpy.context.object
    camera.name = 'Front_Preview'
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 1.55
    aim_object(camera, (0,0,1.48))
    scene.camera = camera
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'TEXTURE'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.display.shading.background_type = 'WORLD'
    scene.world.color = (0.11,0.13,0.16)
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    # Save a convenient camera preview layout in the editable scene.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.shading.color_type = 'TEXTURE'
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    if job['embed_textures']:
        bpy.ops.file.pack_all()
    report('Saving Blender scene', 0, 1, 'step')
    bpy.ops.wm.save_as_mainfile(filepath=job['blend'])
    report('Saving Blender scene', 1, 1, 'step')
    print('DONE: animation, FBX, Blender scene, mapping report')


if __name__ == '__main__':
    main()
