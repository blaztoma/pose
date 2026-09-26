"""Bake an opt-in palm-local shape; leave wrist, arm, face and raw pose untouched."""
import bpy
import numpy as np
from mathutils import Matrix, Vector

from handshape_profile import FINGERS, validate_profile, weight


def bake_handshape(rig, profile, fps, count):
    validate_profile(profile)
    if profile['strength'][-1][0] > count/fps:
        raise ValueError('Handshape interval extends beyond the target video')
    letter = 'R' if profile['side']=='RIGHT' else 'L'
    hand = rig.data.bones[f'Bip01 {letter} Hand']
    fingers = [[f'Bip01 {letter} Finger{i}{suffix}' for suffix in ('','1','2')]
               for i in range(5)]
    changed = [name for chain in fingers for name in chain]
    scene = bpy.context.scene
    rest = {b.name:b.matrix_local.to_quaternion() for b in rig.data.bones}
    relative = {b.name:rest[b.parent.name].inverted() @ rest[b.name] for b in rig.data.bones if b.parent}
    # Same rest-palm frame used by the existing wrist retargeter.
    x = (rig.data.bones[fingers[2][0]].head_local-hand.head_local).normalized()
    across = rig.data.bones[fingers[1][0]].head_local-rig.data.bones[fingers[4][0]].head_local
    z = x.cross(across).normalized()
    palm = Matrix((x,z.cross(x),z)).transposed()
    hand_to_palm = rest[hand.name].inverted().to_matrix() @ palm
    local_axes = {}
    for chain in fingers:
        for j,name in enumerate(chain):
            # Match animate_pose's distal-bone convention.
            a,b = (name,chain[j+1]) if j<2 else (chain[j-1],name)
            direction = rig.data.bones[b].head_local-rig.data.bones[a].head_local
            local_axes[name] = rest[name].inverted() @ direction.normalized()
    # Snapshot before changing curves: evaluation of later frames stays original.
    originals = {}
    unaffected = [b.name for b in rig.pose.bones if b.name not in changed]
    original_unaffected = {}
    for frame in range(count):
        scene.frame_set(frame+1)
        originals[frame] = {n:rig.pose.bones[n].rotation_quaternion.copy() for n in changed}
        original_unaffected[frame] = {n:np.array(rig.pose.bones[n].matrix_basis) for n in unaffected}
    samples=[]
    for frame in range(count):
        scene.frame_set(frame+1)
        strength = weight(profile,frame/fps)
        if strength <= 0:
            continue
        for name in changed:
            rig.pose.bones[name].rotation_quaternion = originals[frame][name]
        bpy.context.view_layer.update()
        palm_now = rig.pose.bones[hand.name].matrix.to_quaternion().to_matrix() @ hand_to_palm
        errors=[]
        for i,chain in enumerate(fingers):
            for j,name in enumerate(chain):
                pb = rig.pose.bones[name]
                current = pb.matrix.to_quaternion()
                target = palm_now @ Vector(profile['directions'][i][j])
                delta = (current @ local_axes[name]).rotation_difference(target)
                parent = pb.parent.matrix.to_quaternion()
                goal_local = (parent @ relative[name]).inverted() @ (delta @ current)
                original = originals[frame][name]
                if original.dot(goal_local) < 0:
                    goal_local.negate()
                pb.rotation_quaternion = original.slerp(goal_local,strength)
                pb.keyframe_insert('rotation_quaternion',frame=frame+1,group=name)
                bpy.context.view_layer.update()
                actual = pb.matrix.to_quaternion() @ local_axes[name]
                errors.append(float(np.degrees(actual.angle(target))))
        samples.append(dict(frame_zero_based=frame,time_seconds=frame/fps,strength=strength,
                            max_direction_error_degrees=max(errors)))
    maximum=0.
    outside_error=0.
    for frame in range(count):
        scene.frame_set(frame+1)
        for name in unaffected:
            maximum=max(maximum,float(np.max(np.abs(np.array(rig.pose.bones[name].matrix_basis)-original_unaffected[frame][name]))))
        if weight(profile,frame/fps)==0:
            for name in changed:
                outside_error=max(outside_error,float(originals[frame][name].rotation_difference(
                    rig.pose.bones[name].rotation_quaternion).angle))
    full_error=max((s['max_direction_error_degrees'] for s in samples if s['strength']>.9999),default=0.)
    if maximum>1e-6 or outside_error>1e-3 or full_error>.2:
        raise ValueError(f'Handshape verification failed: unaffected={maximum}, outside={outside_error}, directions={full_error}')
    return dict(status='baked',method='finger segment directions relative to target palm; fixed target bone lengths',
        profile=profile,samples=samples,max_unaffected_local_transform_error=maximum,
        max_outside_interval_rotation_error_radians=outside_error,
        max_full_strength_direction_error_degrees=full_error,
        note='Reconstructed shape, not newly observed landmarks. No wrist or arm adjustment.')
