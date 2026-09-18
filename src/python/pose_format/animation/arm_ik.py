"""Automatic proportion calibration and analytical two-bone IK (NumPy only).

Coordinates are armature-local: forward, left, up. Image observations anchor
the visible gesture; monocular world landmarks supply estimated depth only.
"""
import numpy as np
try:
    from .motion import smooth_valid
except ImportError:  # Blender executes the sibling scripts outside the package.
    from motion import smooth_valid


def robust_length(values, valid, minimum_samples=5):
    values = np.asarray(values)
    selected = values[valid & np.isfinite(values) & (values > 1e-6)]
    if len(selected) < minimum_samples:
        return None, len(selected)
    median = np.median(selected)
    mad = np.median(np.abs(selected - median))
    selected = selected[np.abs(selected - median) <= max(4.5 * mad, .15 * median)]
    if len(selected) < minimum_samples:
        return None, len(selected)
    return float(np.median(selected)), len(selected)


def calibrate_arms(data, body, confidence, names, avatar):
    """Return smoothed wrist/elbow offsets from shoulder center and diagnostics.

avatar: shoulder_width and LEFT/RIGHT (upper_length, forearm_length), in rig
units. Never calibrate from hips/legs. Insufficient evidence selects the
existing rotation fallback instead of inventing person measurements.
"""
    names = list(names)
    shoulders = [names.index(s + '_SHOULDER') for s in ('LEFT', 'RIGHT')]
    center = body[:, shoulders].mean(axis=1)
    shoulder_vector = body[:, shoulders[0]] - body[:, shoulders[1]]
    reliable = ((confidence[:, shoulders] > .6).all(axis=1)
                & np.isfinite(body[:, shoulders]).all(axis=(1, 2)))
    width, width_samples = robust_length(np.linalg.norm(shoulder_vector, axis=1), reliable)
    result = {'method': 'image wrist targets + world depth + analytical two-bone IK',
              'source_shoulder_width_m': width, 'shoulder_samples': width_samples,
              'avatar_shoulder_width': float(avatar['shoulder_width']),
              'avatar_units': 'armature local units', 'sides': {}}
    image = data.get('POSE_LANDMARKS')
    image_conf = data.get('POSE_LANDMARKS_confidence')
    image_names = list(data.get('POSE_LANDMARKS_names', []))
    image_scale = None
    if image is not None and image_conf is not None and all(names[i] in image_names for i in shoulders):
        ids = [image_names.index(names[i]) for i in shoulders]
        image_center = image[:, ids, :2].mean(axis=1)
        image_widths = np.linalg.norm(image[:, ids[0], :2] - image[:, ids[1], :2], axis=1)
        image_valid = ((image_conf[:, ids] > .6).all(axis=1)
                       & np.isfinite(image[:, ids, :2]).all(axis=(1, 2)))
        # Prefer near-frontal frames: yaw must not shrink the calibrated width.
        frontal = np.abs(shoulder_vector[:, 0]) < .35 * np.linalg.norm(shoulder_vector, axis=1)
        pixels, samples = robust_length(image_widths, reliable & image_valid & frontal)
        if pixels is None:
            pixels, samples = robust_length(image_widths, reliable & image_valid)
        if pixels is not None:
            image_scale = avatar['shoulder_width'] / pixels
        result.update(source_shoulder_width_pixels=pixels, image_calibration_samples=samples)
    else:
        image_valid = np.zeros(len(body), dtype=bool)
    targets = {}
    for side in ('LEFT', 'RIGHT'):
        indices = [names.index(side + '_' + joint) for joint in ('SHOULDER', 'ELBOW', 'WRIST')]
        points = body[:, indices]
        valid = (reliable & (confidence[:, indices] > .5).all(axis=1)
                 & np.isfinite(points).all(axis=(1, 2)))
        upper, upper_samples = robust_length(np.linalg.norm(points[:, 1] - points[:, 0], axis=1), valid)
        fore, fore_samples = robust_length(np.linalg.norm(points[:, 2] - points[:, 1], axis=1), valid)
        a, b = avatar[side]
        info = {'source_upper_arm_m': upper, 'source_forearm_m': fore,
                'upper_samples': upper_samples, 'forearm_samples': fore_samples,
                'avatar_upper_arm': float(a), 'avatar_forearm': float(b)}
        result['sides'][side] = info
        if width is None or upper is None or fore is None:
            info['status'] = 'rotation fallback: fewer than 5 reliable calibration samples'
            targets[side] = (np.zeros((len(body), 2, 3)), np.zeros(len(body), dtype=bool))
            continue
        scale = avatar['shoulder_width'] / width
        # Shared lateral/vertical scale preserves two-hand spacing. Arm length
        # calibration adjusts depth, not separate lateral scales for each hand.
        depth_scale = (a + b) / (upper + fore)
        offsets = (points[:, [1, 2]] - center[:, None]) * scale
        offsets[..., 0] = (points[:, [1, 2], 0] - center[:, None, 0]) * depth_scale
        hand_samples = 0
        if image_scale is not None and all(side + '_' + j in image_names for j in ('ELBOW', 'WRIST')):
            ids = [image_names.index(side + '_' + j) for j in ('ELBOW', 'WRIST')]
            xy = image[:, ids, :2].copy()
            xy_valid = image_valid & (image_conf[:, ids] > .5).all(axis=1)
            hand = data.get(side + '_HAND_LANDMARKS')
            hand_conf = data.get(side + '_HAND_LANDMARKS_confidence')
            if hand is not None and hand_conf is not None:
                hand_ok = ((hand_conf[:, 0] > 0) & np.isfinite(hand[:, 0, :2]).all(axis=1)
                           & (np.linalg.norm(hand[:, 0, :2] - xy[:, 1], axis=1) < .5 * pixels))
                xy[hand_ok, 1] = hand[hand_ok, 0, :2]
                hand_samples = int((hand_ok & valid & xy_valid).sum())
            xy_valid &= np.isfinite(xy).all(axis=(1, 2))
            xy -= image_center[:, None]
            offsets[xy_valid, :, 1] = xy[xy_valid, :, 0] * image_scale
            offsets[xy_valid, :, 2] = -xy[xy_valid, :, 1] * image_scale
            info['image_target_frames'] = int((xy_valid & valid).sum())
        offsets, usable = smooth_valid(offsets, valid)
        info.update(status='calibrated', lateral_scale=scale, depth_scale=depth_scale,
                    image_scale=image_scale, observed_frames=int(valid.sum()),
                    usable_frames=int(usable.sum()), detailed_hand_wrist_frames=hand_samples)
        targets[side] = offsets, usable
    return targets, result


def solve_two_bone(shoulder, target, pole, upper_length, forearm_length, previous_bend=None):
    """Solve fixed-length links; clamp unreachable targets without stretching.

Returns elbow, reachable wrist, bend direction and target-clamping distance.
The pole determines which side of the shoulder-wrist line the elbow uses.
"""
    shoulder, target, pole = (np.asarray(p, dtype=float) for p in (shoulder, target, pole))
    if not all(np.isfinite(p).all() for p in (shoulder, target, pole)):
        raise ValueError('Non-finite IK target')
    a, b = float(upper_length), float(forearm_length)
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError('IK bone lengths must be positive')
    offset = target - shoulder
    distance = np.linalg.norm(offset)
    direction = offset / distance if distance > 1e-9 else np.array([0., 0., -1.])
    epsilon = (a + b) * 1e-6
    reach = np.clip(distance, abs(a - b) + epsilon, a + b - epsilon)
    wrist = shoulder + reach * direction
    bend = pole - shoulder
    bend -= direction * np.dot(bend, direction)
    if np.linalg.norm(bend) < (a + b) * 1e-4 and previous_bend is not None:
        bend = np.asarray(previous_bend) - direction * np.dot(previous_bend, direction)
    if np.linalg.norm(bend) < (a + b) * 1e-4:
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        bend = axis - direction * np.dot(axis, direction)
    bend /= np.linalg.norm(bend)
    along = (a*a - b*b + reach*reach) / (2 * reach)
    height = np.sqrt(max(0., a*a - along*along))
    elbow = shoulder + along * direction + height * bend
    return elbow, wrist, bend, float(np.linalg.norm(wrist - target))
