"""Numpy-only upper-body orientation estimates for waist-up MediaPipe footage."""
import numpy as np


# Upper face / nose / cheek anchors avoid lips, jaw and refined iris points.
HEAD_ANCHORS = ('10', '33', '133', '362', '263', '168', '6', '1', '127', '356', '234', '454')


def smooth_valid(points, valid, max_gap=5):
    """Interpolate short internal gaps and smooth without crossing long gaps."""
    points, valid = points.copy(), valid.copy()
    ids = np.flatnonzero(valid)
    for a, b in zip(ids[:-1], ids[1:]):
        if 1 < b - a <= max_gap + 1:
            for i in range(a + 1, b):
                t = (i-a)/(b-a)
                points[i] = (1-t)*points[a] + t*points[b]
                valid[i] = True
    result = points.copy()
    for i in np.flatnonzero(valid):
        neighbors = [j for j in range(max(0, i-2), min(len(points), i+3))
                     if valid[min(i, j):max(i, j)+1].all()]
        result[i] = np.average(points[neighbors], axis=0,
                               weights=[3-abs(j-i) for j in neighbors])
    return result, valid


def mp_to_rig(points):
    """Image right/down/away -> Rocketbox forward/left/up."""
    return np.stack((-points[..., 2], points[..., 0], -points[..., 1]), axis=-1)


def rigid_rotation(reference, points):
    """Least-squares rotation, invariant to translation and positive uniform scale."""
    reference = reference - reference.mean(axis=0)
    points = points - points.mean(axis=0)
    a, b = np.linalg.norm(reference), np.linalg.norm(points)
    if min(a, b) < 1e-8:
        raise ValueError('Degenerate head landmarks')
    reference, points = reference/a, points/b
    u, singular, vt = np.linalg.svd(reference.T @ points)
    if singular[1] < 1e-5:
        raise ValueError('Collinear head landmarks')
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    residual = float(np.linalg.norm(reference @ rotation.T - points))
    return rotation, residual


def rotation_angle(rotation):
    return float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))


def reference_frames(valid, fps):
    """Use the first contiguous detected interval, at most 0.4 s, as neutral."""
    ids = np.flatnonzero(valid)
    if not len(ids):
        return ids
    start = ids[0]
    stop = min(len(valid), start + max(1, round(0.4*fps)))
    for i in range(start, stop):
        if not valid[i]:
            stop = i
            break
    return np.arange(start, stop)


def shoulder_basis(axis):
    """Two observable DOFs from the shoulder line; no inferred forward bend."""
    axis = axis/np.linalg.norm(axis)
    yaw = np.arctan2(-axis[0], axis[1])
    roll = np.arctan2(axis[2], np.hypot(axis[0], axis[1]))
    cy, sy, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ rx


def torso_motion(body, confidence, names, fps):
    """Body is already in armature-local metric coordinates. Never uses hips."""
    frames = len(body)
    rotations = np.repeat(np.eye(3)[None], frames, axis=0)
    indices = [list(names).index(side + '_SHOULDER') for side in ('LEFT', 'RIGHT')]
    axis = body[:, indices[0]] - body[:, indices[1]]
    valid = ((confidence[:, indices] > 0.5).all(axis=1) & np.isfinite(axis).all(axis=1)
             & (np.linalg.norm(axis, axis=1) > 0.05))
    observed = int(valid.sum())
    axis, valid = smooth_valid(axis, valid)
    calibration = reference_frames(valid, fps)
    if len(calibration):
        neutral_axis = np.median(axis[calibration], axis=0)
        if np.linalg.norm(neutral_axis) < 1e-8:
            valid[:] = False
            calibration = np.array([], dtype=int)
        else:
            neutral = shoulder_basis(neutral_axis)
    if len(calibration):
        for i in np.flatnonzero(valid):
            if np.linalg.norm(axis[i]) < 1e-8:
                valid[i] = False
                continue
            rotation = shoulder_basis(axis[i]) @ neutral.T
            if rotation_angle(rotation) > np.deg2rad(45):
                valid[i] = False
            else:
                rotations[i] = rotation
    return rotations, valid, {'observed_frames': observed,
                              'usable_frames': int(valid.sum()),
                              'neutral_frames_zero_based': calibration.tolist(),
                              'method': 'shoulder-line yaw and side lean; no hips, forward bend or root translation'}


def head_motion(data):
    frames = len(data['POSE_WORLD_LANDMARKS'])
    rotations = np.repeat(np.eye(3)[None], frames, axis=0)
    valid = np.zeros(frames, dtype=bool)
    if 'FACE_LANDMARKS' not in data:
        return rotations, valid, {'observed_frames': 0, 'usable_frames': 0, 'method': 'no face component'}
    names = list(data['FACE_LANDMARKS_names'])
    if not set(HEAD_ANCHORS).issubset(names):
        return rotations, valid, {'observed_frames': 0, 'usable_frames': 0, 'method': 'missing upper-face anchors'}
    indices = [names.index(name) for name in HEAD_ANCHORS]
    points = data['FACE_LANDMARKS'][:, indices].astype(float).copy()
    points[..., 2] *= float(data['width'])
    points = mp_to_rig(points)
    centered = points - points.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=(1, 2))
    valid = ((data['FACE_LANDMARKS_confidence'][:, indices] > 0).all(axis=1)
             & np.isfinite(points).all(axis=(1, 2)) & (scale > 1e-6))
    observed = int(valid.sum())
    points = np.nan_to_num(centered/np.maximum(scale[:, None, None], 1e-8))
    points, valid = smooth_valid(points, valid)
    calibration = reference_frames(valid, float(data['fps']))
    residuals = []
    if len(calibration):
        neutral = np.median(points[calibration], axis=0)
        for i in np.flatnonzero(valid):
            try:
                rotation, residual = rigid_rotation(neutral, points[i])
            except ValueError:
                valid[i] = False
                continue
            if residual > 0.2 or rotation_angle(rotation) > np.deg2rad(80):
                valid[i] = False
                continue
            rotations[i] = rotation
            residuals.append(residual)
    return rotations, valid, {'observed_frames': observed, 'usable_frames': int(valid.sum()),
                              'neutral_frames_zero_based': calibration.tolist(),
                              'max_fit_residual': max(residuals, default=None),
                              'method': 'rigid upper-face alignment relative to initial neutral interval'}
