"""Match a visible inner-lip gap / mouth-width ratio on the supported avatar.

The target is an image measurement, not metric 3D ground truth. Mesh anchors
were checked on the Rocketbox Male_Adult_01 neutral mesh and jaw-open shape.
"""
import hashlib
import numpy as np

PROFILE = {
    'name': 'Rocketbox Male_Adult_01', 'vertex_count': 4025,
    'topology': 'f2fbce7b54ee54e15ae83ed454fc6169c1f69582c11f18f8551250f919821c6e',
    # Inner upper lip, inner lower lip, screen-left and screen-right mouth corners.
    'vertices': [3250, 3249, 2750, 3127],
}
LIP_CHANNELS = ('mouthLowerDownLeft', 'mouthLowerDownRight',
                'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthClose')


def mesh_mouth_ratio(mesh, ids):
    """Measure actual shape-key geometry at current weights, before skinning."""
    keys = mesh.data.shape_keys.key_blocks
    matrix = mesh.matrix_world.to_3x3()
    points = []
    for index in ids:
        point = keys[0].data[index].co.copy()
        for key in list(keys)[1:]:
            if key.value:
                point += key.value * (key.data[index].co - key.relative_key.data[index].co)
        points.append(matrix @ point)
    width = points[3].x - points[2].x
    return max(0., points[0].z - points[1].z) / max(width, 1e-8)


def source_mouth_ratio(data):
    count = len(data['FACE_values'])
    ratio, valid = np.zeros(count), np.zeros(count, dtype=bool)
    key = 'FACE_LANDMARKS'
    if key not in data or key + '_names' not in data:
        return ratio, valid, 'missing_face_landmarks'
    names = list(data[key + '_names'])
    wanted = ['13', '14', '61', '291', '33', '263', '10', '152']
    if any(name not in names for name in wanted):
        return ratio, valid, 'missing_lip_or_orientation_landmarks'
    ids = [names.index(name) for name in wanted]
    points = np.asarray(data[key][:, ids], dtype=float).copy()
    confidence = data[key + '_confidence'][:, ids]
    # Pose export stores x/y in pixels, face z in units of image width.
    points[..., 2] *= float(data['width'])
    corners = points[:, 3, :2] - points[:, 2, :2]
    width = np.linalg.norm(corners, axis=1)
    up = np.column_stack((corners[:, 1], -corners[:, 0])) / np.maximum(width[:, None], 1e-8)
    gap = np.abs(np.sum((points[:, 0, :2] - points[:, 1, :2]) * up, axis=1))
    ratio = gap / np.maximum(width, 1e-8)
    horizontal = points[:, 5] - points[:, 4]
    vertical = points[:, 7] - points[:, 6]
    normal = np.cross(horizontal, vertical)
    facing = np.abs(normal[:, 2]) / np.maximum(np.linalg.norm(normal, axis=1), 1e-8)
    face_valid = data['FACE_valid'] if 'FACE_valid' in data else np.any(data['FACE_values'] > 0, axis=1)
    valid = (face_valid & (confidence > .5).all(axis=1) & np.isfinite(points).all(axis=(1, 2))
             & (width >= 4) & (facing >= .75) & np.isfinite(ratio) & (ratio <= 1.2))
    ratio[~valid] = 0
    # Subpixel landmark jitter should not turn a closed mouth into an open one.
    ratio[ratio < .02] = 0
    return ratio, valid, 'measured'


def fit_mouth(values, names, gap_base, width_base, gap_delta, width_delta, target, valid):
    """Bounded fit of gap - target*width = 0; keep all unrelated channels intact.

First use jaw range, then minimal changes in lip opening/closing channels.
No per-clip peak normalization: a quiet clip must not become a maximum opening.
"""
    result = np.array(values, dtype=float, copy=True)
    before_gap = gap_base + values @ gap_delta
    before_width = width_base + values @ width_delta
    before = np.maximum(0, before_gap) / np.maximum(before_width, 1e-8)
    jaw = names.index('jawOpen')
    lips = [names.index(name) for name in LIP_CHANNELS if name in names]
    for frame in np.flatnonzero(valid):
        a = (gap_delta - target[frame] * width_delta) / width_base
        b = (target[frame] * width_base - gap_base) / width_base
        row = result[frame]
        if abs(a[jaw]) > 1e-8:
            row[jaw] = np.clip(row[jaw] + (b - a @ row) / a[jaw], 0, 1)
        # Project onto the remaining scalar constraint with box constraints.
        active = list(lips)
        for _ in range(len(lips) + 1):
            residual = b - a @ row
            if abs(residual) < 1e-7 or not active:
                break
            aa = a[active]
            denominator = aa @ aa
            if denominator < 1e-12:
                break
            proposed = row[active] + residual * aa / denominator
            row[active] = np.clip(proposed, 0, 1)
            active = [index for index, value in zip(active, proposed) if 0 < value < 1]
    after_width = width_base + result @ width_delta
    after = np.maximum(0, gap_base + result @ gap_delta) / np.maximum(after_width, 1e-8)
    # A malformed shape combination must not collapse mouth width or worsen the measurement.
    revert = valid & ((after_width < width_base * .2) |
                      (np.abs(after - target) > np.abs(before - target) + 1e-6))
    result[revert] = values[revert]
    after[revert] = before[revert]
    return result.astype(np.float32), before, after


def calibrate_mouth(meshes, mapping, data, values, mode='auto'):
    info = {'status': 'disabled' if mode == 'off' else 'unavailable'}
    if mode == 'off':
        return values, info
    names = list(data['FACE_names'])
    if 'jawOpen' not in names:
        return values, {**info, 'reason': 'missing_jawOpen'}
    target, valid, reason = source_mouth_ratio(data)
    if not valid.any():
        return values, {**info, 'reason': reason if reason != 'measured' else 'no_reliable_frontal_mouth_frames'}
    selected = None
    for mesh in meshes:
        if len(mesh.data.vertices) != PROFILE['vertex_count'] or mesh.data.shape_keys is None:
            continue
        topology = np.array([i for polygon in mesh.data.polygons
                             for i in (len(polygon.vertices), *polygon.vertices)], dtype='<i4')
        if hashlib.sha256(topology.tobytes()).hexdigest() == PROFILE['topology']:
            selected = mesh
            break
    if selected is None:
        return values, {**info, 'reason': 'unsupported_avatar_topology'}
    keys = selected.data.shape_keys.key_blocks
    ids = PROFILE['vertices']
    # Use rest mesh coordinates transformed into canonical world axes, never an animated pose.
    matrix = selected.matrix_world.to_3x3()
    base = np.array([matrix @ keys[0].data[i].co for i in ids])
    gap_base, width_base = base[0, 2] - base[1, 2], base[3, 0] - base[2, 0]
    if width_base <= 1e-8 or gap_base < 0:
        return values, {**info, 'reason': 'unsupported_avatar_orientation'}
    gap_delta, width_delta = np.zeros(len(names)), np.zeros(len(names))
    for item in mapping:
        if item['mesh'] != selected.name:
            continue
        key = keys[item['key']]
        delta = np.array([matrix @ (key.data[i].co - key.relative_key.data[i].co) for i in ids])
        index = names.index(item['channel'])
        gap_delta[index] += delta[0, 2] - delta[1, 2]
        width_delta[index] += delta[3, 0] - delta[2, 0]
    if gap_delta[names.index('jawOpen')] <= width_base * .01:
        return values, {**info, 'reason': 'insufficient_avatar_jaw_response'}
    calibrated, before, after = fit_mouth(values, names, gap_base, width_base,
                                         gap_delta, width_delta, target, valid)
    valid_ids = np.flatnonzero(valid)
    sample_ids = sorted({int(valid_ids[0]), int(valid_ids[np.argmax(target[valid])]),
                         int(valid_ids[np.argmin(target[valid])])})
    info = {'status': 'calibrated', 'profile': PROFILE['name'], 'mesh': selected.name,
            'anchor_vertices_upper_lower_left_right': ids,
            'measured_frames': int(valid.sum()), 'fallback_frames': int((~valid).sum()),
            'mean_ratio_error_before': float(np.abs(before[valid] - target[valid]).mean()),
            'mean_ratio_error_after': float(np.abs(after[valid] - target[valid]).mean()),
            'unmatched_frames': int(np.sum(valid & (np.abs(after - target) > .02))),
            'closed_frames': int(np.sum(valid & (target == 0))),
            'closed_frame_max_ratio': float(after[valid & (target == 0)].max(initial=0)),
            'samples': [{'frame': i + 1, 'target_ratio': float(target[i]),
                         'before_ratio': float(before[i]), 'after_ratio': float(after[i])} for i in sample_ids],
            'note': 'Frontal image lip-gap/width fit in rest mesh; not a ground-truth 3D or phoneme accuracy test.'}
    return calibrated, info
