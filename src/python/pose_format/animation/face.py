"""Aligned MediaPipe blendshapes and explicit Rocketbox/ARKit shape-key mapping.

Pure NumPy helpers also run in Blender's Python. No guessed facial bone rotations.
"""
import json
import re
from pathlib import Path

import numpy as np
if __package__:
    from .mouth_calibration import calibrate_mouth
else:
    from mouth_calibration import calibrate_mouth


def find_extras(source):
    source = Path(source)
    direct = Path(str(source) + '.extras.jsonl')
    if direct.is_file():
        return direct
    if source.stem.lower().endswith('_filtered'):
        original = source.with_name(source.stem[:-9] + '.pose.extras.jsonl')
        if original.is_file():
            return original
    return None


def load_face(path, frames, fps):
    """Require one timestamped record per pose frame, including missing detections."""
    records, names = [], set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= frames or row.get('frame_index') != index:
                raise ValueError(f'Face sidecar frame mismatch at line {index + 1}: {path}')
            stamp = row.get('timestamp_ms')
            if not isinstance(stamp, (int, float)) or not np.isfinite(stamp) or abs(stamp - index * 1000 / fps) > 1.1:
                raise ValueError(f'Face sidecar timestamp mismatch at frame {index}: {path}')
            values = row.get('face_blendshapes', {})
            if not isinstance(values, dict):
                raise ValueError(f'Invalid face blendshapes at frame {index}')
            for name, value in values.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid face coefficient {name} at frame {index}')
            values = {name: value for name, value in values.items() if name != '_neutral'}
            names.update(values)
            records.append(values)
    if len(records) != frames:
        raise ValueError(f'Face sidecar has {len(records)} frames, pose has {frames}: {path}')
    names = sorted(names)
    values = np.zeros((frames, len(names)), dtype=np.float32)
    valid = np.array([bool(record) for record in records])
    for frame, record in enumerate(records):
        values[frame] = [record.get(name, 0.) for name in names]
    # Interpolate only bounded gaps <=100ms. Longer gaps stay neutral; never hold a stale expression.
    ids = np.flatnonzero(valid)
    available = valid.copy()
    filled = 0
    for a, b in zip(ids[:-1], ids[1:]):
        if 1 < b - a <= max(1, int(round(fps * .1))) + 1:
            for frame in range(a + 1, b):
                values[frame] = values[a] + (values[b] - values[a]) * ((frame - a) / (b - a))
                available[frame] = True
                filled += 1
    return {'FACE_names': np.array(names, dtype=str), 'FACE_values': values, 'FACE_valid': available}, {
        'status': 'ready' if valid.any() else 'no_detections', 'source': str(Path(path).resolve()),
        'detected_frames': int(valid.sum()), 'interpolated_frames': filled,
        'channels': names, 'missing_policy': 'Interpolate bounded gaps <=100ms; otherwise neutral.',
        'smoothing': 'None: preserve blink timing and lip movement.'}


def channel_for_key(key, channels):
    # Rocketbox's AK_01_BrowDownLeft -> browDownLeft; also accept exact ARKit names.
    clean = re.sub(r'^AK_\d+_', '', key)
    matches = {name.lower(): name for name in channels}
    return matches.get(clean.lower()) if clean.lower() != '_neutral' else None


def animate_face(meshes, data, progress, mouth_calibration='auto'):
    """Bulk-write curves, avoiding quadratic insertion on long videos."""
    if 'FACE_values' not in data:
        return {'status': 'inactive', 'mapping': []}
    names = list(data['FACE_names'])
    values = data['FACE_values']
    mapping, seen = [], set()
    for mesh in meshes:
        keys = mesh.data.shape_keys
        if keys is None or keys.as_pointer() in seen:
            continue
        seen.add(keys.as_pointer())
        for key in keys.key_blocks:
            channel = channel_for_key(key.name, names)
            if channel is not None:
                mapping.append({'mesh': mesh.name, 'key': key.name, 'channel': str(channel)})
    if names and not mapping:
        raise ValueError('Face data exists but the FBX has no matching ARKit/Rocketbox shape keys; use --face-animation off to omit it')
    by_name = {mesh.name: mesh for mesh in meshes}
    progress('Calibrating mouth', 0, 1, 'step')
    values, mouth_info = calibrate_mouth(meshes, mapping, data, values, mouth_calibration)
    progress('Calibrating mouth', 1, 1, 'step')
    progress('Animating face', 0, len(mapping), 'channel')
    for index, item in enumerate(mapping):
        keys = by_name[item['mesh']].data.shape_keys
        key = keys.key_blocks[item['key']]
        signal = values[:, names.index(item['channel'])]
        key.value = float(signal[0])
        key.keyframe_insert(data_path='value', frame=1)
        action = keys.animation_data.action
        if hasattr(action, 'layers') and len(action.layers):
            curves = [curve for layer in action.layers for strip in layer.strips
                      for bag in strip.channelbags for curve in bag.fcurves]
        else:
            curves = list(action.fcurves)
        curve = next(curve for curve in curves if curve.data_path == key.path_from_id('value'))
        curve.keyframe_points.add(len(signal) - 1)
        xy = np.column_stack((np.arange(1, len(signal) + 1), signal)).astype(np.float32)
        curve.keyframe_points.foreach_set('co', xy.ravel())
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
        curve.update()
        item['peak_frame'] = int(np.argmax(signal)) + 1
        item['min_frame'] = int(np.argmin(signal)) + 1
        item['range'] = [float(signal.min()), float(signal.max())]
        progress('Animating face', index + 1, len(mapping), 'channel')
    used = {item['channel'] for item in mapping}
    return {'status': 'animated' if mapping else 'no_detections', 'mapping': mapping,
            'mouth_calibration': mouth_info,
            'unmapped_channels': sorted(set(names) - used),
            'note': 'Blendshape weights with optional landmark-based mouth calibration; no tongue inference or phoneme generation.'}


def sample_face(objects, mapping, frames, scene):
    """Sample shape weights by exported mesh/key names, failing on dropped keys."""
    result = []
    for frame in frames:
        scene.frame_set(frame)
        result.append([objects[item['mesh']].data.shape_keys.key_blocks[item['key']].value
                       for item in mapping])
    return np.asarray(result)
