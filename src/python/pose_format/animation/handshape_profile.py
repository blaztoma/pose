"""Explicit finger directions in a palm frame, independent of signer size/pose.

Profiles annotate a particular recording; they are not inferred observations.
The original .pose and its confidence values are never rewritten.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def palm_basis(points):
    """Columns: wrist->middle MCP, towards index MCP, their right-handed normal."""
    p = np.asarray(points, dtype=float)
    x = p[9] - p[0]
    across = p[5] - p[17]
    if not np.isfinite(p).all() or min(np.linalg.norm(x), np.linalg.norm(across)) < 1e-8:
        raise ValueError('Degenerate palm')
    x /= np.linalg.norm(x)
    z = np.cross(x, across / np.linalg.norm(across))
    if np.linalg.norm(z) < .1:
        raise ValueError('Nearly collinear palm anchors')
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


def local_directions(points):
    p = np.asarray(points, dtype=float)
    frame = palm_basis(p)
    directions = np.stack([np.diff(p[1+4*i:5+4*i], axis=0) for i in range(5)])
    lengths = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError('Degenerate finger segment')
    return (directions / lengths) @ frame


def extract_template(data, side, frames):
    """Read prepared legacy-schema arrays, correct Z scale, then remove palm pose."""
    key = side + '_HAND_LANDMARKS'
    frames = list(frames)
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError('Select at least two distinct reviewed frames')
    if min(frames) < 0 or max(frames) >= len(data[key]):
        raise ValueError('Donor frames outside recording')
    points = data[key][frames].astype(float).copy()
    if not (data[key + '_confidence'][frames] > 0).all():
        raise ValueError('Donor selection includes missing hands')
    points[..., 2] *= float(data['width'])
    # Same axis transform as animate_pose.mp_to_rig; it has positive determinant.
    points = np.stack((-points[..., 2], points[..., 0], -points[..., 1]), axis=-1)
    samples = np.stack([local_directions(p) for p in points])
    template = np.median(samples, axis=0)
    norms = np.linalg.norm(template, axis=-1, keepdims=True)
    if np.any(norms < .5):
        raise ValueError('Inconsistent donor finger directions')
    template /= norms
    deviations = np.degrees(np.arccos(np.clip(np.sum(samples*template, axis=-1), -1, 1)))
    return template, dict(frames_zero_based=frames,
        seconds=[f / float(data['fps']) for f in frames],
        max_direction_deviation_degrees=float(deviations.max()),
        median_direction_deviation_degrees=float(np.median(deviations)))


def validate_profile(profile):
    if (profile.get('schema_version') != 1 or profile.get('kind') != 'palm_local_finger_directions'
            or profile.get('side') not in ('LEFT','RIGHT')):
        raise ValueError('Unsupported handshape profile')
    values = np.asarray(profile['directions'], dtype=float)
    if values.shape != (5,3,3) or not np.isfinite(values).all():
        raise ValueError('Expected finite 5 x 3 x 3 finger directions')
    if not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-5):
        raise ValueError('Finger directions must be unit vectors')
    keys = np.asarray(profile['strength'], dtype=float)
    if (keys.ndim != 2 or keys.shape[1] != 2 or len(keys) < 3 or not np.isfinite(keys).all()
            or (keys[:,0] < 0).any() or (np.diff(keys[:,0]) <= 0).any()
            or (keys[:,1] < 0).any() or (keys[:,1] > 1).any()
            or keys[0,1] != 0 or keys[-1,1] != 0):
        raise ValueError('Handshape needs increasing times and a bounded 0..1 fade')
    return profile


def weight(profile, time):
    keys = profile['strength']
    if time <= keys[0][0] or time >= keys[-1][0]:
        return 0.
    for (a,x),(b,y) in zip(keys,keys[1:]):
        if time <= b:
            t = (time-a)/(b-a)
            return x+(y-x)*t*t*(3-2*t)
    return 0.


def load_profile(source):
    source = Path(source)
    path = source.parent/'handshape_profile.json'
    if not path.is_file():
        return None
    profile = validate_profile(json.loads(path.read_text(encoding='utf-8')))
    video = (path.parent/profile['target_video']).resolve()
    if video.parent != path.parent.resolve():
        raise ValueError('Handshape target video must be adjacent')
    stem = source.stem.removesuffix('_filtered').removesuffix('_30fps')
    if stem != video.stem:
        raise ValueError('Handshape profile belongs to another recording')
    for file, expected in ((source,profile['target_pose_sha256']),
                           (video,profile['target_video_sha256'])):
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Handshape target changed; review profile: '+str(file))
    return profile
