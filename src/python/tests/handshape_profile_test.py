"""Palm-local transfer must preserve shape under changes of signer and camera."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_format.animation.handshape_profile import (
    extract_template, load_profile, local_directions, palm_basis, validate_profile, weight,
)
from pose_format.animation import animate_poses


def hand():
    points = np.zeros((21, 3))
    for i in range(5):
        points[1+4*i:5+4*i] = [
            [1, 2-i, 0], [2, 2-i, .1], [2.5, 2-i, -.5], [2.2, 2-i, -1],
        ]
    return points


def profile():
    return dict(schema_version=1, kind='palm_local_finger_directions', side='RIGHT',
                directions=local_directions(hand()).tolist(), strength=[[.2,0],[.4,1],[.8,1],[1,0]])


class HandshapeTest(unittest.TestCase):
    def test_shape_is_invariant_under_translation_scale_and_rotation(self):
        p = hand()
        angle = .73
        rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        moved = (p @ rotation.T)*2.7 + [5,-4,9]
        np.testing.assert_allclose(local_directions(p), local_directions(moved), atol=1e-12)
        basis = palm_basis(moved)
        self.assertAlmostEqual(np.linalg.det(basis), 1)
        np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-12)

    def test_extraction_restores_legacy_depth_units_and_rejects_missing_frames(self):
        p = hand()
        encoded = p.copy()
        encoded[:,2] /= 352
        data = dict(RIGHT_HAND_LANDMARKS=np.stack([encoded]*3),
                    RIGHT_HAND_LANDMARKS_confidence=np.ones((3,21)), width=352, fps=25)
        directions, summary = extract_template(data, 'RIGHT', [0,1,2])
        # Axis conversion is a rigid rotation, hence palm-local vectors agree.
        np.testing.assert_allclose(directions, local_directions(p), atol=1e-12)
        self.assertEqual(summary['seconds'], [0,.04,.08])
        data['RIGHT_HAND_LANDMARKS_confidence'][1,4] = 0
        with self.assertRaisesRegex(ValueError, 'missing hands'):
            extract_template(data, 'RIGHT', [0,1])

    def test_fade_is_bounded_and_zero_outside_selected_interval(self):
        p = validate_profile(profile())
        for t in (-1,0,.2,1,2):
            self.assertEqual(weight(p,t), 0)
        self.assertAlmostEqual(weight(p,.3), .5)
        self.assertAlmostEqual(weight(p,.9), .5)
        self.assertEqual(weight(p,.6), 1)
        for t in np.linspace(0,2,100):
            self.assertTrue(0 <= weight(p,t) <= 1)

    def test_invalid_directions_and_intervals_fail(self):
        for keys in ([[0,1],[1,1],[2,0]], [[0,0],[1,1],[1,0]], [[0,0],[1,2],[2,0]]):
            p = profile()
            p['strength'] = keys
            with self.assertRaises(ValueError):
                validate_profile(p)
        p = profile()
        p['directions'][0][0] = [0,0,0]
        with self.assertRaises(ValueError):
            validate_profile(p)
        with self.assertRaises(ValueError):
            local_directions(np.zeros((21,3)))

    def test_profile_binds_to_recording_and_invalidates_animation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root/'AS_filtered.pose'
            video = root/'AS.mp4'
            model = root/'model.fbx'
            for path in (pose,video,model):
                path.write_bytes(b'original')
            self.assertIsNone(load_profile(pose))
            p = profile()
            p.update(target_video=video.name,
                     target_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                     target_pose_sha256=hashlib.sha256(pose.read_bytes()).hexdigest())
            path = root/'handshape_profile.json'
            path.write_text(json.dumps(p), encoding='utf-8')
            job = animate_poses.make_job(pose, model)
            self.assertEqual(job['handshape_profile'], p)
            first = animate_poses.fingerprint(job)
            changed = copy.deepcopy(job)
            changed['handshape_profile']['strength'][1][0] = .5
            self.assertNotEqual(first, animate_poses.fingerprint(changed))
            pose.write_bytes(b'new pose extraction')
            with self.assertRaisesRegex(ValueError, 'target changed'):
                load_profile(pose)
            p['target_video'] = '../AS.mp4'
            path.write_text(json.dumps(p), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'adjacent'):
                load_profile(pose)


if __name__ == '__main__':
    unittest.main()
