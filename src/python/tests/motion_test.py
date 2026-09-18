"""Synthetic rotations test observable motion separately from noisy video."""
import unittest
import numpy as np

from pose_format.animation.motion import (
    HEAD_ANCHORS, head_motion, rigid_rotation, shoulder_basis,
    smooth_valid, torso_motion,
)


def rz(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


class UpperBodyMotionTest(unittest.TestCase):
    def test_rigid_fit_recovers_rotation_despite_translation_and_scale(self):
        points = np.random.default_rng(42).normal(size=(12, 3))
        rotation = rz(27)
        found, residual = rigid_rotation(points, 2.7*(points @ rotation.T) + [8, 9, 10])
        np.testing.assert_allclose(found, rotation, atol=1e-10)
        self.assertLess(residual, 1e-10)
        self.assertAlmostEqual(np.linalg.det(found), 1)

    def test_degenerate_face_is_rejected(self):
        with self.assertRaises(ValueError):
            rigid_rotation(np.zeros((12, 3)), np.zeros((12, 3)))
        with self.assertRaises(ValueError):
            rigid_rotation(np.array([[i, 0, 0] for i in range(12)]),
                           np.array([[i, 0, 0] for i in range(12)]))

    def test_torso_uses_shoulders_not_masked_hips(self):
        names = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
        body = np.zeros((20, 4, 3))
        body[:, 0, 1] = 0.2
        body[:, 1, 1] = -0.2
        target = shoulder_basis(rz(20) @ np.array([0, np.cos(.1), np.sin(.1)]))
        body[10:, :2] = body[10:, :2] @ target.T
        body[10:, :2] += [3, 4, 5]
        body[:, 2:] = np.nan  # Invalid hips must not affect rotation.
        confidence = np.ones((20, 4))
        confidence[:, 2:] = 0
        rotations, valid, _ = torso_motion(body, confidence, names, 25)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rotations[-1], target, atol=1e-10)
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-10)

    def test_face_motion_ignores_translation_and_person_size(self):
        face = np.random.default_rng(42).normal(size=(len(HEAD_ANCHORS), 3)) * 12
        rotation = rz(17)
        rig = np.repeat(face[None], 20, axis=0)
        rig[10:] = 1.8 * (rig[10:] @ rotation.T) + [9, 10, 11]
        raw = np.stack((rig[..., 1], -rig[..., 2], -rig[..., 0]/320), axis=-1)
        data = {'POSE_WORLD_LANDMARKS': np.zeros((20, 33, 3)), 'FACE_LANDMARKS': raw,
                'FACE_LANDMARKS_names': np.array(HEAD_ANCHORS), 'width': 320,
                'FACE_LANDMARKS_confidence': np.ones((20, len(HEAD_ANCHORS))), 'fps': 25}
        rotations, valid, _ = head_motion(data)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(rotations[-1], rotation, atol=1e-10)

    def test_missing_or_collapsed_face_does_not_invent_motion(self):
        data = {'POSE_WORLD_LANDMARKS': np.zeros((20, 33, 3))}
        rotations, valid, _ = head_motion(data)
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(rotations, np.repeat(np.eye(3)[None], 20, axis=0))
        data.update(FACE_LANDMARKS=np.zeros((20, 12, 3)),
                    FACE_LANDMARKS_names=np.array(HEAD_ANCHORS),
                    FACE_LANDMARKS_confidence=np.ones((20, 12)), width=320, fps=25)
        _, valid, _ = head_motion(data)
        self.assertFalse(valid.any())

    def test_long_gaps_remain_invalid(self):
        points = np.arange(20)[:, None].astype(float)
        valid = np.ones(20, dtype=bool)
        valid[3:5] = False
        valid[8:16] = False
        _, result = smooth_valid(points, valid)
        self.assertTrue(result[3:5].all())
        self.assertFalse(result[8:16].any())


if __name__ == '__main__':
    unittest.main()
