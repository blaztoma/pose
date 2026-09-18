"""Proportion invariance, absent observations and fixed-length IK geometry."""
import unittest
import numpy as np
from pose_format.animation.arm_ik import calibrate_arms, solve_two_bone


class ArmIKTest(unittest.TestCase):
    def observations(self, scale=1., shift=0.):
        names = [s + '_' + p for s in ('LEFT', 'RIGHT') for p in ('SHOULDER', 'ELBOW', 'WRIST')]
        points = np.array([[0,.2,0],[.1,.28,-.22],[.2,.03,-.08],
                           [0,-.2,0],[.1,-.28,-.22],[.2,-.03,-.08]])
        body = np.repeat((points * scale + shift)[None], 20, axis=0)
        image = np.stack((body[..., 1]*500+960, -body[..., 2]*500+300, body[..., 0]*0), axis=-1)
        confidence = np.ones((20, 6))
        data = dict(POSE_LANDMARKS=image, POSE_LANDMARKS_confidence=confidence,
                    POSE_LANDMARKS_names=names)
        avatar = {'shoulder_width': 40., 'LEFT': (32., 25.), 'RIGHT': (32., 25.)}
        return data, body, confidence, names, avatar

    def test_calibration_is_invariant_to_person_size_and_translation(self):
        first, report = calibrate_arms(*self.observations())
        second, _ = calibrate_arms(*self.observations(1.8, 2.7))
        for side in ('LEFT', 'RIGHT'):
            self.assertTrue(first[side][1].all())
            np.testing.assert_allclose(first[side][0], second[side][0], atol=1e-10)
            self.assertEqual(report['sides'][side]['status'], 'calibrated')
        # Shared scale preserves near-contact wrist spacing (not shoulder offsets).
        self.assertAlmostEqual(first['LEFT'][0][0, 1, 1] - first['RIGHT'][0][0, 1, 1], 6.)

    def test_missing_observations_choose_fallback(self):
        data, body, confidence, names, avatar = self.observations()
        confidence[:] = 0
        body[:] = np.nan
        targets, report = calibrate_arms(data, body, confidence, names, avatar)
        self.assertFalse(targets['LEFT'][1].any())
        self.assertIn('fallback', report['sides']['LEFT']['status'])

    def test_outlier_does_not_change_calibration(self):
        data, body, confidence, names, avatar = self.observations()
        baseline = calibrate_arms(data, body, confidence, names, avatar)[1]
        body[8] *= 100
        actual = calibrate_arms(data, body, confidence, names, avatar)[1]
        self.assertAlmostEqual(actual['source_shoulder_width_m'], baseline['source_shoulder_width_m'])
        self.assertAlmostEqual(actual['sides']['LEFT']['source_upper_arm_m'], baseline['sides']['LEFT']['source_upper_arm_m'])

    def test_reachable_target_and_pole_preserve_lengths(self):
        elbow, wrist, bend, clamped = solve_two_bone([0,0,0], [2,0,0], [1,1,0], 2, 1.5)
        np.testing.assert_allclose(wrist, [2,0,0])
        self.assertAlmostEqual(np.linalg.norm(elbow), 2)
        self.assertAlmostEqual(np.linalg.norm(wrist-elbow), 1.5)
        self.assertGreater(elbow[1], 0)
        self.assertEqual(clamped, 0)

    def test_unreachable_and_collapsed_targets_never_stretch(self):
        for target in ([9,0,0], [0,0,0], [.01,0,0]):
            elbow, wrist, _, error = solve_two_bone([0,0,0], target, [0,0,0], 2, 1.5)
            self.assertTrue(np.isfinite(elbow).all())
            self.assertAlmostEqual(np.linalg.norm(elbow), 2)
            self.assertAlmostEqual(np.linalg.norm(wrist-elbow), 1.5)
            self.assertGreater(error, 0)

    def test_invalid_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_two_bone([0,0,0], [1,0,0], [0,1,0], 0, 1)


if __name__ == '__main__':
    unittest.main()
