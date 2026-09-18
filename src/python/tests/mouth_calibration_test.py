"""Physical bounds, closure, timing, and scale/roll invariance of mouth fitting."""
import unittest
import numpy as np
from pose_format.animation.mouth_calibration import source_mouth_ratio, fit_mouth, calibrate_mouth


class MouthCalibrationTest(unittest.TestCase):
    def source(self):
        # inner lips, corners, outer eyes, forehead, chin; x/y pixels, z normalized.
        points = np.array([[0,-5,0],[0,5,0],[-20,0,0],[20,0,0],
                           [-40,-30,0],[40,-30,0],[0,-60,0],[0,40,0]], dtype=float)
        return {'FACE_LANDMARKS': np.tile(points, (4,1,1)),
                'FACE_LANDMARKS_names': np.array(['13','14','61','291','33','263','10','152']),
                'FACE_LANDMARKS_confidence': np.ones((4,8)), 'width': np.array(640),
                'FACE_values': np.ones((4,1)), 'FACE_valid': np.ones(4,dtype=bool)}

    def test_ratio_is_scale_translation_and_roll_invariant(self):
        data = self.source()
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        data['FACE_LANDMARKS'][1,:,:2] = data['FACE_LANDMARKS'][1,:,:2] @ rotation.T
        data['FACE_LANDMARKS'][2,:,:2] *= 3
        data['FACE_LANDMARKS'][3,:,:2] += [500,200]
        ratio, valid, _ = source_mouth_ratio(data)
        np.testing.assert_allclose(ratio, .25)
        self.assertTrue(valid.all())

    def test_missing_sideways_and_unreliable_frames_do_not_drive_mouth(self):
        data = self.source()
        data['FACE_valid'][0] = False
        data['FACE_LANDMARKS_confidence'][1,0] = 0
        data['FACE_LANDMARKS'][2,4,2] = -.2
        data['FACE_LANDMARKS'][2,5,2] = .2
        data['FACE_LANDMARKS'][3,0,0] = np.nan
        ratio, valid, _ = source_mouth_ratio(data)
        self.assertFalse(valid.any())
        np.testing.assert_equal(ratio, 0)

    def test_closed_lips_and_missing_landmarks(self):
        data = self.source()
        data['FACE_LANDMARKS'][:,0:2,1] = 0
        ratio, valid, _ = source_mouth_ratio(data)
        self.assertTrue(valid.all())
        np.testing.assert_equal(ratio, 0)
        del data['FACE_LANDMARKS']
        self.assertFalse(source_mouth_ratio(data)[1].any())

    def test_fit_uses_lips_when_jaw_range_is_exhausted_and_preserves_other_channels(self):
        names = ['jawOpen','mouthLowerDownLeft','mouthLowerDownRight','mouthClose','eyeBlinkLeft']
        weights = np.array([[.3,0,0,0,.7], [.2,0,0,0,.2], [.3,0,0,0,.8]])
        gap = np.array([.35,.1,.1,-.35,0])
        target = np.array([.5,0,.8])
        out, before, after = fit_mouth(weights,names,.005,1,gap,np.zeros(5),target,np.ones(3,dtype=bool))
        np.testing.assert_allclose(after[:2], target[:2], atol=1e-6)
        self.assertAlmostEqual(out[0,0],1)
        self.assertGreater(out[0,1],0)
        self.assertAlmostEqual(out[0,1],out[0,2])
        self.assertTrue(((out>=0)&(out<=1)).all())
        np.testing.assert_allclose(out[:,4],weights[:,4])
        self.assertLess(abs(after[2]-target[2]),abs(before[2]-target[2]))

    def test_current_mouth_width_and_all_existing_shapes_affect_fit(self):
        names = ['jawOpen','mouthSmileLeft']
        values = np.array([[.1,.8]])
        # Smile doubles neutral width at full weight; must not use a fixed neutral width.
        out, _, after = fit_mouth(values,names,.01,1,np.array([.5,.05]),np.array([0,1]),
                                 np.array([.2]),np.array([True]))
        self.assertAlmostEqual(float(after[0]),.2,places=6)
        self.assertAlmostEqual(float(out[0,1]),.8,places=6)

    def test_invalid_frames_are_unchanged_and_small_motion_is_not_peak_normalized(self):
        values = np.array([[.3],[.2]])
        out, _, after = fit_mouth(values,['jawOpen'],0,1,np.array([.5]),np.zeros(1),
                                 np.array([.02,1]),np.array([True,False]))
        self.assertAlmostEqual(float(out[0,0]),.04,places=6)
        self.assertEqual(out[1,0],np.float32(.2))
        self.assertAlmostEqual(float(after[0]),.02,places=6)

    def test_disabled_or_unsupported_avatar_keeps_original_weights(self):
        data = self.source()
        data['FACE_names'] = np.array(['jawOpen'])
        values = data['FACE_values']
        for mode, expected in [('off','disabled'), ('auto','unavailable')]:
            out, report = calibrate_mouth([], [], data, values, mode)
            self.assertIs(out, values)
            self.assertEqual(report['status'], expected)
        self.assertEqual(report['reason'], 'unsupported_avatar_topology')


if __name__ == '__main__':
    unittest.main()
