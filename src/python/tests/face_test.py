"""Prevent facial/body timing drift, incorrect pairing and stale expressions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pose_format.animation.face import find_extras, load_face, channel_for_key
from pose_format.animation.animate_poses import fingerprint, make_job


class FaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_rows(self, weights, fps=30):
        path = self.root / 'sample.pose.extras.jsonl'
        rows = [{'frame_index': i, 'timestamp_ms': round(i * 1000 / fps),
                 'face_blendshapes': value} for i, value in enumerate(weights)]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return path

    def test_pairing_and_resume_include_sidecar(self):
        source, model = self.root / 'sample_filtered.pose', self.root / 'model.fbx'
        source.touch()
        model.touch()
        job = make_job(source, model)
        before = fingerprint(job, None)
        sidecar = self.write_rows([{'jawOpen': .5}])
        self.assertEqual(find_extras(source), sidecar)
        self.assertNotEqual(before, fingerprint(job, None))
        job['face_animation'] = 'off'
        disabled = fingerprint(job, None)
        self.write_rows([{'jawOpen': .8}, {'jawOpen': .2}])
        self.assertEqual(disabled, fingerprint(job, None))
        kept = self.root / 'clip.mp4.pose.extras.jsonl'
        kept.touch()
        self.assertEqual(find_extras(self.root / 'clip.mp4_filtered.pose'), kept)
        self.assertIsNone(find_extras(self.root / 'other_filtered.pose'))

    def test_weights_and_short_gap_preserve_timing(self):
        path = self.write_rows([{'jawOpen': .2, '_neutral': .7}, {}, {'jawOpen': .8}])
        arrays, report = load_face(path, 3, 30)
        self.assertEqual(list(arrays['FACE_names']), ['jawOpen'])
        np.testing.assert_allclose(arrays['FACE_values'][:, 0], [.2, .5, .8])
        self.assertEqual(report['detected_frames'], 2)
        self.assertEqual(report['interpolated_frames'], 1)

    def test_long_gaps_and_edges_are_neutral(self):
        path = self.write_rows([{}, {'eyeBlinkLeft': 1}, {}, {}, {}, {}, {'eyeBlinkLeft': .8}, {}])
        arrays, _ = load_face(path, 8, 30)
        np.testing.assert_allclose(arrays['FACE_values'][:, 0], [0, 1, 0, 0, 0, 0, .8, 0])

    def test_empty_detections(self):
        arrays, report = load_face(self.write_rows([{}, {}]), 2, 30)
        self.assertEqual(arrays['FACE_values'].shape, (2, 0))
        self.assertEqual(report['status'], 'no_detections')

    def test_reject_misaligned_or_corrupt_data(self):
        path = self.write_rows([{'jawOpen': .2}] * 4)
        for count, fps in [(3, 30), (5, 30), (4, 25)]:
            with self.subTest(count=count, fps=fps), self.assertRaises(ValueError):
                load_face(path, count, fps)
        original = path.read_text()
        for corrupt in [original.replace('"frame_index": 1', '"frame_index": 0'),
                        original.replace('0.2', 'NaN'), original.replace('0.2', '1.2')]:
            path.write_text(corrupt)
            with self.assertRaises(ValueError):
                load_face(path, 4, 30)

    def test_mapping_keeps_anatomical_sides_and_avoids_other_shape_sets(self):
        channels = ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen']
        self.assertEqual(channel_for_key('AK_09_EyeBlinkLeft', channels), 'eyeBlinkLeft')
        self.assertEqual(channel_for_key('AK_10_EyeBlinkRight', channels), 'eyeBlinkRight')
        self.assertEqual(channel_for_key('jawOpen', channels), 'jawOpen')
        for key in ['AU_26_JawDrop', 'AA_VI_10_aa', 'AK_52_TongueOut', '_Neutral']:
            self.assertIsNone(channel_for_key(key, channels))


if __name__ == '__main__':
    unittest.main()
