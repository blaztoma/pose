"""Backend compatibility, failure handling, and Tasks landmark conversion."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.bin.directory import find_videos_with_missing_pose_files, get_corresponding_pose_path
from pose_format.bin.pose_estimation import pose_video
from pose_format.estimation.base import create_estimator
from pose_format.estimation.mediapipe_tasks import COMPONENTS, holistic_header, result_arrays, result_extras, validate_runtime


class EstimationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def result(self):
        point = SimpleNamespace(x=.25, y=.5, z=-.1, visibility=.75)
        return SimpleNamespace(**{name: [point] * count for name, count, _ in COMPONENTS},
                               left_hand_world_landmarks=[point] * 21,
                               right_hand_world_landmarks=[], face_blendshapes=[])

    def test_conversion_roundtrip_preserves_animation_coordinates(self):
        data, scores = result_arrays(self.result(), 320, 240)
        self.assertEqual(data.shape, (586, 3))
        np.testing.assert_allclose(data[-1], [80, 120, -.1])
        np.testing.assert_allclose(data[-1] / [320, 240, 1], [.25, .5, -.1])
        self.assertAlmostEqual(scores[-1], .75)
        self.assertEqual(scores[33], 1)
        pose = Pose(holistic_header(320, 240), NumPyPoseBody(25, data[None, None], scores[None, None]))
        buffer = io.BytesIO()
        pose.write(buffer)
        restored = Pose.read(buffer.getvalue())
        np.testing.assert_array_equal(restored.body.confidence[0, 0], scores)
        np.testing.assert_allclose(restored.body.data[0, 0], data)
        self.assertEqual(restored.header.components[2].points[0], 'WRIST')

    def test_missing_hand_is_zero_confidence_and_keeps_indices(self):
        result = self.result()
        result.left_hand_landmarks = []
        data, scores = result_arrays(result, 320, 240)
        np.testing.assert_array_equal(scores[511:532], 0)
        np.testing.assert_array_equal(data[511:532], 0)
        np.testing.assert_array_equal(scores[532:553], 1)

    def test_wrong_landmark_count_rejected(self):
        result = self.result()
        result.face_landmarks = result.face_landmarks[:468]
        with self.assertRaisesRegex(ValueError, '478'):
            result_arrays(result, 320, 240)

    def test_extras_keep_native_metric_hand_coordinates(self):
        row = result_extras(self.result(), 4, 160)
        self.assertEqual(row['left_hand_world_landmarks'][0], [.25, .5, -.1])
        self.assertEqual(row['right_hand_world_landmarks'], [])
        self.assertEqual(row['timestamp_ms'], 160)
        json.dumps(row, allow_nan=False)

    def test_gpu_rejected_before_video_read_on_windows(self):
        with patch('pose_format.estimation.mediapipe_tasks.platform.system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'Windows'):
                pose_video('missing.mp4', str(self.root / 'test.pose'), 'mediapipe',
                           backend='mediapipe-tasks', device='gpu')
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'only supports'):
            create_estimator(device='gpu')

    def test_tasks_rejects_legacy_options_and_parallel_frame_tracking(self):
        with self.assertRaisesRegex(ValueError, 'model_complexity'):
            create_estimator('mediapipe-tasks', additional_config={'model_complexity': 2})
        with self.assertRaisesRegex(ValueError, 'workers 1'):
            create_estimator('mediapipe-tasks', pose_workers=2)

    def test_old_tasks_runtime_rejected_before_video_read(self):
        with patch('pose_format.estimation.mediapipe_tasks.version', return_value='0.10.21'), \
             patch('pose_format.bin.pose_estimation.video_metadata') as read_metadata:
            with self.assertRaisesRegex(ValueError, r'found 0\.10\.21.*\.venv-tasks'):
                pose_video('input.mp4', str(self.root / 'out.pose'), 'mediapipe', backend='mediapipe-tasks')
            read_metadata.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_tasks_runtime_accepts_supported_version(self):
        with patch('pose_format.estimation.mediapipe_tasks.version', return_value='1.0.1'):
            validate_runtime()

    def test_tasks_runtime_check_keeps_legacy_available(self):
        with patch('pose_format.estimation.mediapipe_tasks.version', return_value='0.10.21'):
            create_estimator('mediapipe-legacy')

    def test_failed_estimation_preserves_existing_output_and_cleans_staging(self):
        output = self.root / 'test.pose'
        output.write_bytes(b'original')
        estimator = SimpleNamespace(estimate=lambda *a, **kw: (_ for _ in ()).throw(ValueError('failed')))
        with patch('pose_format.bin.pose_estimation.create_estimator', return_value=estimator), \
             patch('pose_format.bin.pose_estimation.video_metadata', return_value=SimpleNamespace(width=320, height=240, fps=25)), \
             patch('pose_format.bin.pose_estimation.read_frames_exact', return_value=iter([])):
            with self.assertRaisesRegex(ValueError, 'failed'):
                pose_video('input.mp4', str(output), 'mediapipe')
        self.assertEqual(output.read_bytes(), b'original')
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_separate_batch_outputs_and_resume(self):
        source = self.root / 'input'
        source.mkdir()
        nested = source / 'nested'
        nested.mkdir()
        video = nested / 'lt.MP4'
        video.touch()
        video.with_suffix('.pose').touch()
        output = self.root / 'tasks'
        self.assertEqual(find_videos_with_missing_pose_files(source, recursive=True), [])
        self.assertEqual(find_videos_with_missing_pose_files(source, recursive=True, output_directory=output), [video])
        destination = get_corresponding_pose_path(video, False, source, output)
        self.assertEqual(destination, output / 'nested' / 'lt.pose')
        destination.parent.mkdir(parents=True)
        destination.touch()
        self.assertEqual(find_videos_with_missing_pose_files(source, recursive=True, output_directory=output), [])

    def test_entrypoints_do_not_import_mediapipe_eagerly(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; import pose_format.bin.directory; '
                        'assert "mediapipe" not in sys.modules; '
                        'assert "pose_format.utils.holistic" not in sys.modules'], check=True)

    def test_overlay_finds_source_for_separate_and_filtered_outputs(self):
        from pose_format.bin.pose_visualizer import find_background_video
        video = self.root / 'original.mp4'
        video.touch()
        output = self.root / 'tasks' / 'original.pose'
        output.parent.mkdir()
        output.touch()
        output.with_name(output.name + '.meta.json').write_text(json.dumps({'source': str(video)}))
        self.assertEqual(find_background_video(output), video)
        filtered = output.with_name('original_filtered.pose')
        filtered.touch()
        self.assertEqual(find_background_video(filtered), video)


if __name__ == '__main__':
    unittest.main()
