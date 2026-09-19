"""Directory filtering preserves originals and resumes without inference."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.estimation.mediapipe_tasks import holistic_header
from pose_format.bin.directory import find_videos_with_missing_pose_files, process_video
from pose_format.bin.filter_poses import filtered_pose_path, write_filtered_pose
from pose_format.animation.face import find_extras


class DirectoryFilterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        header = holistic_header(320, 240)
        points = sum(len(c.points) for c in header.components)
        self.pose = Pose(header, NumPyPoseBody(30, np.ones((2, 1, points, 3), dtype=np.float32),
                                             np.ones((2, 1, points), dtype=np.float32)))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            self.pose.write(stream)

    def check_filtered(self, source):
        original = Pose.read(source.read_bytes())
        result = Pose.read(filtered_pose_path(source).read_bytes())
        np.testing.assert_array_equal(original.body.confidence, 1)
        expected = original.body.confidence.copy()
        offset = 0
        for component in original.header.components:
            for i, name in enumerate(component.points):
                if (component.name in ('POSE_LANDMARKS', 'POSE_WORLD_LANDMARKS')
                        and name.split('_', 1)[-1] in ('KNEE', 'ANKLE', 'HEEL', 'FOOT_INDEX')):
                    expected[..., offset+i] = 0
            offset += len(component.points)
        np.testing.assert_array_equal(result.body.confidence, expected)
        self.assertEqual(result.body.fps, original.body.fps)
        self.assertEqual(result.body.data.shape, original.body.data.shape)

    def test_fresh_estimation_filters_in_memory_and_preserves_original_and_extras(self):
        video = self.root/'sample.mp4'
        video.touch()
        source = video.with_suffix('.pose')
        extras = Path(str(source)+'.extras.jsonl')

        def estimate(*args, **kwargs):
            self.save(Path(args[1]))
            extras.write_text('{"face_blendshapes": {"jawOpen": 0.4}}\n')
            return self.pose

        with patch('pose_format.bin.directory.pose_video', side_effect=estimate) as estimator, \
                patch('pose_format.bin.filter_poses.Pose.read', side_effect=AssertionError('Unnecessary disk read')):
            self.assertTrue(process_video(False, 'mediapipe', {}, video, filter_output=True, progress=False))
        estimator.assert_called_once()
        self.check_filtered(source)
        self.assertEqual(find_extras(filtered_pose_path(source)), extras)
        self.assertIn('jawOpen', extras.read_text())

    def test_default_does_not_filter(self):
        video = self.root/'sample.mp4'
        source = video.with_suffix('.pose')
        self.save(source)
        with patch('pose_format.bin.directory.pose_video') as estimator:
            self.assertTrue(process_video(False, 'mediapipe', {}, video, progress=False))
        estimator.assert_not_called()
        self.assertFalse(filtered_pose_path(source).exists())

    def test_resume_cli_parallel_with_output_directory_and_no_model(self):
        videos, output = self.root/'videos', self.root/'output'
        for name in ('one', 'two'):
            video = videos/name/'clip.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
            self.save(output/name/'clip.mp4.pose')
        pending = find_videos_with_missing_pose_files(videos, recursive=True, keep_video_suffixes=True,
                                                    output_directory=output, filter_output=True)
        self.assertEqual(len(pending), 2)
        command = [sys.executable, '-m', 'pose_format.bin.directory', '-d', str(videos), '--recursive',
                   '--keep-video-suffixes', '--output-directory', str(output), '--filter', '--no-progress',
                   '--num-workers', '2', '--backend', 'mediapipe-tasks', '--model', str(self.root/'absent.task')]
        first = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
        for name in ('one', 'two'):
            self.check_filtered(output/name/'clip.mp4.pose')
        files = list(output.rglob('*.pose'))
        before = {p:p.stat().st_mtime_ns for p in files}
        second = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 0, second.stdout+second.stderr)
        self.assertIn('Found 0 videos', second.stdout)
        self.assertEqual(before, {p:p.stat().st_mtime_ns for p in files})

    def test_failed_write_leaves_original_retryable(self):
        source = self.root/'sample.pose'
        self.save(source)
        before = source.read_bytes()
        with patch.object(Pose, 'write', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                write_filtered_pose(source)
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(filtered_pose_path(source).exists())
        self.assertFalse(list(self.root.glob('.filter-*')))
        write_filtered_pose(source)
        self.check_filtered(source)


if __name__ == '__main__':
    unittest.main()
