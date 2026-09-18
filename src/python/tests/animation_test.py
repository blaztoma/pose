"""Batch discovery, video pairing, resumability, and failure isolation."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch

from pose_format.animation import animate_poses as cli


class AnimationBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def touch(self, relative, content=b'input'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_recursive_filtered_only(self):
        first = self.touch('a_filtered.pose')
        second = self.touch('nested/b_filtered.POSE')
        self.touch('nested/b.pose')
        self.touch('nested/b_filtered_preview.mp4')
        self.assertEqual(cli.find_poses(self.root), [first, second])
        self.assertEqual(cli.find_poses(self.root, False), [first])
        self.assertEqual(cli.find_poses(second), [second])

    def test_pairs_original_without_using_generated_video(self):
        source = self.touch('lt_filtered.pose')
        self.touch('lt_overlay.mp4')
        self.touch('lt_filtered_preview.mp4')
        self.assertIsNone(cli.find_video(source))
        video = self.touch('lt.mp4')
        self.assertEqual(cli.find_video(source), video)
        self.touch('lt.webm')
        with self.assertRaises(ValueError):
            cli.find_video(source)

    def test_keep_video_suffix(self):
        source = self.touch('lt.webm_filtered.pose')
        video = self.touch('lt.webm')
        self.touch('lt.mp4')
        self.assertEqual(cli.find_video(source), video)

    def test_nearest_reference_and_explicit_model(self):
        source = self.touch('nested/lt_filtered.pose')
        parent_model = self.touch('reference_model/Export/' + cli.MODEL_NAME)
        self.assertEqual(cli.find_model(source), parent_model)
        nearer = self.touch('nested/reference_model/Export/' + cli.MODEL_NAME)
        self.assertEqual(cli.find_model(source), nearer)
        self.assertEqual(cli.find_model(source, parent_model), parent_model)

    def test_outputs_do_not_collide_or_overwrite_input(self):
        source = self.touch('a_filtered.pose')
        other = self.touch('b_filtered.pose')
        model = self.touch('model.fbx')
        a, b = cli.make_job(source, model), cli.make_job(other, model)
        self.assertEqual(Path(a['fbx']).name, 'a_filtered_animated.fbx')
        for key in ('data', 'report', 'validation', 'blend', 'fbx', 'preview', 'comparison'):
            self.assertNotEqual(a[key], b[key])
            self.assertNotEqual(Path(a[key]), source)

    def test_completion_requires_matching_signature_and_nonempty_artifacts(self):
        output = self.touch('result.mp4', b'abc')
        marker = self.touch('completed.json', json.dumps(
            {'fingerprint': 'abc', 'sizes': {str(output): 3}}).encode())
        self.assertTrue(cli.is_complete(marker, 'abc', [output]))
        self.assertFalse(cli.is_complete(marker, 'different', [output]))
        output.write_bytes(b'')
        self.assertFalse(cli.is_complete(marker, 'abc', [output]))
        marker.write_text('broken')
        self.assertFalse(cli.is_complete(marker, 'abc', [output]))

    def test_failure_does_not_stop_next_pose(self):
        first = self.touch('a_filtered.pose')
        second = self.touch('b_filtered.pose')
        with patch('sys.argv', ['animate_poses', '-i', str(self.root)]), \
                patch.object(cli, 'find_executable', return_value='blender'), \
                patch.object(cli, 'animate_one', side_effect=[ValueError('bad pose'), 'created']) as run:
            with self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual([c.args[0] for c in run.call_args_list], [first, second])

    def test_dry_run_does_not_launch_or_write(self):
        self.touch('a_filtered.pose')
        with patch('sys.argv', ['animate_poses', '-i', str(self.root), '--dry-run']), \
                patch.object(cli, 'animate_one') as run, patch.object(cli, 'find_executable') as tool:
            cli.main()
            run.assert_not_called()
            tool.assert_not_called()
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_child_progress_is_consumed_before_process_finishes(self):
        signal = self.root / 'continue'
        event = {'stage': 'Animating', 'completed': 1, 'total': 3, 'unit': 'frame'}
        code = (
            'import pathlib, time\n'
            f'print({("POSE_PROGRESS " + json.dumps(event))!r}, flush=True)\n'
            f'signal = pathlib.Path({str(signal)!r})\n'
            'for _ in range(500):\n'
            '    if signal.exists(): break\n'
            '    time.sleep(0.01)\n'
            'else: raise RuntimeError("progress was buffered until exit")\n'
            'print("child completed", flush=True)\n')
        progress = Mock()
        progress.update.side_effect = lambda **unused: signal.write_text('continue')
        log = self.root / 'live.log'
        cli.run_logged([sys.executable, '-u', '-c', code], log, progress=progress)
        progress.update.assert_called_once_with(**event)
        self.assertIn('child completed', log.read_text())

    def test_failed_child_keeps_log_and_partial_progress(self):
        event = {'stage': 'Animating', 'completed': 1, 'total': 3, 'unit': 'frame'}
        code = f'import sys; print({("POSE_PROGRESS " + json.dumps(event))!r}, flush=True); sys.exit(3)'
        progress = Mock()
        log = self.root / 'failed.log'
        with self.assertRaisesRegex(RuntimeError, r'Command failed \(3\).*failed.log'):
            cli.run_logged([sys.executable, '-u', '-c', code], log, progress=progress)
        progress.update.assert_called_once_with(**event)
        self.assertIn('POSE_PROGRESS', log.read_text())


if __name__ == '__main__':
    unittest.main()
