"""Batch discovery, video pairing, resumability, and failure isolation."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

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


class AnimationStagesTest(unittest.TestCase):
    """Exercise real job/marker logic with Blender's outputs supplied by a stub."""
    touch = AnimationBatchTest.touch

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.touch('clip_filtered.pose')
        self.args = SimpleNamespace(model=self.touch('model.fbx'), embed_textures=False,
                                    overwrite=False, ffmpeg=None, no_progress=True, render=False)
        self.calls = []
        self.fail_on = None
        self.summary = {'frames': 8, 'fps': 30, 'face': {'status': 'missing_sidecar'}}

        def prepare(source, data, **kwargs):
            data.write_bytes(b'data')
            data.with_suffix('.json').write_text(json.dumps(self.summary))
            return self.summary

        def run(command, log, **kwargs):
            job = json.loads((log.parent/'job.json').read_text())
            stage = Path(command[command.index('--python')+1]).name if '--python' in command else 'comparison'
            self.calls.append(stage)
            if stage == self.fail_on:
                raise RuntimeError('stage failed')
            keys = {'animate_pose.py': ('fbx', 'blend', 'report'),
                    'verify_animation.py': ('validation',), 'render_preview.py': ('preview',),
                    'comparison': ('comparison',)}[stage]
            for key in keys:
                Path(job[key]).write_bytes(b'output')
            if stage == 'animate_pose.py':
                (log.parent/'ik_targets.npz').write_bytes(b'targets')

        def start_patch(name, **kwargs):
            patcher = patch.object(cli, name, **kwargs)
            value = patcher.start()
            self.addCleanup(patcher.stop)
            return value

        self.prepare = start_patch('prepare_pose', side_effect=prepare)
        start_patch('run_logged', side_effect=run)
        self.tools = start_patch('find_executable', return_value='ffmpeg')

    def execute(self):
        return cli.animate_one(self.source, self.args, 'blender')

    def test_default_builds_and_validates_without_video_or_ffmpeg(self):
        with patch.object(cli, 'find_video', side_effect=AssertionError('video not needed')):
            self.assertEqual(self.execute(), 'created')
            self.assertEqual(self.execute(), 'skipped')
        self.assertEqual(self.calls, ['animate_pose.py', 'verify_animation.py'])
        self.tools.assert_not_called()

    def test_enable_render_later_and_repeat_without_rebuilding(self):
        self.touch('clip.mp4')
        self.execute()
        self.calls.clear()
        self.args.render = True
        self.assertEqual(self.execute(), 'created')
        self.assertEqual(self.calls, ['render_preview.py', 'comparison'])
        self.calls.clear()
        self.assertEqual(self.execute(), 'skipped')
        self.args.render = False
        self.assertEqual(self.execute(), 'skipped')
        self.assertEqual(self.calls, [])
        self.prepare.assert_called_once()

    def test_render_failure_retains_animation_and_retry_only_renders(self):
        self.args.render = True
        self.fail_on = 'render_preview.py'
        with self.assertRaisesRegex(RuntimeError, 'stage failed'):
            self.execute()
        self.calls.clear()
        self.fail_on = None
        self.assertEqual(self.execute(), 'created')
        self.assertEqual(self.calls, ['render_preview.py'])
        self.prepare.assert_called_once()

    def test_validation_failure_does_not_publish_animation_completion(self):
        self.fail_on = 'verify_animation.py'
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertFalse(list(self.root.rglob('*completed.json')))
        self.fail_on = None
        self.calls.clear()
        self.execute()
        self.assertEqual(self.calls, ['animate_pose.py', 'verify_animation.py'])

    def test_missing_preview_only_renders_and_overwrite_rebuilds(self):
        self.args.render = True
        self.execute()
        (self.root/'clip_filtered_preview.mp4').unlink()
        self.calls.clear()
        self.execute()
        self.assertEqual(self.calls, ['render_preview.py'])
        self.args.overwrite = True
        self.calls.clear()
        self.execute()
        self.assertEqual(self.calls, ['animate_pose.py', 'verify_animation.py', 'render_preview.py'])

    def test_video_changes_only_invalidate_render(self):
        video = self.touch('clip.mp4')
        self.args.render = True
        self.execute()
        video.write_bytes(b'changed video')
        self.calls.clear()
        self.execute()
        self.assertEqual(self.calls, ['render_preview.py', 'comparison'])

    def test_render_script_changes_do_not_invalidate_animation(self):
        scripts = self.root/'scripts'
        self.touch('scripts/animate_pose.py')
        render = self.touch('scripts/render_preview.py')
        job = cli.make_job(self.source, self.args.model)
        Path(job['blend']).write_bytes(b'blend')
        with patch.object(cli, 'SCRIPTS', scripts):
            before = cli.fingerprint(job)
            preview_before = cli.fingerprint(job, stage='render')
            render.write_bytes(b'new render code')
            self.assertEqual(before, cli.fingerprint(job))
            self.assertNotEqual(preview_before, cli.fingerprint(job, stage='render'))


if __name__ == '__main__':
    unittest.main()
