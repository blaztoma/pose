"""Recursively animate *_filtered.pose files using Blender and a Rocketbox FBX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from contextlib import ExitStack
from tqdm import tqdm

from .prepare_pose import prepare_pose
from .face import find_extras
from .progress import parse as parse_progress

SCRIPTS = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}
MODEL_NAME = 'Male_Adult_01_facial.fbx'


def find_poses(path, recursive=True):
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = path.rglob('*') if recursive else path.iterdir()
    else:
        raise ValueError(f'Input does not exist: {path}')
    return sorted(p.resolve() for p in candidates
                  if p.is_file() and p.suffix.lower() == '.pose' and p.stem.lower().endswith('_filtered'))


def find_video(source):
    stem = source.stem[:-len('_filtered')]
    exact = source.with_name(stem)
    if exact.suffix.lower() in VIDEO_SUFFIXES and exact.is_file():
        return exact
    matches = sorted(p for p in source.parent.iterdir()
                     if p.is_file() and p.stem == stem and p.suffix.lower() in VIDEO_SUFFIXES)
    if len(matches) > 1:
        raise ValueError(f'Multiple original videos match {source}: {matches}')
    return matches[0] if matches else None


def find_model(source, explicit=None):
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f'Reference FBX does not exist: {explicit}')
        return explicit.resolve()
    for parent in dict.fromkeys([*source.parents, *SCRIPTS.parents]):
        for relative in (Path('reference_model/Export') / MODEL_NAME,
                         Path('models/rocketbox/Male_Adult_01/Export') / MODEL_NAME):
            candidate = parent / relative
            if candidate.is_file():
                return candidate.resolve()
    raise ValueError(f'No Rocketbox reference model for {source}; specify --model PATH.fbx')


def find_executable(name, explicit=None):
    if explicit:
        result = shutil.which(str(explicit))
        if result:
            return result
        raise ValueError(f'{name} executable not found: {explicit}')
    result = shutil.which(name)
    if result:
        return result
    if name == 'blender' and os.name == 'nt':
        base = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Blender Foundation'
        candidates = sorted(base.glob('Blender */blender.exe'),
                            key=lambda p: tuple(int(v) for v in re.findall(r'\d+', p.parent.name)), reverse=True)
        if candidates:
            return str(candidates[0])
    raise ValueError(f'{name} was not found; specify --{name} PATH')


def make_job(source, model, embed_textures=False):
    work = source.parent / '.animation' / source.stem
    prefix = source.parent / source.stem
    return {'name': source.stem, 'source': str(source), 'model': str(model),
            'data': str(work / 'pose_data.npz'), 'report': str(work / 'retarget_report.json'),
            'validation': str(work / 'validation.json'),
            'blend': str(prefix) + '_animated.blend', 'fbx': str(prefix) + '_animated.fbx',
            'preview': str(prefix) + '_preview.mp4', 'comparison': str(prefix) + '_comparison.mp4',
            'embed_textures': embed_textures}


def fingerprint(job, video):
    """Invalidate completion markers when input, model, textures or code changes."""
    files = [Path(job['source']), Path(job['model']), *sorted(SCRIPTS.glob('*.py'))]
    if job.get('face_animation', 'auto') != 'off':
        extras = find_extras(Path(job['source']))
        if extras is not None:
            files.append(extras)
    model = Path(job['model'])
    for folder in (model.parent.parent / 'Textures', model.parent / 'Textures'):
        if folder.is_dir():
            files.extend(sorted(p for p in folder.rglob('*') if p.is_file()))
    if video:
        files.append(video)
    info = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps([job, info], sort_keys=True).encode()).hexdigest()


def is_complete(marker, signature, outputs):
    try:
        state = json.loads(marker.read_text(encoding='utf-8'))
        return (state.get('fingerprint') == signature and
                all(p.is_file() and p.stat().st_size > 0 and
                    state.get('sizes', {}).get(str(p)) == p.stat().st_size for p in outputs))
    except (OSError, ValueError):
        return False


class StageProgress:
    """One live bar per animation, reset for each measured processing stage."""
    def __init__(self, name, enabled=True):
        self.name, self.enabled = name, enabled
        self.bar = None
        self.stage = None

    def update(self, stage, completed, total, unit='frame'):
        if not self.enabled:
            return
        if stage != self.stage:
            self.close()
            self.stage = stage
            self.bar = tqdm(total=total, desc=f'{self.name} | {stage}', unit=unit,
                            position=1, dynamic_ncols=True, leave=False)
        self.bar.update(max(0, completed - self.bar.n))
        if completed == total:
            self.bar.refresh()

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.stage = None


def run_logged(command, log, progress=None, ffmpeg_total=None):
    with log.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps([str(arg) for arg in command]) + '\n')
        stream.flush()
        with subprocess.Popen([str(arg) for arg in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    event = parse_progress(line)
                    if event is not None and progress is not None:
                        progress.update(**event)
                    elif ffmpeg_total is not None and progress is not None:
                        match = re.fullmatch(r'frame=\s*(\d+)\s*', line)
                        if match:
                            progress.update('Creating comparison', min(int(match[1]), ffmpeg_total), ffmpeg_total)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if returncode:
        raise RuntimeError(f'Command failed ({returncode}); see {log}')


def animate_one(source, args, blender):
    with ExitStack() as stack:
        progress = StageProgress(source.name, enabled=not getattr(args, 'no_progress', False))
        stack.callback(progress.close)
        return _animate_one(source, args, blender, progress)


def _animate_one(source, args, blender, progress):
    model = find_model(source, args.model)
    video = find_video(source)
    job = make_job(source, model, args.embed_textures)
    job['arm_solver'] = getattr(args, 'arm_solver', 'ik')
    job['face_animation'] = getattr(args, 'face_animation', 'auto')
    job['mouth_calibration'] = getattr(args, 'mouth_calibration', 'auto')
    job['blender'] = blender
    work = Path(job['data']).parent
    marker = work / 'completed.json'
    outputs = [Path(job[k]) for k in ('fbx', 'blend', 'preview', 'report', 'validation', 'data')]
    outputs.append(Path(job['data']).with_suffix('.json'))
    outputs.append(work / 'ik_targets.npz')
    if video:
        outputs.append(Path(job['comparison']))
    signature = fingerprint(job, video)
    if not args.overwrite and is_complete(marker, signature, outputs):
        tqdm.write(f'Skipping completed: {source}')
        return 'skipped'
    work.mkdir(parents=True, exist_ok=True)
    # An interrupted or failed rerun must never leave a valid completion marker.
    if marker.exists():
        marker.unlink()
    ffmpeg = find_executable('ffmpeg', args.ffmpeg) if video else None
    tqdm.write('  Preparing pose data...')
    progress.update('Preparing pose', 0, 1, 'step')
    summary = prepare_pose(source, Path(job['data']), face_animation=job['face_animation'])
    job['face_input'] = summary['face']
    tqdm.write('  Face animation: ' + summary['face']['status'])
    progress.update('Preparing pose', 1, 1, 'step')
    job_file = work / 'job.json'
    job_file.write_text(json.dumps(job, indent=2), encoding='utf-8')
    for script, label in [('animate_pose.py', 'Building FBX and Blender scene'),
                          ('verify_animation.py', 'Checking exported FBX'),
                          ('render_preview.py', 'Rendering preview')]:
        tqdm.write(f'  {label} ({summary["frames"]} frames, {summary["fps"]:g} FPS)...')
        progress.update(label + ' / starting', 0, 1, 'step')
        run_logged([blender, '--background', '--factory-startup', '--python-exit-code', '1',
                    '--python', SCRIPTS / script, '--', '--job', job_file], work / (Path(script).stem + '.log'),
                   progress=progress)
    if video:
        tqdm.write('  Creating comparison video...')
        progress.update('Creating comparison', 0, summary['frames'])
        fps = format(summary['fps'], '.12g')
        graph = (f'[0:v]setpts=PTS-STARTPTS,fps={fps},scale=640:480:force_original_aspect_ratio=decrease,'
                 'pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1[a];'
                 '[1:v]setpts=PTS-STARTPTS,setsar=1[b];[a][b]hstack=inputs=2:shortest=1[v]')
        run_logged([ffmpeg, '-y', '-v', 'error', '-nostats', '-progress', 'pipe:1', '-i', video, '-i', job['preview'],
                    '-filter_complex', graph, '-map', '[v]', '-an', '-r', fps, '-fps_mode', 'cfr',
                    '-c:v', 'libx264', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', job['comparison']], work / 'comparison.log',
                   progress=progress, ffmpeg_total=summary['frames'])
    else:
        tqdm.write('  No original video found; comparison omitted.')
    if not all(p.is_file() and p.stat().st_size > 0 for p in outputs):
        raise RuntimeError(f'Not all expected outputs were created; see {work}')
    marker.write_text(json.dumps({'fingerprint': signature,
                                  'sizes': {str(p): p.stat().st_size for p in outputs}}, indent=2), encoding='utf-8')
    return 'created'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True, type=Path, help='Directory or one *_filtered.pose file')
    recursion = parser.add_mutually_exclusive_group()
    recursion.add_argument('-r', '--recursive', dest='recursive', action='store_true', help='Include subdirectories (default)')
    recursion.add_argument('--no-recursive', dest='recursive', action='store_false', help='Only process the input directory')
    parser.set_defaults(recursive=True)
    parser.add_argument('--model', type=Path, help='Rocketbox FBX (otherwise find a local reference_model or project models folder)')
    parser.add_argument('--blender', help='Blender executable (otherwise auto-detected)')
    parser.add_argument('--ffmpeg', help='FFmpeg executable (otherwise PATH)')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate completed animations')
    parser.add_argument('--embed-textures', action='store_true', help='Embed textures in each FBX/Blender file (larger outputs)')
    parser.add_argument('--arm-solver', choices=['ik', 'rotation'], default='ik',
                        help='Automatic proportion calibration and wrist IK (default), or previous rotation mapping')
    parser.add_argument('--dry-run', action='store_true', help='List filtered poses without creating files')
    parser.add_argument('--face-animation', choices=['auto', 'off'], default='auto',
                        help='Animate ARKit/Rocketbox face shapes from matching .pose.extras.jsonl (default: auto)')
    parser.add_argument('--mouth-calibration', choices=['auto', 'off'], default='auto',
                        help='Fit mouth opening to face landmarks on the supported Rocketbox avatar (default: auto)')
    parser.add_argument('--no-progress', action='store_true', help='Disable animation and stage progress bars')
    args = parser.parse_args()
    try:
        sources = find_poses(args.input.resolve(), args.recursive)
        print(f'Found {len(sources)} filtered pose files.', flush=True)
        if not sources or args.dry_run:
            for source in sources:
                print(source)
            return
        blender = find_executable('blender', args.blender)
    except ValueError as error:
        parser.error(str(error))
    totals = {'created': 0, 'skipped': 0, 'failed': 0}
    for index, source in enumerate(tqdm(sources, desc='Animations', unit='pose', position=0,
                                        dynamic_ncols=True, disable=args.no_progress), 1):
        tqdm.write(f'[{index}/{len(sources)}] {source}')
        try:
            totals[animate_one(source, args, blender)] += 1
        except Exception as error:
            totals['failed'] += 1
            tqdm.write(f'FAILED {source}: {type(error).__name__}: {error}')
    print('Finished: ' + ', '.join(f'{key}={value}' for key, value in totals.items()), flush=True)
    if totals['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
