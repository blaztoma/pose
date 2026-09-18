"""Render a saved animation directly to MP4 in Blender, without a PNG sequence."""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progress import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    job = json.loads(args.job.read_text(encoding='utf-8'))
    bpy.ops.wm.open_mainfile(filepath=job['blend'])
    scene = bpy.context.scene
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scene.render.ffmpeg.audio_codec = 'NONE'
    scene.render.filepath = job['preview']
    total = len(range(scene.frame_start, scene.frame_end + 1, scene.frame_step))
    completed = 0

    def frame_written(scene, *unused):
        nonlocal completed
        completed += 1
        report('Rendering preview', completed, total)

    report('Rendering preview', 0, total)
    bpy.app.handlers.render_write.append(frame_written)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.render_write.remove(frame_written)


if __name__ == '__main__':
    main()
