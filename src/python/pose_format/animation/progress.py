"""Small stdout protocol usable by Blender without installing tqdm there."""
import json
import time

PREFIX = 'POSE_PROGRESS '
_last_stage = None
_last_time = 0.0


def report(stage, completed, total, unit='frame'):
    global _last_stage, _last_time
    now = time.monotonic()
    if stage != _last_stage or completed in (0, total) or now - _last_time >= 0.1:
        print('\n' + PREFIX + json.dumps({'stage': stage, 'completed': completed,
                                          'total': total, 'unit': unit}), flush=True)
        _last_stage, _last_time = stage, now


def parse(line):
    if PREFIX not in line:
        return None
    try:
        event = json.loads(line.partition(PREFIX)[2])
        if (isinstance(event['stage'], str) and isinstance(event['completed'], int)
                and isinstance(event['total'], int) and 0 <= event['completed'] <= event['total']
                and isinstance(event['unit'], str)):
            return event
    except (ValueError, KeyError, TypeError):
        pass
    return None
