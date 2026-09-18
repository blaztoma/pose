"""Render multiprocessing frame progress in the parent process only."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
from queue import Empty

from tqdm import tqdm

_EVENTS = None


def _initialize_worker(events):
    global _EVENTS
    _EVENTS = events


def in_worker():
    return _EVENTS is not None


def send_event(kind, label, completed=None, total=None):
    _EVENTS.put((kind, label, completed, total))


def status(message):
    if in_worker():
        send_event('message', message)
    else:
        tqdm.write(message)


def parallel_results(func, videos, workers):
    """Yield results as videos finish, draining frame events while jobs run."""
    context = multiprocessing.get_context('spawn')
    events = context.Queue()
    bars = {}
    slots = set(range(1, workers + 1))  # Position zero is the overall video bar.

    def drain():
        while True:
            try:
                kind, label, completed, total = events.get_nowait()
            except Empty:
                return
            if kind == 'message':
                tqdm.write(label)
            elif kind == 'frame':
                if label not in bars:
                    position = min(slots)
                    slots.remove(position)
                    bars[label] = (tqdm(total=total, desc=label, unit='frame', position=position,
                                        dynamic_ncols=True, leave=False), position)
                bar, _ = bars[label]
                bar.update(completed - bar.n)
            elif kind == 'finished' and label in bars:
                bar, position = bars.pop(label)
                summary = str(bar)
                bar.close()
                slots.add(position)
                tqdm.write(summary)

    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_initialize_worker, initargs=(events,)) as pool:
            pending = {pool.submit(func, video) for video in videos}
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                drain()
                for future in done:
                    yield future.result()
        # Shutdown has joined the workers and flushed their queue feeder threads.
        drain()
    finally:
        for bar, _ in bars.values():
            bar.close()
        events.close()
        events.join_thread()
