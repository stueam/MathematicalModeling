"""Bounded local work: incremental dispatch, cancellation and process-group cleanup."""
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import subprocess
import time
import traceback

from .experiment import dump


def _worker(function, incoming, outgoing):
    while True:
        item = incoming.get()
        if item is None:
            return
        index, job = item
        try:
            outgoing.put((index, function(job), None))
        except Exception:
            outgoing.put((index, None, traceback.format_exc()))


def bounded_map(function, jobs, workers, dispatch_deadline, hard_deadline, consume):
    """Keep at most workers jobs in flight; no long, uncancellable work queue."""
    if workers < 1:
        raise ValueError('workers must be positive')
    ctx = multiprocessing.get_context('spawn')
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(function, incoming, outgoing))
                 for _ in range(min(workers, len(jobs)))]
    next_index, active, completed = 0, 0, 0
    try:
        for process in processes:
            process.start()
        while completed < len(jobs):
            now = time.monotonic()
            if now >= hard_deadline:
                raise TimeoutError('Round hard deadline')
            while active < len(processes) and next_index < len(jobs) and time.monotonic() < dispatch_deadline:
                incoming.put((next_index, jobs[next_index]))
                next_index += 1
                active += 1
            if active == 0:
                break
            try:
                index, result, error = outgoing.get(timeout=.2)
            except queue.Empty:
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError('Work process exited before delivering its result')
                continue
            active -= 1
            completed += 1
            if error:
                raise RuntimeError(f'Job {index} failed:\n{error}')
            consume(result)
        return {'planned': len(jobs), 'dispatched': next_index, 'completed': completed,
                'status': 'complete' if completed == len(jobs) else 'dispatch_cutoff'}
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=.3)
            if process.is_alive():
                process.kill()
                process.join(timeout=.3)
        incoming.close()
        outgoing.close()


def live_group_members(group):
    """Linux verification of running descendants, excluding reaped/dead zombies."""
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def supervise(command, out, hard_limit_s=1800., dispatch_limit_s=1620., env=None, cwd=None):
    """Own a fresh session, enforce the 30 minute ceiling, cancel all descendants.

    STOP in the output directory or SIGINT/SIGTERM immediately cancels the round.
    Shorter limits are supported for regression tests; longer limits are refused.
    """
    if not 0 < dispatch_limit_s < hard_limit_s <= 1800:
        raise ValueError('Require 0 < dispatch limit < hard limit <= 1800 seconds')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    interrupted = []
    previous = {}

    def cancel(number, frame):
        interrupted.append(number)

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, cancel)
    child_env = dict(os.environ if env is None else env)
    child_env.update(Q4_ROUND_DISPATCH_DEADLINE=str(started + dispatch_limit_s),
                     Q4_ROUND_HARD_DEADLINE=str(started + hard_limit_s - min(5., hard_limit_s / 10)))
    child = None
    status, error = 'running', None
    try:
        child = subprocess.Popen(command, env=child_env, cwd=cwd, start_new_session=True)
        dump(out / 'process.json', {'supervisor_pid': os.getpid(), 'worker_pgid': child.pid,
                                   'hard_limit_s': hard_limit_s, 'dispatch_limit_s': dispatch_limit_s})
        while child.poll() is None:
            if interrupted or (out / 'STOP').exists():
                status = 'interrupted_by_user'
                break
            # Reserve up to two seconds to kill/reap and save the control record.
            if time.monotonic() - started >= hard_limit_s - min(2., hard_limit_s / 5):
                status = 'hard_deadline'
                break
            time.sleep(.1)
        else:
            status = 'complete' if child.returncode == 0 else 'failed'
    except BaseException as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Under host CPU/I/O load a killed process can take longer to reap.
            # Never let TimeoutExpired skip the control record or cleanup audit.
            reap_deadline = min(time.monotonic() + 1.5, started + hard_limit_s - .05)
            while child.poll() is None and time.monotonic() < reap_deadline:
                try:
                    child.wait(timeout=min(.05, max(.001, reap_deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            members = live_group_members(child.pid)
            cleanup_deadline = min(time.monotonic() + .5, started + hard_limit_s - .05)
            while members and time.monotonic() < cleanup_deadline:
                time.sleep(.01)
                members = live_group_members(child.pid)
        else:
            members = []
        if members:
            status, error = 'cleanup_failed', f'Live group members: {members}'
        result = {'status': status, 'real_time_s': time.monotonic() - started,
                  'hard_limit_s': hard_limit_s, 'dispatch_limit_s': dispatch_limit_s,
                  'returncode': child.returncode if child else None,
                  'live_worker_pids_after_shutdown': members, 'error': error,
                  'automatic_next_round': False, 'local_simulator_only': True}
        dump(out / 'control.json', result)
        for number, handler in previous.items():
            signal.signal(number, handler)
    return result
