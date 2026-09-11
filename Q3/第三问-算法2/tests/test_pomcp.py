import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pomcp
import numpy as np
import pytest
from q3.core import Action, Belief, Observation, point_key
from q3.simulator import World, Source, LocalSimulator, generate_world
from q3.policy import Baseline
from pomcp.belief import ParticleBelief
from pomcp.world_sampler import ParticleWorldSampler
from pomcp.generative_model import MacroAction, simulate, environment
from pomcp.routing import length, insert, improve, OpenRoute
from pomcp.config import Config
from pomcp.planner import Planner
from pomcp.action_gen import MacroPolicy
from pomcp.heuristic import remaining_time
from pomcp.rollout import evaluate_world


def observed(seed=2, n=16):
    b = ParticleBelief(256, 17)
    world = generate_world(seed, n)
    env = LocalSimulator(world)
    for c in range(1, 21):
        a = Action('measure', (0., 0.), c)
        b.apply(a, env.execute(a, str(c)), str(c))
    return b, world


def test_macro_exact_cost_and_idempotence():
    b = ParticleBelief(128)
    env = LocalSimulator(World([Source(3, (300., 400.), 1000)]))
    macro = MacroAction('SHARED_PROBE', (Action('measure', (300., 400.), 1), Action('measure', (300., 400.), 2)))
    obs, dt = simulate(env, b, macro)
    assert dt == 111 and b.receiver == 2
    clear = MacroAction.atomic(Action('clear', (300., 400.), 3))
    assert simulate(env, b, clear)[1] == 5
    assert b.receiver == 2
    failed = MacroAction.atomic(Action('clear', (300., 400.), 4))
    assert simulate(env, b, failed)[1] == 3
    assert b.channels[4].status == 'unresolved'


@pytest.mark.parametrize('bearing', [0., 359.99, 90., 180.])
def test_particles_respect_rounding_and_fixed_radius(bearing):
    b = ParticleBelief(512)
    a = Action('measure', (0., 0.), 1)
    b.apply(a, {'accepted': True, 'virtual_time_s': 5, 'measure_result': 'direction', 'svd_deg': bearing}, 'a')
    cloud = b.cloud(1)
    phi = np.degrees(np.arctan2(cloud.xyz[:, 1], cloud.xyz[:, 0]))
    assert np.all(np.abs((phi-bearing+180)%360-180) <= 1.0050001)
    assert np.all(np.linalg.norm(cloud.xyz[:, :2], axis=1) <= cloud.xyz[:, 2])
    q = (-1400., 0.) if bearing < 90 else (1400., 0.)
    b.apply(Action('measure', q, 1), {'accepted': True, 'virtual_time_s': 300, 'measure_result': 'no_signal'}, 'b')
    cloud = b.cloud(1)
    assert np.all(np.linalg.norm(cloud.xyz[:, :2]-q, axis=1) > cloud.xyz[:, 2])


def test_joint_worlds_and_repeat_observations():
    b, _ = observed()
    for world in ParticleWorldSampler(12).sample(b, 20):
        assert 10 <= world.score()['source_count'] <= 16
        for c, p in b.channels.items():
            for o in p.history:
                response, _ = world.feedback(o.action)
                assert response['measure_result'] == o.result
                if o.bearing is not None:
                    assert response['svd_deg'] == o.bearing


def test_strict_absence_and_particle_depletion_not_certificate():
    b = ParticleBelief(128)
    b.clouds.clear()
    assert not b.done()
    for i, pos in enumerate(Baseline().stations):
        a = Action('measure', pos, 1)
        b.apply(a, {'accepted': True, 'virtual_time_s': 1000*i+5, 'measure_result': 'no_signal'}, str(i))
    assert b.status(1) == 'ABSENT'
    for c in range(2, 12):
        b.channels[c].status = 'cleared'
    assert not b.done()


def test_failed_clear_reconditions_particles_and_clone_isolation():
    b = ParticleBelief(512)
    a = Action('measure', (0., 0.), 1)
    b.apply(a, {'accepted': True, 'virtual_time_s': 5, 'measure_result': 'direction', 'svd_deg': 0.}, 'one')
    cloud = b.cloud(1)
    clone = b.clone()
    a = Action('clear', (500., 0.), 1)
    clone.apply(a, {'accepted': True, 'virtual_time_s': 108, 'clear_result': 'no_target_in_range'}, 'two')
    assert clone.channels[1].status == 'detected'
    assert np.all(np.linalg.norm(clone.cloud(1).xyz[:, :2]-a.position, axis=1) > 20)
    assert b.cloud(1) is cloud and b.steps == 1
    with pytest.raises(ValueError):
        clone.cloud(1).xyz[0, 0] = 10


def test_open_tour_does_not_return_and_insertion():
    start, route = (0., 0.), [(10., 0.), (20., 0.)]
    assert length(start, route) == 20
    assert length(start, insert(start, route, (15., 0.))) == 20
    assert length(start, improve(start, [(20., 0.), (10., 0.)])) == 20


def test_rollout_real_belief_untouched_and_terminal_tail():
    b, _ = observed()
    world = ParticleWorldSampler(1).sample(b, 1)[0]
    action = MacroAction.atomic(MacroPolicy().base.choose(b))
    old = [(p.status, tuple(p.history)) for p in b.channels.values()]
    row = evaluate_world((b, world, [action], 2, 'geometric', math.inf))[0]
    assert row['cost'] == row['prefix']+row['tail'] and row['tail'] > 0
    assert old == [(p.status, tuple(p.history)) for p in b.channels.values()]
    for p in b.channels.values():
        p.status = 'cleared'
    assert remaining_time(b) == 0


def test_candidates_preserve_baseline_and_action_families():
    b, _ = observed()
    p = MacroPolicy()
    raw = p.generate(b)
    selected = p.prune(b, raw, 12)
    assert selected[0].kind == 'BASELINE'
    assert raw[1] in selected
    for kind in ('CLEAR_TARGET', 'PROBE_TARGET', 'SHARED_PROBE', 'SEARCH_BLIND'):
        if any(a.kind == kind for a in raw):
            assert any(a.kind == kind for a in selected)


def test_parallel_reproducibility():
    b, _ = observed()
    selected, stages = [], []
    for workers in (1, 2):
        p = Planner(Config(particles=256, workers=workers, candidates=4, finalists=2,
                           coarse_worlds=2, fine_worlds=2, horizon=1, fine_horizon=1, budget_s=60))
        try:
            selected.append(p.choose(b.clone()))
            stages.append(p.records[-1]['stages'])
        finally:
            p.close()
    assert selected[0] == selected[1]
    assert stages[0] == stages[1]


def test_repeated_site_and_rejected_feedback_do_not_change_particles():
    b = ParticleBelief(128)
    a = Action('measure', (0., 0.), 1)
    r = {'accepted': True, 'virtual_time_s': 5, 'measure_result': 'direction', 'svd_deg': 0.}
    b.apply(a, r, 'first')
    cloud = b.cloud(1)
    b.apply(a, dict(r, virtual_time_s=10), 'repeat')
    assert b.cloud(1) is cloud and len(b.channels[1].history) == 1
    assert not b.apply(a, {'accepted': False, 'virtual_time_s': 0}, 'rejected')
    assert b.virtual_time == 10 and b.cloud(1) is cloud


def test_upper_bound_only_and_no_empty_unknown_shortcut():
    b = ParticleBelief(128)
    for c in range(1, 16):
        b.channels[c].status = 'cleared'
    assert not b.done()
    a = Action('measure', (0., 0.), 16)
    b.apply(a, {'accepted': True, 'virtual_time_s': 6, 'measure_result': 'near'}, 'sixteen')
    assert not b.done() and all(b.status(c) == 'ABSENT' for c in range(17, 21))
    b.apply(Action('clear', (0., 0.), 16), {'accepted': True, 'virtual_time_s': 11,
                                        'clear_result': 'success'}, 'clear')
    assert b.done()


def test_shaping_telescopes_without_extra_rewards():
    # gamma=1, Phi=-H, sum(-dt+Phi_next-Phi)= -sum(dt)+H0-Hleaf.
    h, dt = [100., 72., 19., 0.], [8., 20., 25.]
    shaped = sum(-d+h[i]-h[i+1] for i, d in enumerate(dt))
    assert shaped == -sum(dt)+h[0]


def test_partial_paired_batch_is_discarded(monkeypatch):
    import pomcp.planner as module
    b, _ = observed()
    calls = 0
    original = module.evaluate_world
    def interrupted(job):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('Injected slow world')
        return original(job)
    monkeypatch.setattr(module, 'evaluate_world', interrupted)
    planner = Planner(Config(workers=1, candidates=4, finalists=2, coarse_worlds=2,
                             fine_worlds=2, horizon=1, fine_horizon=1, tail='geometric', budget_s=60))
    try:
        selected = planner.choose(b)
        record = planner.records[-1]
        assert record['stages'] == [] and record['discarded_worlds'] == 2
        assert record['status'] == 'TimeoutError' and selected.kind == 'BASELINE'
    finally:
        planner.close()


def test_completion_tail_finishes_all_work():
    b, _ = observed()
    world = ParticleWorldSampler(901).sample(b, 1)[0]
    action = MacroAction.atomic(MacroPolicy().base.choose(b))
    row = evaluate_world((b, world, [action], 1, 'completion', math.inf))[0]
    assert row['terminal'] and row['tail'] > 0 and row['physical_steps'] > 20


def test_official_never_exits_unfinished(monkeypatch, tmp_path):
    import importlib.util
    from types import SimpleNamespace
    import q3.client
    spec = importlib.util.spec_from_file_location('algorithm_two_run', Path(__file__).resolve().parents[1]/'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class Client:
        def __init__(self, *args):
            self.log = []
            self.deadline = math.inf
        def enter(self):
            return 'enter', {'max_virtual_duration_s': 360000., 'virtual_time_s': 0.}
        def exit(self):
            pytest.fail('Unfinished belief must not call exit')
    monkeypatch.setattr(q3.client, 'HttpClient', Client)
    runner.official(SimpleNamespace(robot_id='test', base_url='unused', max_steps=0),
                    Config(workers=1), tmp_path)
    import json
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['entered'] and not summary['exited'] and not summary['certified_complete']


def test_worker_crash_uses_finite_completion(monkeypatch):
    import pomcp.planner as module
    from concurrent.futures.process import BrokenProcessPool
    b, _ = observed()
    def crashed(job):
        raise BrokenProcessPool('Injected process failure')
    monkeypatch.setattr(module, 'evaluate_world', crashed)
    p = Planner(Config(workers=1, candidates=4, finalists=2, coarse_worlds=2, fine_worlds=2, budget_s=60))
    try:
        assert p.choose(b).kind == 'BASELINE'
        assert p.completion_mode and p.records[-1]['discarded_worlds'] == 2
        assert p.records[-1]['status'] == 'BrokenProcessPool'
    finally:
        p.close()
