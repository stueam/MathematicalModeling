"""Relative-cost safety contracts, grouping, public replay and cancellation."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nnq4.bridge import Action, Belief, Config
from nnq4.cost_learning import replay_snapshots, make_rollout_jobs
from nnq4.network import PolicyNetwork
from nnq4.relative import (RelativeCostNetwork, four_candidates, action_identity,
                           weighted_huber, calibration_margin, conservative_selection)
from nnq4.relative_data import completed_records, paired_targets, group_split, _match_frame
from nnq4.relative_training import bootstrap_rows
from nnq4.relative_round import evaluate_gate
from nnq4.round_control import supervise, live_group_members, bounded_map
from nnq4.state import Frame, MenuPlanner
from q4.rollout import continue_world
from q4.simulator import World, Source
from test_cost_learning import public_mission


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example_frame():
    actions = [Action('measure', (float(i * 10), 0.), i + 1) for i in range(7)]
    values = torch.randn(7, 48, generator=torch.Generator().manual_seed(51))
    values[:, 43] = torch.tensor([9., 1., 4., 3., 2., 5., -100.])
    return Frame(torch.randn(4, 40), values, torch.randn(20), actions,
                 [False, False, True, True, False, True, True], 0, 'test', ['local'] * 7)


def test_four_slots_permutation_ties_and_mask():
    frame = example_frame()
    scores = torch.tensor([0., 3., 4., 5., 9., 1., 1000.])
    mask = torch.tensor([True] * 6 + [False])
    indices, reasons = four_candidates(frame, scores, mask)
    assert indices == [0, 1, 3, 4]
    order = torch.tensor([6, 4, 3, 1, 5, 0, 2])
    other = replace(frame, candidates=frame.candidates[order], actions=[frame.actions[i] for i in order],
                    survey=[frame.survey[i] for i in order], target=5)
    selected, other_reasons = four_candidates(other, scores[order], mask[order])
    assert [action_identity(other.actions[i], other.survey[i]) for i in selected] == [
        action_identity(frame.actions[i], frame.survey[i]) for i in indices]
    assert reasons == other_reasons
    # Baseline is also the best local: deduplicate, then fill by proxy order.
    frame.candidates[0, 43] = -1
    assert four_candidates(frame, scores, mask)[0] == [0, 3, 4, 1]


def test_cost_heads_are_signed_reference_zero_and_padding_invariant():
    model = RelativeCostNetwork(PolicyNetwork(d_model=16, ffn_dim=32, n_layers=1), width=16).eval()
    z = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor([[True] * 4, [True, False, True, True]])
    reference = torch.tensor([2, 0])
    expected = model(z, mask, reference)
    assert torch.equal(expected[:, torch.arange(2), reference], torch.zeros(3, 2))
    padded = model(F.pad(z, (0, 0, 0, 2), value=math.nan), F.pad(mask, (0, 2)), reference)
    torch.testing.assert_close(padded[..., :4], expected)
    assert (padded[..., 4:] == 0).all()
    permutation = torch.tensor([3, 0, 2, 1])
    other_reference = torch.argsort(permutation)[reference]
    actual = model(z[:, permutation], mask[:, permutation], other_reference)
    torch.testing.assert_close(actual, expected[:, :, permutation])
    # Signed heads use subtraction, so exchanging baseline/action negates cost.
    first = model(z[:1], mask[:1], torch.tensor([0]))[:, 0, 2]
    torch.testing.assert_close(first, -expected[:, 0, 0])


def test_frozen_attention_never_evaluates_absolute_q_and_uses_v2(monkeypatch):
    model = RelativeCostNetwork(PolicyNetwork(d_model=16, ffn_dim=32, n_layers=1), width=16)
    def forbidden(*args):
        raise AssertionError('Absolute Q must never run')
    monkeypatch.setattr(model.encoder.q_head, 'forward', forbidden)
    frame = example_frame()
    a = frame.tensors()
    inputs = [a[k][None] for k in ('nodes', 'node_mask', 'candidates', 'candidate_mask', 'global_features')]
    z, _ = model.features(*inputs)
    altered = inputs.copy()
    altered[2] = inputs[2].clone()
    altered[2][..., 42] += 1
    different, _ = model.features(*altered)
    assert not torch.equal(z, different)
    model.train()
    model(z, inputs[3], torch.tensor([0])).sum().backward()
    assert not model.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    assert all(any(p.grad is not None for p in head.parameters()) for head in model.heads)


def test_missing_labels_are_isolated_from_values_and_gradients():
    prediction = torch.tensor([[50., 10., -9.]], requires_grad=True)
    target = torch.tensor([[0., math.nan, math.inf]])
    error = torch.tensor([[50., math.nan, math.inf]])
    mask = torch.tensor([[True, False, False]])
    loss = weighted_huber(prediction, target, error, mask)
    assert float(loss.detach()) == pytest.approx(.125)
    loss.backward()
    assert torch.equal(prediction.grad[:, 1:], torch.zeros(1, 2))
    with pytest.raises(ValueError, match='No evaluated'):
        weighted_huber(prediction, target, error, torch.zeros_like(mask))


def test_uncertainty_threshold_and_calibration_direction():
    mask = torch.ones(3, dtype=torch.bool)
    prediction = torch.tensor([[0., -50., -100.], [0., -50., 0.], [0., -50., -20.]])
    assert conservative_selection(prediction, mask, 0, 29.)[0] == 1
    assert conservative_selection(prediction, mask, 0, 30.)[0] == 0  # strictly below -20
    selected = torch.tensor([True, True, False])
    assert calibration_margin(torch.tensor([10., 10., math.nan]), torch.tensor([20., 40., math.nan]), selected) == pytest.approx(28.)
    assert calibration_margin(torch.tensor([30., 40., math.nan]), torch.tensor([10., 20., math.nan]), selected) == 0.


def test_checkpoint_rejects_untrained_or_incompatible_rule(tmp_path):
    model = RelativeCostNetwork(PolicyNetwork(d_model=16, ffn_dim=32, n_layers=1), width=16)
    path = tmp_path / 'model.pt'
    model.save(path)
    with pytest.raises(ValueError, match='not all been trained'):
        RelativeCostNetwork.load(path)
    model.heads_trained, model.margin_s = [True] * 3, 12.
    model.save(path)
    restored = RelativeCostNetwork.load(path)
    assert restored.margin_s == 12.
    z = torch.randn(1, 3, model.input_dim)
    mask, reference = torch.ones(1, 3, dtype=torch.bool), torch.tensor([1])
    torch.testing.assert_close(model(z, mask, reference), restored(z, mask, reference))
    saved = torch.load(path, weights_only=True)
    saved['candidate_rule'] = 'stale'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='Incompatible'):
        RelativeCostNetwork.load(path)


def test_group_bootstrap_moves_whole_worlds_and_split_is_fixed():
    rows = [{'group': f'world-{g}', 'step': step} for g in range(12) for step in range(3)]
    sample, draws = bootstrap_rows(rows, torch.Generator().manual_seed(123))
    for group in set(draws):
        assert [r['step'] for r in sample if r['group'] == group] == [0, 1, 2] * draws.count(group)
    train = {r['group'] for r in rows if group_split(r['group']) == 'train'}
    cal = {r['group'] for r in rows if group_split(r['group']) == 'calibration'}
    assert train and cal and not train & cal
    assert all(group_split(r['group']) == group_split(r['group']) for r in reversed(rows))


def test_paired_standard_error_is_of_differences_and_failure_rejects():
    state = {'coverage': {'indices': [5, 2], 'teacher_index': 5},
             'sampling': [{'world_index': w, 'accepted': True} for w in range(3)],
             'rollouts': [{'world_index': w, 'candidate_index': c, 'complete': True, 'cost_s': cost}
                          for w, costs in enumerate(((1000., 1010.), (2000., 2010.), (3000., 3010.)))
                          for c, cost in zip((5, 2), costs)]}
    _, mean, se, _ = paired_targets(state)
    assert mean.tolist() == [0., 10.] and se.tolist() == [0., 0.]
    with pytest.raises(ValueError, match='Incomplete'):
        paired_targets({**state, 'rollouts': state['rollouts'][:-2]})
    state['rollouts'][-1]['complete'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        paired_targets(state)


def test_manifest_only_ignores_orphan_diagnostics(tmp_path):
    original = {'seed': 10, 'source_group': 'world10', 'split': 'train',
                'public_path': str(tmp_path / 'public.jsonl.gz')}
    Path(original['public_path']).write_bytes(b'public-only-marker')
    folder = tmp_path / 'finished'
    folder.mkdir()
    path = folder / 'episode.npz'
    np.savez(path, target=np.array([0]))
    (folder / 'diagnostics.json').write_text(json.dumps(dict(input_record=original, status='complete',
                  states=[dict(step=0, retained=True, status='complete')], retained_states=1)))
    record = {**original, 'path': str(path), 'complete': True, 'original_public_path': original['public_path']}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': [record]}))
    orphan = tmp_path / 'orphan'
    orphan.mkdir()
    (orphan / 'diagnostics.json').write_bytes(b'INVALID: must never be read')
    (tmp_path / 'world.json').write_bytes(b'INVALID: must never be read')
    jobs, detail = completed_records(manifest, expected_states=1)
    assert len(jobs) == detail['completed_states'] == 1
    assert not detail['real_world_files_read'] and not detail['orphan_diagnostics_included']


def test_public_v2_replay_matches_each_action_and_old_prefix(public_mission):
    record, _, steps, _ = public_mission
    states, _ = replay_snapshots(record, [steps[0], steps[-1]], snapshot_features='v2')
    for state in states:
        old, new = state['source_frame'], state['frame']
        assert old.actions == new.actions and old.survey == new.survey
        assert torch.equal(old.candidates, new.candidates[:, :32])
        assert torch.equal(old.nodes, new.nodes[:, :32])
        assert new.global_features.shape == (20,)
        saved = {k: v[None].numpy() for k, v in old.tensors().items()}
        _match_frame(saved, old, 0)
        saved['candidates'][0, 0, 0] += .1
        with pytest.raises(ValueError, match='mismatch'):
            _match_frame(saved, new, 0)


def test_survey_macro_complete_continuation_charges_every_channel():
    belief = Belief()
    belief.deadline = math.inf
    planner = MenuPlanner()
    frame = planner.frame(belief)
    assert frame.survey[frame.target]
    world = World([Source(c, (0., 0.), 1200.) for c in range(1, 17)], seed=812)
    snapshot = dict(frame=frame, belief=belief, config=Config(), points=planner.points,
                    shared_checked=planner.shared_checked, completion_mode=planner.completion_mode)
    job = make_rollout_jobs(snapshot, [world], [frame.target], 100, 10)[0]
    result = continue_world(job)
    assert result['complete'] and result['steps'] == 32
    assert result['cost_s'] == pytest.approx(16 * 5 + 15 + 16 * 5)
    job['max_steps'] = 1
    failed = continue_world(job)
    assert not failed['complete'] and failed['cost_s'] is None


def test_pause_kills_work_process_and_grandchild(tmp_path):
    child_code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
    command = ('import subprocess,sys,time,pathlib; '
               f'subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
               f'time.sleep(.3); pathlib.Path({str(tmp_path / "STOP")!r}).touch(); time.sleep(60)')
    control = supervise([sys.executable, '-c', command], tmp_path, hard_limit_s=5, dispatch_limit_s=3)
    process = json.loads((tmp_path / 'process.json').read_text())
    assert control['status'] == 'interrupted_by_user'
    assert control['real_time_s'] < 5
    # kill delivery is asynchronous; allow a short kernel scheduling interval.
    for _ in range(20):
        if not live_group_members(process['worker_pgid']):
            break
        time.sleep(.01)
    assert live_group_members(process['worker_pgid']) == []


def test_deadline_enforced_and_overlong_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match='1800'):
        supervise([], tmp_path, hard_limit_s=1801)
    control = supervise([sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path,
                        hard_limit_s=1, dispatch_limit_s=.2)
    assert control['status'] == 'hard_deadline' and control['real_time_s'] < 1.2


def test_no_new_dispatch_after_cutoff(tmp_path):
    results = []
    now = time.monotonic()
    status = bounded_map(abs, [-1, -2], 1, now - 1, now + 10, results.append)
    assert status['dispatched'] == 0 and not results


def test_empty_or_full_baseline_retreat_cannot_pass_gate():
    assert evaluate_gate([])['continue_condition_met'] is False
    def row(mode):
        return dict(seed=1, mode=mode, complete=True, audit_ok=True, public_complete=True,
                    world_sha256='same', virtual_time_s=1000., source_count=10, movement_m=100.,
                    real_time_s=1., actions=10, executor_actions={}, stats={}, case=mode)
    gate = evaluate_gate([row(mode) for mode in ('probes', 'neural', 'hybrid')], expected_seeds=[1])
    assert not gate['continue_condition_met'] and 'no_network_override' in gate['reasons']
