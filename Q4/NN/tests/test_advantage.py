"""Cost-sensitive gradients, support isolation and controlled ablation contracts."""
from dataclasses import replace
import json
import math

import pytest
import torch
from torch.nn import functional as F

from nnq4.advantage import AdvantageNetwork, loss_terms, choose_supported, INPUTS, GUARD, DATA_FORMAT
from nnq4.advantage_data import collate, load_states
from nnq4.advantage_round import paired_improvement, conclusion
from nnq4.network import NetworkConfig
from nnq4.relative import CANDIDATE_RULE
from nnq4.features_v2 import FEATURE_VERSION
from nnq4.relative_data import group_split


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(k=4, n=3, reference=0):
    return {'nodes': torch.randn(n, 40), 'node_mask': torch.ones(n, dtype=torch.bool),
            'candidates': torch.randn(k, 48), 'candidate_mask': torch.ones(k, dtype=torch.bool),
            'global_features': torch.randn(20), 'reference': torch.tensor(reference),
            'target_s': torch.arange(k, dtype=torch.float32) * 20 - reference * 20,
            'standard_error_s': torch.zeros(k), 'evaluated': torch.ones(k, dtype=torch.bool),
            'teacher_scores': torch.randn(k)}


def model(finetune=True):
    return AdvantageNetwork(NetworkConfig(d_model=16, n_layers=1, ffn_dim=32), width=16, finetune=finetune)


def test_padding_permutations_and_exact_reference_zero():
    net = model().eval()
    rows = [example(reference=2), example(3, 2, 1)]
    data = collate(rows)
    expected = net(*(data[k] for k in INPUTS), data['reference'])
    assert torch.equal(expected['delta_s'][:, torch.arange(2), data['reference']], torch.zeros(3, 2))
    order = torch.tensor([3, 0, 2, 1])
    alternate = dict(data, candidates=data['candidates'][:, order], candidate_mask=data['candidate_mask'][:, order])
    actual = net(*(alternate[k] for k in INPUTS), torch.argsort(order)[data['reference']])
    torch.testing.assert_close(actual['delta_s'], expected['delta_s'][:, :, order])
    torch.testing.assert_close(actual['scores'], expected['scores'][:, order], atol=2e-6, rtol=1e-5)
    altered = {**data, 'nodes': F.pad(data['nodes'], (0, 0, 0, 2), value=math.nan),
               'node_mask': F.pad(data['node_mask'], (0, 2)),
               'candidates': F.pad(data['candidates'], (0, 0, 0, 2), value=math.inf),
               'candidate_mask': F.pad(data['candidate_mask'], (0, 2))}
    padded = net(*(altered[k] for k in INPUTS), data['reference'])
    torch.testing.assert_close(padded['delta_s'][..., :4], expected['delta_s'])
    torch.testing.assert_close(padded['scores'][..., :4], expected['scores'], atol=2e-6, rtol=1e-5)


def test_missing_costs_do_not_enter_regression_or_preferences():
    prediction = torch.randn(3, 1, 4, requires_grad=True)
    targets = torch.tensor([[0., -40., math.nan, math.inf]])
    se = torch.tensor([[0., 10., math.inf, math.nan]])
    evaluated = torch.tensor([[True, True, False, False]])
    terms = loss_terms(prediction, torch.zeros(1, 4), torch.zeros(1, 4), targets, se,
                       evaluated, torch.ones_like(evaluated), torch.tensor([0]))
    assert all(torch.isfinite(value) for value in terms.values())
    (terms['regression'] + terms['preference']).backward()
    assert torch.equal(prediction.grad[..., 2:], torch.zeros(3, 1, 2))


def test_awr_prefers_lower_cost_but_softens_noisy_advantage():
    def gradient(se):
        logits = torch.zeros(1, 2, requires_grad=True)
        terms = loss_terms(torch.zeros(3, 1, 2), logits, torch.zeros(1, 2), torch.tensor([[0., -100.]]),
                           torch.tensor([[0., se]]), torch.ones(1, 2, dtype=torch.bool),
                           torch.ones(1, 2, dtype=torch.bool), torch.tensor([0]))
        terms['awr'].backward()
        return logits.grad[0, 1]
    assert gradient(0.) < gradient(1000.) < 0  # gradient descent raises the better action logit


@pytest.mark.parametrize('finetune', [False, True])
def test_gradient_ablation_and_no_absolute_q(finetune, monkeypatch):
    net = model(finetune)
    def forbidden(*args):
        raise AssertionError('Untrained absolute Q/value head was used')
    monkeypatch.setattr(net.encoder.q_head, 'forward', forbidden)
    monkeypatch.setattr(net.encoder.value_head, 'forward', forbidden)
    rows = [example(), example(reference=1)]
    data = collate(rows)
    net.train()
    output = net(*(data[k] for k in INPUTS), data['reference'])
    terms = loss_terms(output['delta_s'], output['scores'], data['teacher_scores'], data['target_s'],
                       data['standard_error_s'], data['evaluated'], data['candidate_mask'], data['reference'])
    terms['loss'].backward()
    assert all(p.grad is None and not p.requires_grad for p in net.teacher.parameters())
    gradient = net.encoder.candidate_cross_attention.in_proj_weight.grad
    if finetune:
        assert gradient is not None and gradient.norm() > 0 and torch.isfinite(gradient).all()
    else:
        assert gradient is None and all(p.grad is None for p in net.encoder.parameters())
    assert net.actor[-1].weight.grad.norm() > 0


def test_guard_requires_cost_consensus_actor_preference_and_gain():
    costs = torch.tensor([[0., -30., -50.], [0., -31., 10.], [0., -29., -80.]])
    assert choose_supported(costs, torch.tensor([0., 1., 3.]))[0] == 1
    assert choose_supported(costs, torch.tensor([2., 1., 3.]))[0] == 0
    costs[:, 1] = -10.
    assert choose_supported(costs, torch.tensor([0., 1., 3.]))[0] == 0


def test_checkpoints_refuse_missing_training_or_changed_guard(tmp_path):
    net = model()
    path = tmp_path / 'model.pt'
    net.save(path)
    with pytest.raises(ValueError, match='Untrained'):
        AdvantageNetwork.load(path)
    net.trained = True
    net.save(path)
    restored = AdvantageNetwork.load(path)
    data = collate([example(reference=1)])
    for k, v in net(*(data[k] for k in INPUTS), data['reference']).items():
        torch.testing.assert_close(v, restored(*(data[k] for k in INPUTS), data['reference'])[k])
    saved = torch.load(path, weights_only=True)
    saved['guard'] = {**GUARD, 'gain_threshold_s': 0.}
    torch.save(saved, path)
    with pytest.raises(ValueError, match='incompatible'):
        AdvantageNetwork.load(path)


def test_loader_rejects_world_leakage_and_encoder_mismatch(tmp_path):
    row = {**example(), 'format': DATA_FORMAT, 'feature_version': FEATURE_VERSION,
           'candidate_rule': CANDIDATE_RULE, 'bc_checkpoint_sha256': 'correct',
           'group': 'public-group', 'step': 12, 'split': group_split('public-group')}
    path = tmp_path / 'state.pt'
    torch.save(row, path)
    record = {'path': str(path), 'group': row['group'], 'split': row['split']}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'format': DATA_FORMAT, 'feature_version': FEATURE_VERSION, 'records': [record]}))
    assert len(load_states(manifest, 'correct')) == 1
    with pytest.raises(ValueError, match='provenance'):
        load_states(manifest, 'wrong')
    row['split'] = 'train' if row['split'] == 'calibration' else 'calibration'
    torch.save(row, path)
    with pytest.raises(ValueError, match='split'):
        load_states(manifest, 'correct')


def test_pairing_and_incomplete_round_cannot_claim_improvement():
    pair = paired_improvement({1: 100., 2: 200.}, {1: 90., 2: 180.})
    assert pair['relative_improvement'] == pytest.approx(.1)
    with pytest.raises(ValueError, match='identical'):
        paired_improvement({1: 100.}, {2: 90.})
    result = conclusion([])
    assert result['conclusion'] == 'stop' and not result['all_complete_and_audited']


def test_budget_extension_counts_time_from_original_start():
    from nnq4.advantage_resume import remaining_budget
    hard, dispatch = remaining_budget(100., 1000.)
    assert hard == 899. and dispatch == 719.
    with pytest.raises(TimeoutError, match='extended dispatch'):
        remaining_budget(100., 1720.)
