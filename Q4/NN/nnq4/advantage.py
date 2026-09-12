"""Cost-sensitive attention fine-tuning with an advantage-weighted actor.

Uses complete rollout differences, not Bellman targets or the legacy absolute Q
head. AWR/AggreVaTeD motivate the losses; this is a Q4 adaptation, not a paper
reproduction. The deployment guard is an empirical experiment, not a guarantee.
"""
import copy
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .features_v2 import FEATURE_VERSION
from .network import PolicyNetwork
from .relative import RelativeCostNetwork, CANDIDATE_RULE, four_candidates


FORMAT = 'q4-cost-advantage-v1'
DATA_FORMAT = 'q4-all-paired-costs-v2'
GUARD = dict(gain_threshold_s=10., std_multiplier=1., std_correction=0,
             actor_log_odds=math.log(1.1), require_all_heads_negative=True,
             fixed_before_development=True, empirical_only=True)
INPUTS = ('nodes', 'node_mask', 'candidates', 'candidate_mask', 'global_features')


def loss_terms(predicted, actor_logits, teacher_logits, targets, errors, evaluated,
               candidate_mask, reference, bootstrap_weights=None):
    """No arithmetic on missing labels. Baseline is a known zero-cost choice."""
    if evaluated.dtype != torch.bool or bool((evaluated & ~candidate_mask).any()):
        raise ValueError('Invalid supervised action mask')
    if not bool(evaluated.gather(1, reference[:, None]).all()):
        raise ValueError('Baseline cost must be evaluated')
    if not torch.isfinite(targets[evaluated]).all() or not torch.isfinite(errors[evaluated]).all():
        raise ValueError('Nonfinite evaluated cost')
    if bool((errors[evaluated] < 0).any()):
        raise ValueError('Negative standard error')
    if not bool((targets.gather(1, reference[:, None]) == 0).all()):
        raise ValueError('Nonzero baseline cost target')
    alternatives = evaluated.clone()
    alternatives.scatter_(1, reference[:, None], False)
    target, se = targets[alternatives], errors[alternatives]
    quality = (1 / (1 + (se / 50.).square())).clamp_min(.1)
    preference_quality = quality * torch.exp(-target.abs() / 250.)
    soft_preference = torch.sigmoid(-target / torch.sqrt(se.square() + 30.**2))
    if bootstrap_weights is None:
        bootstrap_weights = torch.ones(3, len(targets), device=targets.device)
    regression, preference = [], []
    for head in range(3):
        weight = bootstrap_weights[head, :, None].expand_as(targets)[alternatives]
        estimate = predicted[head][alternatives]
        huber = F.smooth_l1_loss(estimate / 100., target / 100., reduction='none')
        ranking = F.binary_cross_entropy_with_logits(-estimate / 50., soft_preference, reduction='none')
        regression.append((huber * quality * weight).sum() / (quality * weight).sum().clamp_min(1.))
        preference.append((ranking * preference_quality * weight).sum() /
                          (preference_quality * weight).sum().clamp_min(1.))
    # A KL-regularized improvement target, restricted to actually evaluated
    # actions. Missing actions get no cost target; the actor learns data support.
    safe_target = targets.masked_fill(~evaluated, 0.)
    safe_error = errors.masked_fill(~evaluated, 0.)
    temperature = torch.sqrt(50.**2 + safe_error.square())
    tilted = teacher_logits.detach() + (-safe_target / temperature).clamp(-5., 5.)
    probability = tilted.masked_fill(~evaluated, -10000.).softmax(-1).detach()
    log_policy = actor_logits.masked_fill(~candidate_mask, -10000.).log_softmax(-1)
    awr = -(probability * log_policy).sum(-1).mean()
    teacher_probability = teacher_logits.detach().masked_fill(~candidate_mask, -10000.).softmax(-1)
    kl = F.kl_div(log_policy, teacher_probability, reduction='batchmean')
    regression, preference = torch.stack(regression).mean(), torch.stack(preference).mean()
    total = regression + .5 * preference + .3 * awr + .1 * kl
    return {'loss': total, 'regression': regression, 'preference': preference, 'awr': awr, 'kl': kl}


def choose_supported(predictions, actor_scores, reference=0, guard=GUARD):
    if predictions.shape != (3, len(actor_scores)) or not torch.isfinite(predictions).all():
        raise ValueError('Invalid cost ensemble')
    if not torch.isfinite(actor_scores).all() or not bool((predictions[:, reference] == 0).all()):
        raise ValueError('Invalid actor scores or baseline residual')
    mean = predictions.mean(0)
    std = predictions.std(0, correction=guard['std_correction'])
    upper = mean + guard['std_multiplier'] * std
    eligible = (upper < -guard['gain_threshold_s']) & (predictions < 0).all(0)
    eligible &= actor_scores - actor_scores[reference] > guard['actor_log_odds']
    eligible[reference] = False
    selected = int(actor_scores.masked_fill(~eligible, -math.inf).argmax()) if bool(eligible.any()) else reference
    return selected, mean, std, upper, eligible


class AdvantageNetwork(nn.Module):
    def __init__(self, encoder_config, width=96, finetune=True):
        super().__init__()
        self.encoder = PolicyNetwork(encoder_config)
        self.teacher = PolicyNetwork(encoder_config).requires_grad_(False).eval()
        self.width, self.finetune = width, finetune
        self.input_dim = 2 * self.encoder.config.d_model + 48 + 20 + 8
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(self.input_dim, width), nn.GELU(), nn.Linear(width, 1))
                                   for _ in range(3))
        self.actor = nn.Sequential(nn.Linear(self.input_dim, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.encoder.requires_grad_(finetune)
        self.encoder.q_head.requires_grad_(False)
        self.encoder.value_head.requires_grad_(False)
        self.trained = False
        self.metadata = {}

    @classmethod
    def initialize(cls, checkpoint, finetune, seed):
        prior = RelativeCostNetwork.load(checkpoint)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(prior.encoder.get_config(), prior.width, finetune)
        model.encoder.load_state_dict(prior.encoder.state_dict())
        model.teacher.load_state_dict(prior.encoder.state_dict())
        model.heads.load_state_dict(prior.heads.state_dict())
        return model

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if not self.finetune:
            self.encoder.eval()
        return self

    def forward(self, nodes, node_mask, candidates, candidate_mask, global_features, reference):
        if (nodes.shape[-1], candidates.shape[-1], global_features.shape[-1]) != (40, 48, 20):
            raise ValueError('Expected v2 public features')
        if not bool(candidate_mask.gather(1, reference[:, None]).all()):
            raise ValueError('Baseline is padded')
        encoded = self.encoder.encode(nodes[..., :32], node_mask, candidates[..., :32], candidate_mask,
                                      global_features[..., :16])
        clean = candidates.masked_fill(~candidate_mask[..., None], 0.)
        extra = nodes[..., 32:].masked_fill(~node_mask[..., None], 0.).sum(1)
        extra = extra / node_mask.sum(1, keepdim=True).clamp_min(1)
        count = candidates.shape[1]
        z = torch.cat((encoded['joint'], clean, global_features[:, None].expand(-1, count, -1),
                       extra[:, None].expand(-1, count, -1)), -1).masked_fill(~candidate_mask[..., None], 0.)
        values = torch.stack([head(z).squeeze(-1) for head in self.heads])
        residual = (values - values.gather(2, reference[None, :, None].expand(3, -1, -1))) * 100.
        residual = residual.masked_fill(~candidate_mask[None], 0.)
        logits = self.encoder.policy_scores(encoded, candidate_mask) + self.actor(z).squeeze(-1)
        return {'delta_s': residual, 'scores': logits.masked_fill(~candidate_mask, -10000.)}

    @torch.no_grad()
    def teacher_scores(self, nodes, node_mask, candidates, candidate_mask, global_features):
        encoded = self.teacher.encode(nodes[..., :32], node_mask, candidates[..., :32], candidate_mask,
                                      global_features[..., :16])
        return self.teacher.policy_scores(encoded, candidate_mask)

    @torch.no_grad()
    def choose_frame(self, frame):
        if not self.trained:
            raise ValueError('Advantage model has not been trained')
        device = next(self.parameters()).device
        data = frame.tensors()
        inputs = [data[k][None].to(device) for k in INPUTS]
        reference = torch.tensor([frame.target], device=device)
        outputs = self(*inputs, reference)
        teacher = self.teacher_scores(*inputs)
        indices, reasons = four_candidates(frame, teacher[0])
        predictions, actor = outputs['delta_s'][:, 0, indices], outputs['scores'][0, indices]
        selected, mean, std, upper, eligible = choose_supported(predictions, actor)
        chosen = indices[selected]
        detail = {'candidate_rule': CANDIDATE_RULE, 'comparison_indices': indices, 'comparison_reasons': reasons,
                  'reference_action': asdict(frame.actions[frame.target]), 'selected_action': asdict(frame.actions[chosen]),
                  'predicted_delta_s': mean.tolist(), 'ensemble_std_s': std.tolist(),
                  'head_delta_s': predictions.tolist(), 'upper_delta_s': upper.tolist(),
                  'actor_logits': actor.tolist(), 'actor_eligible': eligible.tolist(), 'selected_slot': selected,
                  'guard': GUARD, 'learning_method': FORMAT,
                  'reason': 'advantage_actor_and_cost_agreement' if selected else 'advantage_retain_baseline'}
        return chosen, detail

    def save(self, path, **metadata):
        payload = dict(format=FORMAT, feature_version=FEATURE_VERSION, candidate_rule=CANDIDATE_RULE,
                       guard=GUARD, encoder_config=self.encoder.get_config(), model=self.state_dict(),
                       width=self.width, finetune=self.finetune, trained=self.trained,
                       metadata={**self.metadata, **metadata})
        path = Path(path)
        temp = path.with_suffix('.pt.tmp')
        torch.save(payload, temp)
        temp.replace(path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved.get('format') != FORMAT or saved.get('feature_version') != FEATURE_VERSION or
                saved.get('candidate_rule') != CANDIDATE_RULE or saved.get('guard') != GUARD or not saved.get('trained')):
            raise ValueError('Untrained or incompatible advantage checkpoint')
        model = cls(saved['encoder_config'], saved['width'], saved['finetune'])
        model.load_state_dict(saved['model'])
        model.trained, model.metadata = True, saved['metadata']
        return model.to(device).eval()
