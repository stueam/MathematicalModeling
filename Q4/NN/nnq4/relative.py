"""Frozen BC attention with three signed, baseline-relative cost heads.

Only public tensors enter this module. The old absolute Q head is never called.
The uncertainty threshold is an empirical control, not a performance guarantee.
"""
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .features_v2 import FEATURE_VERSION
from .network import PolicyNetwork


CANDIDATE_RULE = 'baseline-local_proxy-survey_proxy-bc_other-fill_proxy-v1'
COST_SCALE = 100.
SWITCH_THRESHOLD_S = 20.
HEADS = 3


def action_identity(action, survey):
    # Exact coordinates, kind, channel AND macro semantics identify an action.
    return action.kind, int(action.channel), tuple(map(float, action.position)), bool(survey)


def four_candidates(frame, scores, mask=None):
    """One deterministic rule for training/deployment; ties use action identity."""
    if frame.candidates.shape[-1] != 48:
        raise ValueError('Four-candidate rule requires v2 features')
    valid = torch.ones(len(frame.actions), dtype=torch.bool) if mask is None else mask.cpu().bool()
    if valid.shape != (len(frame.actions),) or not bool(valid[frame.target]):
        raise ValueError('Baseline must be a valid candidate')
    scores = scores.detach().cpu()
    keys = [action_identity(a, s) for a, s in zip(frame.actions, frame.survey)]
    pool = [i for i in range(len(keys)) if valid[i]]
    if not torch.isfinite(scores[pool]).all() or not torch.isfinite(frame.candidates[pool, 43]).all():
        raise ValueError('Nonfinite candidate ranking features')
    proxy = sorted(pool, key=lambda i: (float(frame.candidates[i, 43]), keys[i]))
    chosen, reasons, identities = [], [], set()

    def add(index, reason):
        if index is not None and keys[index] not in identities and len(chosen) < 4:
            chosen.append(index)
            reasons.append(reason)
            identities.add(keys[index])

    add(frame.target, 'baseline')
    for survey, reason in ((False, 'best_local_proxy'), (True, 'best_survey_proxy')):
        add(next((i for i in proxy if bool(frame.survey[i]) == survey), None), reason)
    bc_order = sorted(pool, key=lambda i: (-float(scores[i]), keys[i]))
    add(next((i for i in bc_order if keys[i] not in identities), None), 'best_bc_other')
    for index in proxy:
        add(index, 'proxy_fill')
    return chosen, reasons


def weighted_huber(prediction_s, target_s, standard_error_s, label_mask):
    """Index BEFORE arithmetic: missing labels may contain NaN, never targets."""
    if not bool(label_mask.any()):
        raise ValueError('No evaluated alternatives in loss')
    target, error = target_s[label_mask], standard_error_s[label_mask]
    if not torch.isfinite(target).all() or not torch.isfinite(error).all() or (error < 0).any():
        raise ValueError('Invalid supervised paired costs')
    weights = (1 / (1 + (error / 50.).square())).clamp_min(.1)
    losses = F.smooth_l1_loss(prediction_s[label_mask] / COST_SCALE, target / COST_SCALE,
                              reduction='none')
    return (weights * losses).sum() / weights.sum()


def calibration_margin(prediction_s, target_s, label_mask):
    if not bool(label_mask.any()):
        raise ValueError('Calibration requires evaluated alternatives')
    errors = (target_s[label_mask] - prediction_s[label_mask]).double()
    if not torch.isfinite(errors).all():
        raise ValueError('Invalid calibration errors')
    return max(0., float(torch.quantile(errors, .9)))


def conservative_selection(predictions, mask, reference, margin_s):
    """Predictions [head,candidate] are signed seconds, with exact zero reference."""
    if predictions.ndim != 2 or predictions.shape[0] != HEADS:
        raise ValueError('Three cost heads required')
    if not math.isfinite(margin_s) or margin_s < 0 or not bool(mask[reference]):
        raise ValueError('Invalid calibration margin or baseline')
    if not torch.isfinite(predictions[:, mask]).all():
        raise ValueError('Nonfinite cost prediction')
    if not bool((predictions[:, reference] == 0).all()):
        raise ValueError('Baseline residual must be exactly zero')
    mean, std = predictions.mean(0), predictions.std(0, correction=0)
    upper = mean + 2 * std + margin_s
    choices = upper.masked_fill(~mask, math.inf)
    selected = int(choices.argmin())
    if selected == reference or not float(choices[selected]) < -SWITCH_THRESHOLD_S:
        selected = reference
    return selected, mean, std, upper


class RelativeCostNetwork(nn.Module):
    def __init__(self, encoder, width=96, seed=280120000):
        super().__init__()
        if (encoder.config.node_dim, encoder.config.candidate_dim, encoder.config.global_dim) != (32, 32, 16):
            raise ValueError('First round requires the frozen v1 BC encoder')
        self.encoder = encoder.eval().requires_grad_(False)
        self.width, self.seed = width, seed
        self.input_dim = 2 * encoder.config.d_model + 48 + 20 + 8
        self.heads = nn.ModuleList()
        for index in range(HEADS):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + index)
                self.heads.append(nn.Sequential(nn.Linear(self.input_dim, width), nn.GELU(),
                                                nn.Linear(width, 1)))
        self.heads_trained = [False] * HEADS
        self.margin_s = None
        self.metadata = {}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def features(self, nodes, node_mask, candidates, candidate_mask, global_features):
        if nodes.shape[-1] != 40 or candidates.shape[-1] != 48 or global_features.shape[-1] != 20:
            raise ValueError('Relative cost model requires v2 40/48/20 tensors')
        encoded = self.encoder.encode(nodes[..., :32], node_mask, candidates[..., :32],
                                      candidate_mask, global_features[..., :16])
        clean = candidates.masked_fill(~candidate_mask[..., None], 0.)
        extra = nodes[..., 32:].masked_fill(~node_mask[..., None], 0.).sum(1)
        extra = extra / node_mask.sum(1, keepdim=True).clamp_min(1)
        count = candidates.shape[1]
        z = torch.cat((encoded['joint'], clean,
                       global_features[:, None].expand(-1, count, -1),
                       extra[:, None].expand(-1, count, -1)), dim=-1)
        return z.masked_fill(~candidate_mask[..., None], 0.), self.encoder.policy_scores(encoded, candidate_mask)

    def residual(self, index, z, mask, reference):
        if not bool(mask.gather(1, reference[:, None]).all()):
            raise ValueError('Masked baseline')
        clean = z.masked_fill(~mask[..., None], 0.)
        values = self.heads[index](clean).squeeze(-1)
        residual = (values - values.gather(1, reference[:, None])) * COST_SCALE
        return residual.masked_fill(~mask, 0.)

    def forward(self, z, mask, reference):
        return torch.stack([self.residual(i, z, mask, reference) for i in range(HEADS)])

    @torch.no_grad()
    def evaluate_frame(self, frame):
        device = next(self.parameters()).device
        data = frame.tensors()
        z, scores = self.features(*(data[k][None].to(device) for k in
                                  ('nodes', 'node_mask', 'candidates', 'candidate_mask', 'global_features')))
        indices, reasons = four_candidates(frame, scores[0])
        # Candidate zero is reference by construction; use an explicit reference
        # tensor throughout regression so padding/permutations cannot change it.
        z = z[:, indices]
        mask = torch.ones(1, len(indices), dtype=torch.bool, device=device)
        reference = torch.zeros(1, dtype=torch.long, device=device)
        return z, mask, reference, indices, reasons

    @torch.no_grad()
    def choose_frame(self, frame):
        if not all(self.heads_trained) or self.margin_s is None:
            raise ValueError('Untrained or uncalibrated relative cost checkpoint')
        z, mask, reference, indices, reasons = self.evaluate_frame(frame)
        predictions = self(z, mask, reference)[:, 0]
        chosen, mean, std, upper = conservative_selection(predictions, mask[0], 0, self.margin_s)
        selected = indices[chosen]
        detail = {'candidate_rule': CANDIDATE_RULE, 'comparison_indices': indices,
                  'comparison_reasons': reasons, 'reference_action': asdict(frame.actions[frame.target]),
                  'selected_action': asdict(frame.actions[selected]),
                  'predicted_delta_s': mean.tolist(), 'ensemble_std_s': std.tolist(),
                  'head_delta_s': predictions.tolist(), 'upper_delta_s': upper.tolist(),
                  'calibration_margin_s': self.margin_s, 'switch_threshold_s': SWITCH_THRESHOLD_S,
                  'std_multiplier': 2., 'std_correction': 0,
                  'selected_slot': chosen, 'empirical_control_only': True}
        return selected, detail

    def save(self, path, **metadata):
        payload = dict(format='q4-relative-cost-v1', feature_version=FEATURE_VERSION,
                       candidate_rule=CANDIDATE_RULE, encoder_config=self.encoder.get_config(),
                       model=self.state_dict(), width=self.width, seed=self.seed,
                       heads_trained=list(self.heads_trained), calibration_margin_s=self.margin_s,
                       cost_scale_s=COST_SCALE, switch_threshold_s=SWITCH_THRESHOLD_S,
                       std_multiplier=2., std_correction=0, encoder_frozen=True,
                       calibration_quantile=.9, metadata={**self.metadata, **metadata})
        path = Path(path)
        temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temp)
        temp.replace(path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        expected = dict(format='q4-relative-cost-v1', feature_version=FEATURE_VERSION,
                        candidate_rule=CANDIDATE_RULE, cost_scale_s=COST_SCALE,
                        switch_threshold_s=SWITCH_THRESHOLD_S, std_multiplier=2.,
                        std_correction=0, encoder_frozen=True, calibration_quantile=.9)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Incompatible relative cost schema/rule/calibration')
        if saved.get('heads_trained') != [True] * HEADS:
            raise ValueError('Cost heads have not all been trained')
        margin = saved.get('calibration_margin_s')
        if margin is None or not math.isfinite(margin) or margin < 0:
            raise ValueError('Missing or invalid calibration')
        model = cls(PolicyNetwork(saved['encoder_config']), saved['width'], saved['seed'])
        model.load_state_dict(saved['model'])
        model.heads_trained, model.margin_s = saved['heads_trained'], margin
        model.metadata = saved['metadata']
        return model.to(device).eval()
