"""Controlled frozen-vs-finetuned cost/advantage learning on identical worlds."""
from collections import Counter
import copy
import json
from pathlib import Path
import time

import torch

from .advantage import AdvantageNetwork, loss_terms, choose_supported, INPUTS, GUARD
from .advantage_data import collate, load_states
from .experiment import dump
from .relative_data import sha256


@torch.no_grad()
def evaluate(model, rows, device, batch_size=32):
    totals, count, errors, all_targets = {}, 0, [], []
    preferences, selected, unsupported = [], [], 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        data = collate(chunk, device)
        output = model(*(data[k] for k in INPUTS), data['reference'])
        terms = loss_terms(output['delta_s'], output['scores'], data['teacher_scores'], data['target_s'],
                           data['standard_error_s'], data['evaluated'], data['candidate_mask'], data['reference'])
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.) + float(value) * len(chunk)
        count += len(chunk)
        mask = data['evaluated'].clone()
        mask.scatter_(1, data['reference'][:, None], False)
        prediction = output['delta_s'].mean(0)
        errors.extend((data['target_s'][mask] - prediction[mask]).cpu().tolist())
        all_targets.extend(data['target_s'][mask].cpu().tolist())
        preferences.extend(((prediction[mask] < 0) == (data['target_s'][mask] < 0)).cpu().tolist())
        for index, row in enumerate(chunk):
            ix = row['deploy_indices']
            decision = choose_supported(output['delta_s'][:, index, ix], output['scores'][index, ix])[0]
            action = ix[decision]
            if action == int(row['reference']):
                continue
            if not bool(row['evaluated'][action]):
                unsupported += 1
            else:
                selected.append(dict(group=row['group'], step=row['step'], index=action,
                                     delta_s=float(row['target_s'][action]),
                                     standard_error_s=float(row['standard_error_s'][action])))
    error = torch.tensor(errors, dtype=torch.float64)
    return {**{k: v / count for k, v in totals.items()}, 'states': count,
            'cost_labels': len(errors), 'mae_s': float(error.abs().mean()),
            'underestimate_error_p90_s': float(torch.quantile(error, .9)),
            'sign_accuracy': sum(preferences) / len(preferences),
            'supported_switches': selected, 'switches_without_cost_label': unsupported,
            'supported_switch_total_delta_s': sum(r['delta_s'] for r in selected),
            'calibration_not_closed_loop_performance': True}


def train_variant(manifest, prior, bc_checkpoint, out, finetune, device='cuda',
                  seed=280130000, epochs=40, batch_size=32, patience=8,
                  hard_deadline=float('inf'), dispatch_deadline=float('inf')):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = load_states(manifest, sha256(bc_checkpoint))
    train = [r for r in rows if r['split'] == 'train']
    calibration = [r for r in rows if r['split'] == 'calibration']
    train_groups, cal_groups = {r['group'] for r in train}, {r['group'] for r in calibration}
    if not train_groups or not cal_groups or train_groups & cal_groups:
        raise ValueError('Nonempty isolated world groups required')
    model = AdvantageNetwork.initialize(prior, finetune, seed).to(device)
    encoder_before = {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}
    teacher_before = {key: value.detach().cpu().clone() for key, value in model.teacher.state_dict().items()}
    head_params = [*model.heads.parameters(), *model.actor.parameters()]
    parameter_groups = [{'params': head_params, 'lr': 1e-3}]
    if finetune:
        parameter_groups.append({'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 3e-5})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    groups = sorted(train_groups)
    bootstrap, counts = [], []
    for head in range(3):
        rng = torch.Generator().manual_seed(seed + 1000 + head)
        draw = [groups[i] for i in torch.randint(len(groups), (len(groups),), generator=rng).tolist()]
        bootstrap.append(draw)
        counts.append(Counter(draw))
    history, stale, best_score, best_weights, best_epoch = [], 0, float('inf'), None, 0
    encoder_gradient_norm = 0.
    for epoch in range(1, epochs + 1):
        if time.monotonic() >= min(hard_deadline, dispatch_deadline):
            raise TimeoutError('Round cutoff before new training epoch')
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        losses = []
        for offset in range(0, len(order), batch_size):
            if time.monotonic() >= hard_deadline:
                raise TimeoutError('Round deadline during training')
            chunk = [train[index] for index in order[offset:offset + batch_size]]
            data = collate(chunk, device)
            weight = torch.tensor([[counts[head][r['group']] for r in chunk] for head in range(3)],
                                  device=device, dtype=torch.float32)
            output = model(*(data[k] for k in INPUTS), data['reference'])
            terms = loss_terms(output['delta_s'], output['scores'], data['teacher_scores'], data['target_s'],
                               data['standard_error_s'], data['evaluated'], data['candidate_mask'], data['reference'], weight)
            optimizer.zero_grad(set_to_none=True)
            terms['loss'].backward()
            gradient = model.encoder.candidate_cross_attention.in_proj_weight.grad
            if gradient is not None:
                encoder_gradient_norm = max(encoder_gradient_norm, float(gradient.norm()))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            losses.append(float(terms['loss'].detach()))
        model.eval()
        metrics = evaluate(model, calibration, device)
        row = {'epoch': epoch, 'train_loss': sum(losses)/len(losses), 'calibration': metrics}
        history.append(row)
        if metrics['loss'] < best_score:
            best_score, best_weights, best_epoch, stale = metrics['loss'], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        dump(out / 'history.json', history)
        if epoch % 5 == 0:
            print(f"{'finetuned' if finetune else 'frozen'} epoch={epoch} cal_loss={metrics['loss']:.5f} MAE={metrics['mae_s']:.2f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    model.trained = True
    model.eval()
    changed = [key for key, value in encoder_before.items() if not torch.equal(value, model.encoder.state_dict()[key].cpu())]
    if bool(changed) != finetune:
        raise ValueError('Encoder update does not match the ablation')
    if any(not torch.equal(value, model.teacher.state_dict()[key].cpu()) for key, value in teacher_before.items()):
        raise ValueError('Frozen BC candidate selector changed')
    calibration_metrics = evaluate(model, calibration, device)
    train_metrics = evaluate(model, train, device)
    metadata = {'seed': seed, 'finetune': finetune, 'epochs_max': epochs, 'epochs_run': len(history),
                'best_epoch': best_epoch, 'patience': patience, 'batch_size': batch_size,
                'encoder_lr': 3e-5 if finetune else 0., 'heads_actor_lr': 1e-3,
                'encoder_attention_gradient_norm': encoder_gradient_norm, 'encoder_changed_keys': changed,
                'bc_selector_exactly_frozen': True, 'train_groups': sorted(train_groups),
                'calibration_groups': sorted(cal_groups), 'head_world_bootstrap': bootstrap,
                'ensemble_shares_encoder': True, 'independent_full_model_calibration': False,
                'calibration': calibration_metrics, 'train': train_metrics, 'guard': GUARD,
                'prior_checkpoint_sha256': sha256(prior), 'bc_checkpoint_sha256': sha256(bc_checkpoint),
                'manifest_sha256': sha256(manifest), 'real_time_s': time.monotonic() - started,
                'device': str(device), 'local_simulator_only': True, 'bellman_bootstrapping': False}
    model.metadata = metadata
    model.save(out / 'best.pt')
    dump(out / 'training-summary.json', metadata)
    print(f"trained {'finetuned' if finetune else 'frozen'} best_epoch={best_epoch} cal_MAE={calibration_metrics['mae_s']:.2f}", flush=True)
    return out / 'best.pt'
