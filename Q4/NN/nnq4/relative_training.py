"""Independent world-bootstrap regression of frozen-attention cost heads."""
from collections import Counter
import copy
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .cost_learning import _network
from .experiment import dump
from .relative import RelativeCostNetwork, weighted_huber, calibration_margin, HEADS
from .relative_data import sha256, group_split, SPLIT_RULE


def load_states(manifest):
    document = json.loads(Path(manifest).read_text())
    rows, groups, seen = [], {'train': set(), 'calibration': set()}, set()
    for record in document['records']:
        row = torch.load(record['path'], map_location='cpu', weights_only=True)
        if row['group'] != record['group'] or row['split'] != group_split(row['group']) or row['split'] != record['split']:
            raise ValueError('World split provenance mismatch')
        identity = (row['group'], row['step'])
        if identity in seen:
            raise ValueError('Duplicate state')
        seen.add(identity)
        if bool((row['label_mask'] & ~row['mask']).any()):
            raise ValueError('Labels on padded candidates')
        if bool(row['label_mask'].gather(1, row['reference'][:, None]).any()):
            raise ValueError('Baseline is not a regression alternative')
        groups[row['split']].add(row['group'])
        rows.append(row)
    if groups['train'] & groups['calibration']:
        raise ValueError('World grouping leakage')
    # Baseline-only states remain in provenance; they have no regression loss.
    usable = [r for r in rows if bool(r['label_mask'].any())]
    if not all(any(r['split'] == split for r in usable) for split in groups):
        raise ValueError('Both splits need evaluated nonbaseline alternatives')
    return rows, groups


def batch(rows):
    count = max(r['z'].shape[1] for r in rows)
    result = {}
    for key in ('z', 'mask', 'target_s', 'standard_error_s', 'label_mask'):
        values = []
        for row in rows:
            padding = count - row['z'].shape[1]
            values.append(F.pad(row[key], (0, 0, 0, padding) if key == 'z' else (0, padding)))
        result[key] = torch.cat(values)
    result['reference'] = torch.cat([r['reference'] for r in rows])
    return result


def bootstrap_rows(rows, generator):
    groups = sorted({r['group'] for r in rows})
    draws = [groups[i] for i in torch.randint(len(groups), (len(groups),), generator=generator).tolist()]
    by_group = {g: [r for r in rows if r['group'] == g] for g in groups}
    return [r for g in draws for r in by_group[g]], draws


def train_relative(manifest, checkpoint, out, seed=280120000, epochs=40, batch_size=64,
                   learning_rate=1e-3, patience=5, hard_deadline=float('inf'), device='cuda'):
    if not 1 <= epochs <= 40 or batch_size < 1 or patience < 1:
        raise ValueError('Invalid first-round training limits')
    torch.set_num_threads(1)
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, groups = load_states(manifest)
    train = [r for r in rows if r['split'] == 'train']
    calibration = {k: v.to(device) for k, v in batch([r for r in rows if r['split'] == 'calibration']).items()}
    model = RelativeCostNetwork(_network(str(checkpoint)), seed=seed).to(device)
    frozen = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    histories, bootstrap = [], []
    for head in range(HEADS):
        generator = torch.Generator().manual_seed(seed + 1000 + head)
        samples, draws = bootstrap_rows(train, generator)
        samples = [r for r in samples if bool(r['label_mask'].any())]
        if not samples:
            raise ValueError('Bootstrap contains no supervised alternative')
        bootstrap.append(dict(head=head, group_draws=draws, multiplicity=dict(Counter(draws))))
        optimizer = torch.optim.AdamW(model.heads[head].parameters(), lr=learning_rate)
        best, best_weights, stale, history = float('inf'), None, 0, []
        for epoch in range(1, epochs + 1):
            if time.monotonic() >= hard_deadline:
                raise TimeoutError('Round deadline during cost head training')
            model.train()
            order = torch.randperm(len(samples), generator=generator).tolist()
            losses = []
            for offset in range(0, len(order), batch_size):
                data = {k: v.to(device) for k, v in batch([samples[i] for i in order[offset:offset + batch_size]]).items()}
                prediction = model.residual(head, data['z'], data['mask'], data['reference'])
                loss = weighted_huber(prediction, data['target_s'], data['standard_error_s'], data['label_mask'])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                prediction = model.residual(head, calibration['z'], calibration['mask'], calibration['reference'])
                score = float(weighted_huber(prediction, calibration['target_s'], calibration['standard_error_s'],
                                             calibration['label_mask']))
            improved = score < best
            history.append(dict(epoch=epoch, train_loss=sum(losses)/len(losses), calibration_loss=score,
                                improved=improved))
            if improved:
                best, best_weights, stale = score, copy.deepcopy(model.heads[head].state_dict()), 0
            else:
                stale += 1
            dump(out / f'head-{head}-history.json', history)
            if stale >= patience:
                break
        model.heads[head].load_state_dict(best_weights)
        model.heads_trained[head] = True
        histories.append(history)
        model.save(out / 'partial.pt', completed_heads=head + 1)
        print(f'cost head {head + 1}/3: epochs={len(history)} best_calibration_huber={best:.6f}', flush=True)
    if any(not torch.equal(value, model.encoder.state_dict()[key]) for key, value in frozen.items()):
        raise RuntimeError('Frozen encoder was modified')
    model.eval()
    with torch.no_grad():
        predictions = model(calibration['z'], calibration['mask'], calibration['reference'])
        model.margin_s = calibration_margin(predictions.mean(0), calibration['target_s'], calibration['label_mask'])
        valid = calibration['label_mask']
        residual = (calibration['target_s'][valid] - predictions.mean(0)[valid]).double().cpu()
    metadata = dict(seed=seed, epochs_max=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    optimizer='AdamW', weight_decay=.01, patience=patience, huber_scale_s=100.,
                    label_weight='max(0.1, 1 / (1 + (paired_standard_error_s / 50)^2))',
                    encoder_checkpoint=str(Path(checkpoint).resolve()), encoder_sha256=sha256(checkpoint),
                    dataset_manifest_sha256=sha256(manifest), split_rule=SPLIT_RULE,
                    groups={k: sorted(v) for k, v in groups.items()}, bootstrap=bootstrap,
                    encoder_frozen_verified=True, calibration_is_independent_full_model_test=False,
                    calibration_caveat='Frozen encoder has seen some cost worlds; head tuning and margin share calibration set',
                    calibration_count=int(valid.sum()), calibration_margin_s=model.margin_s,
                    calibration_underestimate_error_quantiles_s=torch.quantile(residual, torch.tensor([.1, .5, .9], dtype=torch.float64)).tolist(),
                    calibration_mae_s=float(residual.abs().mean()), std_correction=0,
                    selected_epochs=[min(h, key=lambda r: r['calibration_loss'])['epoch'] for h in histories],
                    real_time_s=time.monotonic()-started, device=str(device),
                    device_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
                    local_simulator_only=True)
    model.metadata = metadata
    model.save(out / 'best.pt')
    dump(out / 'training-summary.json', metadata)
    dump(out / 'calibration.json', {'target_s': calibration['target_s'][valid].tolist(),
                                   'mean_prediction_s': predictions.mean(0)[valid].tolist(),
                                   'std_s': predictions.std(0, correction=0)[valid].tolist(),
                                   'margin_s': model.margin_s, 'quantile': .9})
    return out / 'best.pt'
