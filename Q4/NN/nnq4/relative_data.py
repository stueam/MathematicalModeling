"""Manifest-only reuse of completed costs; public replay never opens world.json."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np  # Existing NPZ interchange only.
import torch

from .cost_learning import paired_costs, replay_snapshots, _network
from .experiment import dump
from .relative import RelativeCostNetwork, action_identity, CANDIDATE_RULE


SPLIT_RULE = 'sha256(q4-relative-cost-split-v1:source_group) first8 modulo5; 0=calibration'


def group_split(group):
    value = hashlib.sha256(('q4-relative-cost-split-v1:' + group).encode()).digest()
    return 'calibration' if int.from_bytes(value[:8], 'big') % 5 == 0 else 'train'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_records(manifest, expected_states=213):
    """Only manifest paths are read. Unlisted episode diagnostics are invisible."""
    manifest = Path(manifest).resolve()
    records = json.loads(manifest.read_text())['records']
    seen, count, jobs, provenance = set(), 0, [], {}
    for record in records:
        if record.get('complete') is not True or record.get('split') != 'train':
            raise ValueError('Only completed original training records are allowed')
        path = (manifest.parent / record['path']).resolve()
        if str(path) in seen:
            raise ValueError('Duplicate manifest record')
        seen.add(str(path))
        diagnostics_path = path.parent / 'diagnostics.json'
        diagnostic = json.loads(diagnostics_path.read_text())
        original = diagnostic['input_record']
        if any(original[k] != record[k] for k in ('seed', 'source_group', 'split')):
            raise ValueError('Original world identity mismatch')
        if record.get('original_public_path') != original['public_path']:
            raise ValueError('Public history provenance mismatch')
        states = sorted([s for s in diagnostic['states'] if s.get('retained') is True and s.get('status') == 'complete'],
                        key=lambda s: s['step'])
        if len({s['step'] for s in states}) != len(states) or diagnostic.get('status') != 'complete':
            raise ValueError('Unfinalized or duplicate state diagnostics')
        with np.load(path, allow_pickle=False) as arrays:
            if len(arrays['target']) != len(states) or len(states) != diagnostic['retained_states']:
                raise ValueError('NPZ and completed diagnostics disagree')
        count += len(states)
        jobs.append({'record': record, 'path': str(path), 'diagnostics': str(diagnostics_path)})
        provenance[str(path)] = sha256(path)
        provenance[str(diagnostics_path)] = sha256(diagnostics_path)
        provenance[original['public_path']] = sha256(original['public_path'])
    if count != expected_states:
        raise ValueError(f'Expected exactly {expected_states} manifest states, got {count}')
    return jobs, {'manifest_sha256': sha256(manifest), 'completed_states': count,
                  'original_worlds': len({j['record']['source_group'] for j in jobs}),
                  'files_sha256': provenance, 'split_rule': SPLIT_RULE,
                  'real_world_files_read': False, 'orphan_diagnostics_included': False}


def paired_targets(state):
    indices = state['coverage']['indices']
    reference = state['coverage']['teacher_index']
    if reference not in indices or len(set(indices)) != len(indices):
        raise ValueError('Invalid evaluated candidate identities')
    # The sampling ledger defines the expected batch. Inferring it from the
    # results would silently accept a completely missing final world.
    accepted = [sample['world_index'] for sample in state['sampling'] if sample.get('accepted') is True]
    if len(set(accepted)) != len(accepted) or set(accepted) != set(range(len(accepted))):
        raise ValueError('Invalid shared-world sampling ledger')
    costs = paired_costs(state['rollouts'], len(accepted), indices)
    if costs is None or len(accepted) < 2:
        raise ValueError('Incomplete paired continuations or undefined standard error')
    delta = costs - costs[:, indices.index(reference), None]
    return indices, delta.mean(0), delta.std(0, correction=1) / math.sqrt(len(accepted)), costs.mean(0)


def _match_frame(old, rebuilt, row):
    """Verify the entire old feature menu before interpreting saved numeric IDs."""
    for key in ('nodes', 'candidates'):
        mask_key = 'node_mask' if key == 'nodes' else 'candidate_mask'
        mask = torch.as_tensor(old[mask_key][row])
        value = torch.as_tensor(old[key][row])[mask]
        actual = getattr(rebuilt, key)
        if value.shape != actual.shape or not torch.allclose(value, actual, atol=2e-6, rtol=1e-6):
            raise ValueError(f'Public replay {key} identity/features mismatch')
    if not torch.allclose(torch.as_tensor(old['global_features'][row]), rebuilt.global_features, atol=2e-6, rtol=1e-6):
        raise ValueError('Public replay global state mismatch')


def reuse_episode(job):
    torch.set_num_threads(1)
    diagnostic = json.loads(Path(job['diagnostics']).read_text())
    states = sorted([s for s in diagnostic['states'] if s.get('retained') is True and s.get('status') == 'complete'],
                    key=lambda s: s['step'])
    snapshots, replay = replay_snapshots(diagnostic['input_record'], [s['step'] for s in states], snapshot_features='v2')
    model = RelativeCostNetwork(_network(job['checkpoint']), seed=job['seed']).eval()
    records = []
    with np.load(job['path'], allow_pickle=False) as old:
        for row, (state, snapshot) in enumerate(zip(states, snapshots)):
            if time.monotonic() >= job['hard_deadline']:
                raise TimeoutError('Round deadline during public reuse')
            frame, source = snapshot['frame'], snapshot['source_frame']
            _match_frame(old, source, row)
            source_keys = [action_identity(a, s) for a, s in zip(source.actions, source.survey)]
            keys = [action_identity(a, s) for a, s in zip(frame.actions, frame.survey)]
            if source_keys != keys or len(set(keys)) != len(keys) or frame.target != state['coverage']['teacher_index']:
                raise ValueError('Action-by-action v1/v2/reference identity mismatch')
            evaluated, means, errors, absolute = paired_targets(state)
            expected_mask = torch.zeros(len(keys), dtype=torch.bool)
            expected_mask[evaluated] = True
            if not torch.equal(torch.as_tensor(old['q_mask'][row, :len(keys)]), expected_mask):
                raise ValueError('Saved evaluation mask mismatch')
            if not torch.allclose(torch.as_tensor(old['q_costs'][row, evaluated]).double(), absolute, atol=.002, rtol=1e-6):
                raise ValueError('Saved cost means differ from paired world results')
            z, mask, reference, chosen, reasons = model.evaluate_frame(frame)
            targets = torch.full((1, len(chosen)), math.nan)
            se = torch.full_like(targets, math.nan)
            label_mask = torch.zeros_like(mask)
            for slot, index in enumerate(chosen):
                if index in evaluated:
                    column = evaluated.index(index)
                    targets[0, slot], se[0, slot] = means[column], errors[column]
                    label_mask[0, slot] = index != frame.target
            group = job['record']['source_group']
            state_id = hashlib.sha256(f'{group}:{state["step"]}'.encode()).hexdigest()[:24]
            path = Path(job['out']) / f'{state_id}.pt'
            payload = dict(z=z, mask=mask, reference=reference, target_s=targets,
                           standard_error_s=se, label_mask=label_mask, group=group,
                           step=state['step'], stage=state['stage'], split=group_split(group),
                           chosen_indices=chosen, evaluated_indices=evaluated,
                           action_identities=[keys[i] for i in chosen], candidate_rule=CANDIDATE_RULE)
            temp = path.with_suffix('.pt.tmp')
            torch.save(payload, temp)
            temp.replace(path)
            record = dict(path=str(path.resolve()), group=group, split=group_split(group),
                          step=state['step'], stage=state['stage'], complete=True,
                          labeled_alternatives=int(label_mask.sum()), selected=len(chosen),
                          missing_alternatives=len(chosen) - 1 - int(label_mask.sum()),
                          chosen_indices=chosen, reasons=reasons,
                          input_npz=job['path'], input_row=row)
            # Commit each finished state immediately, even if later work stops.
            dump(path.with_suffix('.json'), record)
            records.append(record)
    return {'records': records, 'public_replay': replay}


def build_manifest(out):
    """Only this fresh round's atomic state commits, never old orphan diagnostics."""
    out = Path(out)
    records = [json.loads(p.read_text()) for p in sorted((out / 'states').glob('*.json'))]
    groups = {split: sorted({r['group'] for r in records if r['split'] == split}) for split in ('train', 'calibration')}
    if set(groups['train']) & set(groups['calibration']):
        raise ValueError('World leakage across cost-head splits')
    result = {'records': records, 'groups': groups, 'split_rule': SPLIT_RULE,
              'states': len(records), 'labeled_alternatives': sum(r['labeled_alternatives'] for r in records),
              'missing_alternatives': sum(r['missing_alternatives'] for r in records),
              'stages': dict(Counter(r['stage'] for r in records)),
              'calibration_is_independent_full_model_test': False,
              'calibration_caveat': 'Frozen BC encoder previously saw some of these original worlds'}
    dump(out / 'manifest.json', result)
    return result
