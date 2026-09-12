"""Restore every evaluated candidate from finalized paired-cost records."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np  # Old NPZ interchange only.
import torch
from torch.nn import functional as F

from .advantage import DATA_FORMAT, INPUTS
from .cost_learning import replay_snapshots, _network
from .features_v2 import FEATURE_VERSION
from .experiment import dump
from .relative import action_identity, four_candidates, CANDIDATE_RULE
from .relative_data import _match_frame, paired_targets, group_split, SPLIT_RULE, sha256


def restore_episode(job):
    torch.set_num_threads(1)
    diagnostic = json.loads(Path(job['diagnostics']).read_text())
    states = sorted([s for s in diagnostic['states'] if s.get('retained') is True and s.get('status') == 'complete'],
                    key=lambda s: s['step'])
    snapshots, _ = replay_snapshots(diagnostic['input_record'], [s['step'] for s in states], snapshot_features='v2')
    network = _network(job['bc_checkpoint'])
    rows = []
    with np.load(job['path'], allow_pickle=False) as old:
        for row, (state, snapshot) in enumerate(zip(states, snapshots)):
            if time.monotonic() >= job['hard_deadline']:
                raise TimeoutError('Round deadline during public replay')
            source, frame = snapshot['source_frame'], snapshot['frame']
            _match_frame(old, source, row)
            keys = [action_identity(a, s) for a, s in zip(frame.actions, frame.survey)]
            if keys != [action_identity(a, s) for a, s in zip(source.actions, source.survey)] or len(set(keys)) != len(keys):
                raise ValueError('v2 action identity mismatch')
            evaluated, means, errors, absolute = paired_targets(state)
            if frame.target != state['coverage']['teacher_index']:
                raise ValueError('Reference action mismatch')
            expected = torch.zeros(len(keys), dtype=torch.bool)
            expected[evaluated] = True
            if not torch.equal(torch.as_tensor(old['q_mask'][row, :len(keys)]), expected):
                raise ValueError('Saved evaluation mask mismatch')
            torch.testing.assert_close(torch.as_tensor(old['q_costs'][row, evaluated]).double(), absolute, atol=.002, rtol=1e-6)
            arrays = source.tensors()
            with torch.no_grad():
                encoded = network.encode(*(arrays[k][None] for k in INPUTS))
                scores = network.policy_scores(encoded, arrays['candidate_mask'][None])[0]
            deploy, reasons = four_candidates(frame, scores)
            targets, se = torch.full((len(keys),), math.nan), torch.full((len(keys),), math.nan)
            targets[evaluated], se[evaluated] = means.float(), errors.float()
            group = job['record']['source_group']
            identifier = hashlib.sha256(f'{group}:{state["step"]}'.encode()).hexdigest()[:24]
            path = Path(job['out']) / f'{identifier}.pt'
            payload = {**frame.tensors(), 'format': DATA_FORMAT, 'feature_version': FEATURE_VERSION,
                       'candidate_rule': CANDIDATE_RULE, 'bc_checkpoint_sha256': job['bc_sha256'],
                       'reference': torch.tensor(frame.target), 'target_s': targets, 'standard_error_s': se,
                       'evaluated': expected, 'teacher_scores': scores, 'deploy_indices': deploy,
                       'group': group, 'split': group_split(group), 'step': state['step'], 'stage': state['stage'],
                       'action_identities': keys}
            temp = path.with_suffix('.pt.tmp')
            torch.save(payload, temp)
            temp.replace(path)
            record = {'path': str(path), 'group': group, 'split': group_split(group), 'step': state['step'],
                      'stage': state['stage'], 'labeled_alternatives': len(evaluated) - 1,
                      'deploy_labeled_alternatives': sum(i in evaluated and i != frame.target for i in deploy),
                      'deploy_reasons': reasons, 'complete': True, 'input_npz': job['path'], 'input_row': row,
                      'bc_checkpoint_sha256': job['bc_sha256']}
            dump(path.with_suffix('.json'), record)
            rows.append(record)
    return rows


def manifest(out):
    out = Path(out)
    records = [json.loads(p.read_text()) for p in sorted((out / 'states').glob('*.json'))]
    groups = {split: sorted({r['group'] for r in records if r['split'] == split}) for split in ('train', 'calibration')}
    if set(groups['train']) & set(groups['calibration']):
        raise ValueError('World split leakage')
    result = dict(format=DATA_FORMAT, feature_version=FEATURE_VERSION, candidate_rule=CANDIDATE_RULE,
                  records=records, groups=groups, states=len(records), split_rule=SPLIT_RULE,
                  labeled_alternatives=sum(r['labeled_alternatives'] for r in records),
                  real_world_files_read=False, calibration_is_independent_full_model_test=False)
    dump(out / 'manifest.json', result)
    return result


def load_states(path, expected_bc_sha256):
    document = json.loads(Path(path).read_text())
    if document['format'] != DATA_FORMAT or document['feature_version'] != FEATURE_VERSION:
        raise ValueError('Wrong cost feature schema')
    rows, seen = [], set()
    for record in document['records']:
        row = torch.load(record['path'], map_location='cpu', weights_only=True)
        identity = (row['group'], row['step'])
        if identity in seen:
            raise ValueError('Duplicate public cost state')
        seen.add(identity)
        if row['split'] != group_split(row['group']) or row['group'] != record['group'] or row['split'] != record['split']:
            raise ValueError('World split mismatch')
        if row['bc_checkpoint_sha256'] != expected_bc_sha256 or row['candidate_rule'] != CANDIDATE_RULE:
            raise ValueError('Encoder/candidate provenance mismatch')
        if not torch.equal(row['evaluated'] & row['candidate_mask'], row['evaluated']):
            raise ValueError('Labels on padded actions')
        if not bool(row['evaluated'][row['reference']]) or row['target_s'][row['reference']] != 0:
            raise ValueError('Missing or nonzero baseline target')
        rows.append(row)
    return rows


def collate(rows, device='cpu'):
    n, k = max(len(r['nodes']) for r in rows), max(len(r['candidates']) for r in rows)
    result = {}
    for key in (*INPUTS, 'reference', 'target_s', 'standard_error_s', 'evaluated', 'teacher_scores'):
        values = []
        for row in rows:
            value = row[key]
            if key in ('nodes', 'candidates'):
                value = F.pad(value, (0, 0, 0, (n if key == 'nodes' else k) - len(value)))
            elif key in ('node_mask', 'candidate_mask', 'target_s', 'standard_error_s', 'evaluated', 'teacher_scores'):
                value = F.pad(value, (0, (n if key == 'node_mask' else k) - len(value)))
            values.append(value)
        result[key] = torch.stack(values).to(device)
    return result
