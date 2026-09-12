"""Verify published model/code identities and every included mission audit."""
import hashlib
import json
from pathlib import Path

import torch

from nnq4.advantage import AdvantageNetwork
from nnq4.advantage_data import load_states


root=Path(__file__).resolve().parent
release=json.loads((root/'release.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in release['source_runtime_sha256'].items():
    assert sha(root/name)==digest,name
for name,digest in release['models_sha256'].items():
    assert sha(root/'models'/name)==digest,name
model=AdvantageNetwork.load(root/'models/best.pt')
assert model.metadata['calibration']['cost_labels']==152
data=load_states(root/'training_data/manifest.json',release['models_sha256']['bc-v1.pt'])
assert len(data)==213
assert not {r['group'] for r in data if r['split']=='train'}&{r['group'] for r in data if r['split']=='calibration'}
rows=json.loads((root/'evidence/summary.json').read_text())
assert len(rows)==96
for r in rows:
    audit=json.loads((root/r['folder']/'physical-audit.json').read_text())
    assert r['complete'] and r['audit_ok'] and r['public_complete']
    assert audit['ok'] and audit['complete'] and audit['certified_complete'] and not audit['errors']
    assert audit['spent_virtual_s']==r['virtual_time_s']
    assert abs(sum(r['costs'].values())-r['virtual_time_s'])<1e-6
    assert abs(sum(r['executor_costs_s'].values())-r['virtual_time_s'])<1e-6
selected=[r for r in rows if r['mode']=='finetuned_advantage']
score=sum(r['virtual_time_s'] for r in selected)/sum(r['source_count'] for r in selected)
assert len(selected)==32 and abs(score-442.62508287841194)<1e-8
print(json.dumps(dict(source_and_model_hashes_verified=True,complete_audited_episodes=96,
                      states=213,paired_cost_labels=sum(int(r['evaluated'].sum())-1 for r in data),
                      best_s_per_source=score,local_simulator_only=True),indent=2))
