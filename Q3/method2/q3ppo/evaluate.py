import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from .physics import Config
from .env import RadioEnv, heuristic_actions, random_actions
from .model import ActorCritic, tensor_obs


def evaluate(config, policy, episodes=100, seed=1_000_000, checkpoint=None, device='cpu'):
    # Initially launch exactly the requested seeds; ignore auto-reset replacement
    # episodes. Every policy therefore evaluates the SAME scenario set, even when
    # fast episodes finish first. Avoid selection bias from first N completions.
    config = replace(config,num_envs=min(episodes,32))
    model = None
    if policy == 'ppo':
        data = torch.load(checkpoint,map_location=device,weights_only=False)
        model = ActorCritic(data['hidden']).to(device)
        model.load_state_dict(data['model']); model.eval()
    results = []
    rng = np.random.default_rng(seed+10_000_000)
    start = time.perf_counter()
    for offset in range(0,episodes,32):
        count = min(32,episodes-offset)
        env = RadioEnv(replace(config,num_envs=count),seed+offset)
        wanted = set(range(seed+offset,seed+offset+count))
        for _ in range(config.max_steps):
            if policy == 'heuristic':
                actions = heuristic_actions(env.obs)
            elif policy == 'random':
                actions = random_actions(env.obs,rng)
            else:
                with torch.no_grad():
                    dist,_ = model(*tensor_obs(env.obs,device))
                    actions = dist.probs.argmax(-1).cpu().numpy()
            _,_,_,ends = env.step(actions)
            for e in ends:
                if e['seed'] in wanted:
                    results.append(e); wanted.remove(e['seed'])
            if not wanted:
                break
        if wanted:
            raise AssertionError('Missing evaluation episodes')
    success = [r['virtual_s'] for r in results if r['success']]
    summary = {'policy':policy,'checkpoint':checkpoint,'episodes':len(results),
               'seed_start':seed,'seed_end':seed+episodes-1,
               'success_rate':float(np.mean([r['success'] for r in results])),
               'mean_success_virtual_s':float(np.mean(success)) if success else None,
               'p90_success_virtual_s':float(np.quantile(success,.9)) if success else None,
               'mean_cleared':float(np.mean([r['cleared'] for r in results])),
               'mean_steps':float(np.mean([r['steps'] for r in results])),
               'wall_s':time.perf_counter()-start, 'config':config.to_dict()}
    return summary,sorted(results,key=lambda r:r['seed'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/full.json')
    p.add_argument('--policy',choices=['ppo','heuristic','random'],default='heuristic')
    p.add_argument('--checkpoint')
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=1_000_000)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--layout',choices=['uniform','edge','cluster'])
    p.add_argument('--radius-mode',choices=['uniform','min','max'])
    p.add_argument('--error-mode',choices=['hash','smooth','zero'])
    p.add_argument('--out',default='reports/evaluation.json')
    args=p.parse_args()
    if args.episodes < 1 or (args.policy=='ppo' and not args.checkpoint):
        p.error('Require positive episodes, and checkpoint for PPO')
    torch.set_num_threads(2)
    config=Config(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    for name in ['layout','radius_mode','error_mode']:
        value=getattr(args,name)
        if value:
            setattr(config,name,value)
    summary,records=evaluate(config,args.policy,args.episodes,args.seed,args.checkpoint,args.device)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'episodes':records},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
