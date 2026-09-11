"""Replay one held-out scenario. Truth is used exclusively by the final plot."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .physics import Config
from .env import RadioEnv, heuristic_actions
from .model import ActorCritic,tensor_obs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--seed',type=int,default=1_000_000)
    p.add_argument('--out',default='reports/trajectory')
    args=p.parse_args()
    torch.set_num_threads(2)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=Config(**{**ckpt['config'],'num_envs':1})
    model=ActorCritic(ckpt['hidden']);model.load_state_dict(ckpt['model']);model.eval()
    env=RadioEnv(config,args.seed)
    truth=env.world.xy[0,env.world.present[0]].copy()
    trace=[]; positions=[np.zeros(2)]; clears=[]
    for step in range(config.max_steps):
        with torch.no_grad():
            dist,_=model(*tensor_obs(env.obs,'cpu'))
            action=int(dist.probs.argmax(-1).item())
        position=env.belief.targets[0,action].copy()
        _,_,done,episodes=env.step(np.array([action]))
        trace.append({'step':step+1,'channel':action//4+1,'operation':'clear' if action%4==3 else 'measure',
                      'position':position.tolist()})
        positions.append(position)
        if action%4==3:
            clears.append(position)
        if done[0]:
            summary=episodes[0];break
    positions=np.array(positions)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'summary':summary,'actions':trace},indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    ax.add_patch(plt.Circle((0,0),1800,fill=False,color='#94a3b8',linestyle='--'))
    ax.plot(positions[:,0],positions[:,1],color='#2563eb',alpha=.65,lw=1,label='Executed route')
    ax.scatter(truth[:,0],truth[:,1],marker='*',s=140,color='#ef4444',label='Sources (audit only)',zorder=4)
    ax.scatter(0,0,c='black',s=50,label='Start',zorder=5)
    if clears:
        pts=np.array(clears)
        ax.scatter(pts[:,0],pts[:,1],facecolors='none',edgecolors='#059669',s=65,label='Clear attempts',zorder=3)
    ax.set(aspect='equal',xlabel='East / m',ylabel='North / m',
           title=f"PPO / seed {args.seed} / {summary['cleared']}/{summary['sources']} cleared\n"
                 f"{summary['virtual_s']/60:.1f} virtual min / {summary['steps']} actions")
    ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
