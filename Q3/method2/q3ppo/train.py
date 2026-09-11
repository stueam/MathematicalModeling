"""Standalone clipped PPO, gamma=1, GAE, masked actions, checkpointing and CSV.

Training is from environment rewards only. No demonstrations or rollout teacher.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .physics import Config
from .env import RadioEnv
from .model import ActorCritic, tensor_obs, gae


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full.json")
    p.add_argument("--out", default="runs/full")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=.01)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", help="Warm-start weights and optimizer; fresh environments and RNG streams")
    p.add_argument("--max-seconds", type=float, default=0, help="Stop after an update, save final checkpoint")
    return p


def main():
    args = parser().parse_args()
    if args.steps < 1 or args.rollout < 1 or args.minibatch < 1 or args.epochs < 1:
        raise ValueError("Training sizes must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "progress.csv").exists():
        raise FileExistsError("Output already contains a run; choose a new --out")
    start = time.perf_counter()
    env = RadioEnv(config, args.seed)
    model = ActorCritic(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    prior_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        prior_steps = ckpt["total_steps"]
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    manifest = {"args": vars(args), "environment": config.to_dict(), "torch": torch.__version__,
                "numpy": np.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "prior_steps": prior_steps, "objective": "undiscounted time + finite failure penalty; potential shaping"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n, t, a = config.num_envs, args.rollout, config.channels*4
    candidates = torch.empty((t, n, a, 20), device=device)
    globals_ = torch.empty((t, n, 8), device=device)
    masks = torch.empty((t, n, a), dtype=torch.bool, device=device)
    actions = torch.empty((t, n), dtype=torch.long, device=device)
    logprobs = torch.empty((t, n), device=device)
    rewards, dones, values = [torch.empty((t, n), device=device) for _ in range(3)]
    total = 0
    fields = ["steps", "wall_s", "sps", "episodes", "success_rate_100", "mean_virtual_s_success_100",
              "mean_cleared_100", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
              "collection_s", "optimization_s"]
    log = (out / "progress.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log, fieldnames=fields)
    writer.writeheader()
    episodes_log = (out / "episodes.jsonl").open("w", encoding="utf-8")
    num_episodes = 0

    def save(name):
        path = out / name
        temp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "hidden": args.hidden, "config": config.to_dict(), "total_steps": prior_steps + total,
                    "torch_rng": torch.get_rng_state(), "args": vars(args)}, temp)
        temp.replace(path)

    print(json.dumps(manifest), flush=True)
    training_start = time.perf_counter()
    updates = math.ceil(args.steps / (n*t))
    try:
        for update in range(updates):
            collect_start = time.perf_counter()
            model.eval()
            for step in range(t):
                ob = tensor_obs(env.obs, device)
                # CPU tensors may alias env observation arrays; snapshot before stepping.
                candidates[step].copy_(ob[0]); globals_[step].copy_(ob[1]); masks[step].copy_(ob[2])
                with torch.no_grad():
                    dist, value = model(*ob)
                    act = dist.sample()
                    actions[step], logprobs[step], values[step] = act, dist.log_prob(act), value
                _, rew, done, episodes = env.step(act.cpu().numpy())
                rewards[step] = torch.as_tensor(rew, device=device)
                dones[step] = torch.as_tensor(done, device=device)
                for item in episodes:
                    episodes_log.write(json.dumps(item) + "\n")
                num_episodes += len(episodes)
            with torch.no_grad():
                _, last_value = model(*tensor_obs(env.obs, device))
                adv, returns = gae(rewards, values, dones, last_value)
            collection_s = time.perf_counter() - collect_start
            opt_start = time.perf_counter()
            model.train()
            b = n*t
            fc, fg, fm = candidates.flatten(0,1), globals_.flatten(0,1), masks.flatten(0,1)
            fa, fl, fr = actions.flatten(), logprobs.flatten(), returns.flatten()
            fadv = adv.flatten()
            fadv = (fadv - fadv.mean()) / (fadv.std(unbiased=False) + 1e-8)
            metrics = []
            early_stop = False
            for epoch in range(args.epochs):
                order = torch.randperm(b, device=device)
                for begin in range(0, b, args.minibatch):
                    ix = order[begin:begin+args.minibatch]
                    dist, value = model(fc[ix], fg[ix], fm[ix])
                    logratio = dist.log_prob(fa[ix]) - fl[ix]
                    ratio = logratio.exp()
                    loss_policy = torch.maximum(-fadv[ix]*ratio, -fadv[ix]*ratio.clamp(.8,1.2)).mean()
                    loss_value = .5 * (value-fr[ix]).square().mean()
                    entropy = dist.entropy().mean()
                    loss = loss_policy + .5*loss_value - args.entropy*entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                    with torch.no_grad():
                        kl = ((ratio-1)-logratio).mean()
                        clipped = ((ratio-1).abs() > .2).float().mean()
                        metrics.append(torch.stack((loss_policy, loss_value, entropy, kl, clipped)).detach())
                    if kl.item() > .03:
                        early_stop = True
                        break
                if early_stop:
                    break
            if device.type == "cuda":
                torch.cuda.synchronize()
            optimization_s = time.perf_counter() - opt_start
            total += b
            recent = list(env.completed)[-100:]
            successful = [e["virtual_s"] for e in recent if e["success"]]
            m = torch.stack(metrics).mean(0).cpu().tolist()
            wall = time.perf_counter() - training_start
            row = dict(zip(fields, [total, wall, total/wall, num_episodes,
                np.mean([e["success"] for e in recent]) if recent else float("nan"),
                np.mean(successful) if successful else float("nan"),
                np.mean([e["cleared"] for e in recent]) if recent else float("nan"),
                *m, collection_s, optimization_s]))
            writer.writerow(row); log.flush(); episodes_log.flush()
            print(f"steps={total} sps={total/wall:.0f} success100={row['success_rate_100']:.3f} "
                  f"clear100={row['mean_cleared_100']:.2f} time100={row['mean_virtual_s_success_100']:.1f} "
                  f"collect={collection_s:.2f}s opt={optimization_s:.2f}s", flush=True)
            if (update+1) % 10 == 0:
                save("latest.pt")
            if args.max_seconds and wall >= args.max_seconds:
                break
    except KeyboardInterrupt:
        print("Interrupted; saving last parameters", flush=True)
    finally:
        save("final.pt")
        log.close(); episodes_log.close()
        (out / "timing.json").write_text(json.dumps({"steps_this_run":total,
            "wall_including_setup_s":time.perf_counter()-start,
            "peak_gpu_allocated_mb":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
