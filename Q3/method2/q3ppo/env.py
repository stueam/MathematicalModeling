"""Observable candidate-action environment with SAME-STEP automatic reset.

No oracle completion signal is exposed: exit only after channel certificates,
or the known upper bound of distinct sources has been cleared.
"""
from collections import deque
import numpy as np
from .physics import Config, WorldBatch
from .belief import BeliefBatch


class RadioEnv:
    def __init__(self, config=None, seed=0):
        self.config = config or Config()
        self.world = WorldBatch(self.config, seed)
        self.belief = BeliefBatch(self.config.num_envs, self.config.channels)
        self.next_seed = seed + self.config.num_envs
        self.episode_seeds = np.arange(seed, self.next_seed, dtype=np.int64)
        self.completed = deque(maxlen=10000)
        self.obs = self.observe()

    def observe(self):
        w = self.world
        return self.belief.observe(w.position, w.channel, w.time, w.steps, self.config.max_steps)

    def potential(self):
        b = self.belief
        detected = b.status == 1
        return self.config.shaping * (
            .4 * detected.sum(1) + (b.status == 2).sum(1)
            + .2 * (b.status == 3).sum(1)
            + .4 * (detected * (1 - np.minimum(b.radii / 1800, 1))).sum(1))

    def step(self, actions):
        actions = np.asarray(actions)
        n = self.config.num_envs
        if actions.shape != (n,) or np.any(actions != np.floor(actions)):
            raise ValueError("One integral action per environment required")
        actions = actions.astype(np.int64)
        if np.any((actions < 0) | (actions >= self.config.channels * 4)):
            raise ValueError("Invalid action index")
        rows = np.arange(n)
        if not np.all(self.belief.mask[rows, actions]):
            raise ValueError("Masked action; no environment advanced")
        positions = self.belief.targets[rows, actions].copy()
        channels, is_clear = actions // 4, actions % 4 == 3
        before = self.potential()
        result, angle, duration = self.world.step(positions, channels, is_clear)
        self.belief.update(positions, channels, result, angle)
        known_clears = (self.belief.status == 2).sum(1)
        certified_done = np.all(self.belief.status >= 2, axis=1) | (known_clears >= self.config.max_sources)
        budget_end = (self.world.steps >= self.config.max_steps) | (self.world.time >= self.config.max_virtual_s)
        done = certified_done | budget_end
        after = self.potential()
        after[done] = 0.  # Absorbing-state potential: telescopes with gamma=1.
        reward = -duration / self.config.time_scale + after - before
        reward[budget_end & ~certified_done] -= self.config.failure_penalty
        episodes = []
        for i in np.flatnonzero(done):
            # Truth is used only for audit metrics, never for state/action/stop logic.
            all_cleared = bool(np.all(~self.world.present[i] | self.world.cleared[i]))
            item = {
                "seed": int(self.episode_seeds[i]), "success": bool(certified_done[i] and all_cleared),
                "certified": bool(certified_done[i]), "all_cleared": all_cleared,
                "sources": int(self.world.present[i].sum()), "cleared": int(known_clears[i]),
                "virtual_s": float(self.world.time[i]), "steps": int(self.world.steps[i]),
                "move_s": float(self.world.move_time[i]), "switch_s": float(self.world.switch_time[i]),
                "operation_s": float(self.world.operation_time[i]),
                "reason": "certified" if certified_done[i] else "budget",
            }
            if certified_done[i] and not all_cleared:
                raise AssertionError("Invalid completion certificate")
            episodes.append(item)
            self.completed.append(item)
        ids = np.flatnonzero(done)
        if len(ids):
            seeds = np.arange(self.next_seed, self.next_seed + len(ids), dtype=np.int64)
            self.next_seed += len(ids)
            self.episode_seeds[ids] = seeds
            self.world.reset(ids, seeds)
            self.belief.reset(ids)
        self.obs = self.observe()
        if not self.obs["mask"].any(1).all():
            raise RuntimeError("Candidate generator produced no legal action")
        return self.obs, reward.astype(np.float32), done, episodes


def heuristic_actions(obs):
    """Observable, deterministic low-cost baseline; NOT method1 or an oracle."""
    f, mask = obs["candidates"], obs["mask"]
    cost = f[:, :, 10]
    clear = f[:, :, 3] > .5
    known = f[:, :, 4] > .5
    safe = f[:, :, 15] > .5
    # Clear certified regions; otherwise move to centroid and measure.
    score = cost.copy()
    score += np.where(known & ~clear, -.05, 0)
    score += np.where(clear & safe, -1., 0)
    score += np.where(clear & ~safe, 100., 0)
    score += np.where((f[:, :, 1] + f[:, :, 2] > .5) & known, .15, 0)
    score[~mask] = 1e9
    return score.argmin(1)


def random_actions(obs, rng):
    score = rng.random(obs["mask"].shape)
    score[~obs["mask"]] = -1
    return score.argmax(1)
