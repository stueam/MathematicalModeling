"""Batched Q3 omnidirectional physics; all hidden truth lives in this module.

Positions are metres, angles degrees CCW from east, channels internally 0..19.
No HTTP, rendering, sleeping or geometry inference in the simulation hot path.
"""
from dataclasses import dataclass, asdict
import numpy as np

NO_SIGNAL, NEAR, DIRECTION, CLEARED, CLEAR_FAILED = range(5)


@dataclass
class Config:
    num_envs: int = 64
    channels: int = 20
    min_sources: int = 10
    max_sources: int = 16
    max_steps: int = 512
    max_virtual_s: float = 360000.0
    time_scale: float = 1000.0
    failure_penalty: float = 400.0
    shaping: float = 2.0
    layout: str = "uniform"
    radius_mode: str = "uniform"
    error_mode: str = "hash"

    def __post_init__(self):
        if not (1 <= self.min_sources <= self.max_sources <= self.channels <= 20):
            raise ValueError("Require 1 <= min_sources <= max_sources <= channels <= 20")
        if self.num_envs < 1 or self.max_steps < 1 or self.time_scale <= 0:
            raise ValueError("Invalid environment/time configuration")
        if self.layout not in ("uniform", "edge", "cluster"):
            raise ValueError("Unknown layout")
        if self.radius_mode not in ("uniform", "min", "max"):
            raise ValueError("Unknown radius mode")
        if self.error_mode not in ("hash", "smooth", "zero"):
            raise ValueError("Unknown error mode")

    def to_dict(self):
        return asdict(self)


class WorldBatch:
    def __init__(self, config: Config, seed=0):
        self.config = config
        n, c = config.num_envs, config.channels
        self.xy = np.zeros((n, c, 2), np.float64)
        self.radius = np.zeros((n, c), np.float64)
        self.present = np.zeros((n, c), bool)
        self.cleared = np.zeros((n, c), bool)
        self.position = np.zeros((n, 2), np.float64)
        self.channel = np.zeros(n, np.int64)
        self.time = np.zeros(n, np.float64)
        self.steps = np.zeros(n, np.int64)
        self.seed = np.zeros(n, np.uint64)
        self.move_time = np.zeros(n)
        self.switch_time = np.zeros(n)
        self.operation_time = np.zeros(n)
        self.reset(np.arange(n), np.arange(seed, seed + n))

    def reset(self, ids, seeds):
        for i, seed in zip(ids, seeds):
            rng = np.random.default_rng(int(seed))
            c = self.config.channels
            count = rng.integers(self.config.min_sources, self.config.max_sources + 1)
            self.present[i] = False
            self.present[i, rng.choice(c, count, replace=False)] = True
            theta = rng.uniform(0, 2 * np.pi, c)
            rho = 1800 * np.sqrt(rng.random(c))
            if self.config.layout == "edge":
                rho = rng.uniform(1650, 1800, c)
            self.xy[i] = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
            if self.config.layout == "cluster":
                center = self.xy[i, 0] * .7
                self.xy[i] = center + rng.normal(0, 150, (c, 2))
                lengths = np.linalg.norm(self.xy[i], axis=1)
                self.xy[i] *= np.minimum(1, 1799 / np.maximum(lengths, 1))[:, None]
            self.radius[i] = rng.uniform(1000, 1500, c)
            if self.config.radius_mode == "min":
                self.radius[i] = 1000
            elif self.config.radius_mode == "max":
                self.radius[i] = 1500
            self.cleared[i] = False
            self.position[i] = 0
            self.channel[i] = 0
            self.steps[i] = 0
            self.time[i] = self.move_time[i] = self.switch_time[i] = self.operation_time[i] = 0
            self.seed[i] = rng.integers(0, 2**63, dtype=np.uint64)

    def angle_error(self, positions, channels):
        if self.config.error_mode == "zero":
            return np.zeros(len(positions))
        if self.config.error_mode == "smooth":
            phase = (self.seed % np.uint64(100000)).astype(float) / 10000
            return np.sin(positions[:, 0] / 120 + positions[:, 1] / 170 + phase + channels)
        # A fixed per-world, per-channel, per-exact-position pseudo-random field.
        # Canonicalize signed zero. No new draw on a repeated measurement.
        p = np.array(positions, dtype=np.float64, copy=True)
        p[p == 0] = 0.
        bits = p.view(np.uint64)
        with np.errstate(over="ignore"):
            x = bits[:, 0] ^ (bits[:, 1] * np.uint64(0x9E3779B97F4A7C15))
            x ^= self.seed ^ ((channels.astype(np.uint64) + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
        return (x >> np.uint64(11)).astype(float) * (2. / 2**53) - 1.

    def step(self, positions, channels, is_clear):
        """Apply one validated action per environment. Validation is atomic."""
        n = self.config.num_envs
        positions = np.asarray(positions, dtype=np.float64)
        raw_channels = np.asarray(channels)
        raw_clear = np.asarray(is_clear)
        if positions.shape != (n, 2) or raw_channels.shape != (n,) or raw_clear.shape != (n,):
            raise ValueError("Invalid action shape")
        if not np.all(np.isfinite(positions)) or np.any(np.abs(positions) > 2_000_000):
            raise ValueError("Invalid coordinates; state was not advanced")
        if not np.all(np.isfinite(raw_channels)) or np.any(raw_channels != np.floor(raw_channels)):
            raise ValueError("Channels must be integral")
        channels = raw_channels.astype(np.int64)
        if np.any((channels < 0) | (channels >= self.config.channels)):
            raise ValueError("Channel out of range")
        if np.any((raw_clear != 0) & (raw_clear != 1)):
            raise ValueError("is_clear must be boolean")
        is_clear = raw_clear.astype(bool)
        if np.any(self.time >= self.config.max_virtual_s):
            raise RuntimeError("Episode exceeded virtual deadline; reset before another action")
        rows = np.arange(n)
        delta = self.xy[rows, channels] - positions
        distance = np.hypot(delta[:, 0], delta[:, 1])
        alive = self.present[rows, channels] & ~self.cleared[rows, channels]
        receive = alive & (distance <= self.radius[rows, channels])
        success = is_clear & alive & (distance <= 20)
        outcome = np.where(is_clear, np.where(success, CLEARED, CLEAR_FAILED),
                           np.where(~receive, NO_SIGNAL, np.where(distance <= 5, NEAR, DIRECTION)))
        angle = (np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) + self.angle_error(positions, channels)) % 360
        angle = np.round(angle, 2) % 360
        angle[outcome != DIRECTION] = np.nan
        move = np.linalg.norm(positions - self.position, axis=1) / 5
        switch = (~is_clear & (channels != self.channel)).astype(float)
        operation = np.where(is_clear & ~success, 3., 5.)
        # Local microsecond rounding approximation; official rounding details are not specified.
        duration = np.round(move + switch + operation, 6)
        self.time += duration
        self.move_time += move
        self.switch_time += switch
        self.operation_time += operation
        self.steps += 1
        self.position[:] = positions
        self.channel[~is_clear] = channels[~is_clear]
        self.cleared[rows[success], channels[success]] = True
        return outcome, angle, duration
