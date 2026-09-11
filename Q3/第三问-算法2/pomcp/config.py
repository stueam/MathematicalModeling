from dataclasses import dataclass, replace


COMPUTE_FIELDS = ('particles', 'candidates', 'coarse_worlds', 'fine_worlds',
                  'finalists', 'horizon', 'fine_horizon', 'budget_s')


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    particles: int = 3000
    candidates: int = 12
    coarse_worlds: int = 16
    fine_worlds: int = 48
    finalists: int = 4
    horizon: int = 6
    fine_horizon: int = 8
    budget_s: float = 5.
    workers: int = 16
    tail: str = 'completion'

    def scaled(self, factor=1):
        """Scale computation only; never seeds, workers, priors, or physical rules."""
        if type(factor) is not int or factor < 1:
            raise ValueError('Compute scale must be a positive integer')
        return replace(self, **{name: getattr(self, name)*factor for name in COMPUTE_FIELDS})

    def __post_init__(self):
        if min(self.particles, self.coarse_worlds, self.horizon, self.workers) < 1:
            raise ValueError('Particle, world, horizon and worker counts must be positive')
        if not 2 <= self.finalists <= self.candidates or self.fine_worlds < self.coarse_worlds:
            raise ValueError('Invalid racing configuration')
        if self.fine_horizon < self.horizon or self.budget_s <= 0:
            raise ValueError('Invalid horizon or time budget')
        if self.tail not in ('geometric', 'completion'):
            raise ValueError('Unknown tail estimator')
