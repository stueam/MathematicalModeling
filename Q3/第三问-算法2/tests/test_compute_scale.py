from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pomcp
import pytest
from pomcp.config import Config, COMPUTE_FIELDS


def test_triple_only_computation_preserves_default_and_workers():
    original = Config()
    cfg = original.scaled(3)
    for field in COMPUTE_FIELDS:
        assert getattr(cfg, field) == getattr(original, field)*3
    assert (cfg.seed, cfg.workers, cfg.tail) == (2026, 16, 'completion')
    assert original.particles == 3000 and original.budget_s == 5
    assert asdict(original.scaled(1)) == asdict(original)
    assert cfg.candidates*cfg.coarse_worlds+cfg.finalists*cfg.fine_worlds == 3456


@pytest.mark.parametrize('factor', [0, -1, 1.5, True])
def test_invalid_compute_scaling_rejected(factor):
    with pytest.raises(ValueError):
        Config().scaled(factor)
