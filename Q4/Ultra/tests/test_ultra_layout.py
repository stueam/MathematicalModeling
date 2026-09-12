from q4.compact import make_policy
from q4.policy import Config
from q4.sector import ring_stations, ring22_stations
from q4.core import DOMAIN
from q4.coverage import full_exclusion

def test_default_probes_uses_s21():
    policy = make_policy('probes', Config())
    assert policy.implementation == 'ultra_s21_route_probes_v1'
    assert len(policy.points) == 21
    assert tuple(policy.points.values()) == ring_stations()
    assert DOMAIN.difference(full_exclusion(ring_stations())).is_empty

def test_baseline_layout_retained_for_comparison():
    assert len(ring22_stations()) == 22
