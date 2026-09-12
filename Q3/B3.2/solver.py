"""Stable strategy factory; receives observations through an action callback only."""
from joint_strategy import JointStrategy
from flexible_b3 import FlexibleB3
from peripheral_prior import PeripheralProposal

LATEST_CONFIG = dict(proposal_radius=1750, trigger_count=0, route_prior=True,
                     share_range=1000, cross_threshold=.1, trial=80, lateral=30,
                     sweeps=2, range_bias=0, reuse_sector=True, sectors=7, skip_far=True)
VALIDATED_CONFIG = dict(share_range=1000, cross_threshold=.1, trial=80, lateral=30,
                        sweeps=2, range_bias=0, reuse_sector=True, sectors=8)

def build_strategy(action, method="current"):
    if method == "current":
        from candidate import OptimizedB3
        return OptimizedB3(action, **LATEST_CONFIG)
    if method in ('mec', 'q2'):
        from q2_lookahead import MECProposal, Q2Proposal
        cls = MECProposal if method == 'mec' else Q2Proposal
        return cls(action, **LATEST_CONFIG)
    if method == "latest":
        return PeripheralProposal(action, **LATEST_CONFIG)
    if method == "b2":
        return JointStrategy(action, stop_at_max=True)
    if method == "validated":
        return FlexibleB3(action, **VALIDATED_CONFIG)
    raise ValueError(f"Unknown method: {method}")
