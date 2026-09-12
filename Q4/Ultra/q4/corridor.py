"""Certified replacement of future stops by probes on an already chosen leg."""
import numpy as np

from .sector import ProbePolicy
from .shared import distance


class CorridorPolicy(ProbePolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v4_ring22_corridor_probes'
        self.counters.update(corridor_reviews=0, corridor_replacements=0)

    def choose(self, b):
        baseline = super().choose(b)
        delta = np.asarray(baseline.position)-b.position
        span = float(np.linalg.norm(delta))
        if self.completion_mode or span < 500 or not self.needed(b, b.position):
            # At a previously scanned location needed(current) is empty even
            # though intermediate locations can be useful. Test unresolved
            # channels separately in that case.
            if self.completion_mode or span < 500 or not any(
                    p.status == 'unresolved' for p in b.channels.values()):
                return baseline
        goals = {k: p for k, p in self.points.items() if self.needed(b, p)}
        goals.update({c: p.summary()[0] for c, p in b.channels.items() if p.status == 'detected'})
        _, before = self.route(b.position, goals)
        stations = [k for k in goals if k < 0 and distance(goals[k], baseline.position) > 1e-6]
        def off_leg(k):
            t = np.clip(float((np.asarray(goals[k])-b.position) @ delta)/(span*span), 0, 1)
            return distance(goals[k], tuple(np.asarray(b.position)+t*delta))
        stations = sorted(stations, key=off_leg)[:3]
        candidates = []
        for fractions in ((.5,), (1/3, 2/3)):
            probes = [tuple(map(float, np.asarray(b.position)+fraction*delta)) for fraction in fractions]
            scans = sum(6*len(self.needed(b, q)) for q in probes)
            for k in stations:
                proposed = {j: p for j, p in self.points.items() if j != k}
                changed = {j: p for j, p in goals.items() if j != k}
                keys = [k]+[min(self.points)-1] if len(probes) == 2 else [k]
                for key, point in zip(keys, probes):
                    proposed[key] = point
                    changed[key] = point
                _, after = self.route(b.position, changed)
                saved = (before-after)/5+6*len(self.needed(b, goals[k]))-scans
                if saved > 5 and self.needed(b, probes[0]):
                    candidates.append((saved, k, proposed, probes, before-after))
        audit = []
        for saving, key, proposed, probes, route_saved in sorted(candidates, key=lambda row: -row[0]):
            valid = self.valid_future(b, proposed)
            self.counters['corridor_reviews'] += 1
            audit.append({'station': key, 'on_leg_probes': probes, 'planned_saved_m': route_saved,
                          'predicted_saved_s': saving, 'full_certificate_valid': valid,
                          'probability_used_for_proof': False})
            if valid:
                self.points = proposed
                self.counters['corridor_replacements'] += 1
                original = self.records.pop()
                self.pending_cover_audit.extend(audit)
                return self.scan(b, probes[0], 'certified_corridor_probe',
                                 baseline_decision=original, corridor_audit=audit)
        self.records[-1]['corridor_audit'] = audit
        return baseline
