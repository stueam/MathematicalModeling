"""Limit concurrent known-source work while search coverage is unfinished.

Deferred channels remain in the full route and its work estimate. Admission is
persistent until a channel is cleared; it is not reselected at every move.
"""
from .localization import guaranteed_clear
from .sector import ProbePolicy


class CapacityPolicy(ProbePolicy):
    def __init__(self, capacity=4, config=None):
        if capacity not in (2, 4, 6):
            raise ValueError('Supported localization capacities are 2, 4 and 6')
        super().__init__(config)
        self.capacity = capacity
        self.active = set()
        self.deferred_since = {}
        self.capacity_audit = {}
        self.implementation = f'q4_github_capacity{capacity}_v1'
        self.counters.update(capacity_reviews=0, capacity_blocked_reviews=0,
                             capacity_peak_active=0, capacity_admissions=0)

    def eligible_tasks(self, b, goals, route):
        known = {c for c in goals if c > 0}
        searching = any(k < 0 for k in goals)
        self.active &= known
        released = []
        if searching:
            admissions = [c for c in route if c in known and c not in self.active]
            admissions = admissions[:max(0, self.capacity-len(self.active))]
            self.active.update(admissions)
            self.counters['capacity_admissions'] += len(admissions)
            # Certified clears spend no additional localization budget.
            safe = {c for c in known if guaranteed_clear(b, c)}
            eligible = self.active | safe | {k for k in goals if k < 0}
        else:
            eligible = set(goals)
        deferred = known-eligible
        for c in deferred:
            self.deferred_since.setdefault(c, (b.virtual_time, b.steps))
        for c in list(self.deferred_since):
            if c not in deferred:
                start, step = self.deferred_since.pop(c)
                released.append({'channel': c, 'wait_s': b.virtual_time-start, 'wait_steps': b.steps-step})
        self.counters['capacity_reviews'] += 1
        self.counters['capacity_blocked_reviews'] += bool(deferred)
        if searching:
            self.counters['capacity_peak_active'] = max(self.counters['capacity_peak_active'], len(self.active))
        self.capacity_audit = {'limit': self.capacity, 'searching': searching,
            'active': sorted(self.active), 'deferred': sorted(deferred), 'released': released,
            'all_known_retained_in_route': sorted(known), 'deferred_work_counted': True}
        return eligible

    def record(self, b, action, reason, **extra):
        if reason in ('joint_route', 'bayes_route'):
            extra['capacity'] = self.capacity_audit
        return super().record(b, action, reason, **extra)
