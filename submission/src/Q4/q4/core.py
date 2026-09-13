"""Q4 public geometry: negative readings never directly subtract a 1000 m disk."""

from dataclasses import dataclass, field

from .coverage import added_exclusion, full_exclusion
from .shared import GeometryError, core, disk, point_key

DOMAIN = core.DOMAIN


@dataclass
class Channel(core.Channel):
    @property
    def negatives(self):
        return tuple(o.action.position for o in self.history if o.result == 'no_signal')

    def update(self, obs):
        if self.status in ('cleared', 'absent_certified'):
            if obs.result in ('direction', 'near', 'success'):
                raise GeometryError('Positive feedback conflicts with completed channel')
            return
        if obs in self.history:
            return
        a, result = obs.action, obs.result
        if a.kind == 'measure':
            old = [
                o
                for o in self.history
                if o.action.kind == 'measure' and point_key(o.action.position) == point_key(a.position)
            ]
            if old and old[0] != obs:
                raise GeometryError('Fixed active source has conflicting same-point feedback')
        candidate = self.clone()
        region, status = candidate.region, candidate.status
        if result == 'direction':
            region = region.intersection(core.bearing_wedge(a.position, obs.bearing))
            region = region.intersection(disk(a.position, 1500, True)).difference(disk(a.position, 5))
            status = 'detected'
        elif result == 'near':
            region = region.intersection(disk(a.position, 5, True))
            status = 'detected'
        elif result == 'no_signal':
            exclusion = added_exclusion(a.position, tuple(sorted(candidate.negatives)))
            region = region.difference(exclusion)
            if not region.is_empty and region.area < 1.0:
                # Re-evaluate a real certificate, NOT an area cutoff. Different
                # orders of polygon subtraction can retain machine-size slivers.
                # Only the union of actual same-channel negative observations
                # may remove them; nonempty residuals of any size remain.
                actual = tuple(sorted(set(candidate.negatives + (a.position,))))
                region = region.difference(full_exclusion(actual))
        elif result == 'no_target_in_range':
            region = region.difference(disk(a.position, 20))
        elif result == 'success':
            region = region.intersection(disk(a.position, 20, True))
            status = 'cleared'
        else:
            raise ValueError('Unknown Q4 feedback')
        if region.is_empty:
            if status != 'unresolved':
                raise GeometryError('Empty detected support; cannot certify absence')
            status = 'absent_certified'
        candidate.region, candidate.status = region, status
        candidate.history.append(obs)
        if a.kind == 'measure':
            candidate.measured.add(point_key(a.position))
        candidate.revision += 1
        candidate._summary = None
        self.__dict__.update(candidate.__dict__)


@dataclass
class Belief(core.Belief):
    channels: dict = field(default_factory=lambda: {c: Channel() for c in range(1, 21)})

    @property
    def known(self):
        return {c for c, p in self.channels.items() if p.status in ('detected', 'cleared')}

    def apply(self, action, response, request_id):
        positive = (
            response.get('measure_result') in ('direction', 'near')
            or response.get('clear_result') == 'success'
        )
        if (
            response.get('accepted') is True
            and positive
            and action.channel not in self.known
            and len(self.known) >= 16
        ):
            raise GeometryError('More than 16 observed source channels')
        return super().apply(action, response, request_id)
