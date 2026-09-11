"""Experimental complete coverage routes with movable sector boundaries.

Each alternative partitions every unresolved channel's full outer geometry.
Changing the partition is a planning operation only: actual feedback remains
the sole source of exclusion/clearance evidence. The unrotated V6 plan always
participates in the same cost comparison.
"""
from dataclasses import dataclass
import math

import shapely
from shapely.geometry import Polygon

from .refinement import RefinementConfig, RefinementPolicy


@dataclass(frozen=True)
class AdaptiveSearchConfig(RefinementConfig):
    sector_rotations: int = 4
    adaptive_sector_counts: tuple = ()
    rotate_with_sources: bool = True
    partition_min_saving_s: float = 0.


class AdaptiveSearchPolicy(RefinementPolicy):
    implementation = 'bayes_tsp_adaptive_sector_boundaries'
    _mask_stride = 1 << 13

    def __init__(self, config=None):
        super().__init__(config or AdaptiveSearchConfig())
        if not 1 <= self.config.sector_rotations <= 8:
            raise ValueError('sector_rotations must be 1..8')
        if self.config.partition_min_saving_s < 0:
            raise ValueError('partition_min_saving_s must be nonnegative')
        counts = tuple(dict.fromkeys((self.config.sectors,)+tuple(self.config.adaptive_sector_counts)))
        if any(not isinstance(n, int) or not 6 <= n <= 12 for n in counts):
            raise ValueError('adaptive_sector_counts must contain integers in 6..12')
        self._grids = []
        for count in counts:
            for rotation in range(self.config.sector_rotations):
                offset = rotation*math.tau/(count*self.config.sector_rotations)
                shapes, centers = [], []
                for j in range(count):
                    angle = offset+j*math.tau/count
                    endpoints = (angle-math.pi/count-1e-9, angle+math.pi/count+1e-9)
                    shapes.append(Polygon([(0., 0.)]+[
                        (5000*math.cos(a), 5000*math.sin(a)) for a in endpoints]))
                    centers.append((1100*math.cos(angle), 1100*math.sin(angle)))
                self._grids.append({'shapes': shapes, 'centers': centers,
                    'count': count, 'angle_deg': math.degrees(offset),
                    'task_key': None, 'tasks': None})
        self._grid_index = 0
        self._plan_logs = {}
        self.partition_decisions = 0
        self.rotated_partition_decisions = 0

    def _activate_grid(self, index):
        old = self._grids[self._grid_index]
        old['task_key'], old['tasks'] = self._task_key, self._tasks_cache
        grid = self._grids[index]
        self.sector_shapes, self.fixed_centers = grid['shapes'], grid['centers']
        self._task_key, self._tasks_cache = grid['task_key'], grid['tasks']
        self._grid_index = index

    def _activate_plan(self, plan):
        self._activate_grid(plan['partition_index'])
        self._last_task_log = self._plan_logs[plan['mask']]

    def _coverage_tasks(self, b):
        tasks = super()._coverage_tasks(b)
        if self._grid_index == 0:
            return tasks
        for task in tasks.values():
            if task.get('adaptive_outer_margin_m'):
                continue
            # Overlaying a rotated ray can round intersections inward on the
            # arena boundary. Keep the entire task an outer approximation;
            # this is never an exclusion region or an actual belief update.
            task['members'] = {c: part.buffer(1e-7) for c, part in task['members'].items()}
            task['region'] = shapely.union_all(list(task['members'].values()))
            task['vertices'] = shapely.get_coordinates(task['region'].convex_hull)
            task['adaptive_outer_margin_m'] = 1e-7
        return tasks

    def _plans(self, b, posts, source_points):
        plans = []
        self._plan_logs = {}
        grids = self._grids
        if ((posts and not self.config.rotate_with_sources)
                or len(b.cleared)+len(posts) == 16
                or not any(c.status == 'unresolved' for c in b.channels.values())):
            grids = grids[:1]
        for index, grid in enumerate(grids):
            self._activate_grid(index)
            for plan in super()._plans(b, posts, source_points):
                plan['mask'] += index*self._mask_stride
                plan['partition_index'] = index
                plan['partition_sectors'] = grid['count']
                plan['partition_angle_deg'] = grid['angle_deg']
                plans.append(plan)
                self._plan_logs[plan['mask']] = {**self._last_task_log,
                    'partition_index': index, 'partition_sectors': grid['count'],
                    'partition_angle_deg': grid['angle_deg']}
        reference = next(plan['proxy'] for plan in plans if plan['partition_index'] == 0)
        plans = [plan for plan in plans if plan['partition_index'] == 0
                 or plan['proxy'] <= reference-self.config.partition_min_saving_s]
        # Pure coverage uses the first plan; source actions compare all plans.
        plans.sort(key=lambda plan: (plan['proxy'], plan['partition_index']))
        self._activate_plan(plans[0])
        return plans

    def _select_task(self, b, posts, stable, pending, points):
        selected, candidates, route, future = super()._select_task(b, posts, stable, pending, points)
        index = selected['mask']//self._mask_stride
        self._activate_grid(index)
        self._last_task_log = self._plan_logs[selected['mask']]
        self._joint_details.update(self._last_task_log)
        self._joint_details['partition_candidates'] = [
            {'mask': key, 'partition_index': row['partition_index'],
             'partition_angle_deg': row['partition_angle_deg'],
             'partition_sectors': row['partition_sectors']}
            for key, row in self._plan_logs.items()]
        self.partition_decisions += 1
        self.rotated_partition_decisions += int(index != 0)
        return selected, candidates, route, future
