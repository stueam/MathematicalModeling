"""Complete paired root-action comparison on public-history hypothetical worlds."""
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict
import multiprocessing
import time

import numpy as np

from .posterior import QuadratureError
from .rollout import continue_world
from .sampling import WorldSampler
from .sector import ProbePolicy
from .shared import Action, distance


class MonteCarloPolicy(ProbePolicy):
    def __init__(self, config=None, mc_worlds=4, mc_workers=16, mc_candidates=3, mc_budget=30., **unused):
        super().__init__(config)
        if min(mc_worlds, mc_workers) < 1 or mc_candidates < 2 or mc_budget <= 0:
            raise ValueError('Invalid paired Monte Carlo configuration')
        self.implementation = 'q4_v5_complete_paired_mc'
        self.worlds, self.workers, self.candidate_cap, self.budget = mc_worlds, mc_workers, mc_candidates, mc_budget
        self.rng = np.random.default_rng(0)
        self.sampler = WorldSampler(self.model)
        self.pool = None
        self.abandoned = []
        self.counters.update(mc_decisions=0, mc_changed=0, mc_effective_worlds=0,
                             mc_effective_rollouts=0, mc_effective_steps=0,
                             mc_discarded_rollouts=0, mc_discarded_steps=0,
                             mc_failed_batches=0, mc_sampling_failures=0)

    def collect_abandoned(self):
        remaining = []
        for future in self.abandoned:
            if not future.done():
                remaining.append(future)
                continue
            if not future.cancelled():
                try:
                    self.counters['mc_discarded_steps'] += future.result()['steps']
                except Exception:
                    pass
        self.abandoned = remaining

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.collect_abandoned()

    def action_batch(self, b, action):
        if action.kind == 'measure' and b.channels[action.channel].status == 'unresolved':
            return action.position
        return None

    def required_gain(self,standard_error):
        return 2.

    def choose(self, b):
        self.collect_abandoned()
        baseline = super().choose(b)
        record = self.records[-1]
        if self.completion_mode or distance(b.position, baseline.position) < 200:
            return baseline
        candidates = [baseline]
        used_tasks = set()
        ordered = sorted(record.get('candidates', []), key=lambda row: row['score_s'])
        for row in ordered:
            a = row['action']; action = Action(a['kind'], tuple(a['position']), a['channel'])
            if action == baseline:
                used_tasks.add(row['task'])
        for pass_index in range(2):
            for row in ordered:
                a = row['action']; action = Action(a['kind'], tuple(a['position']), a['channel'])
                if action in candidates or len(candidates) >= self.candidate_cap:
                    continue
                if pass_index == 0 and row['task'] in used_tasks:
                    continue
                if pass_index == 1 and min(distance(action.position, x.position) for x in candidates) < 80:
                    continue
                candidates.append(action); used_tasks.add(row['task'])
        if len(candidates) < 2:
            record['mc_review'] = {'accepted_batch':False, 'reason':'no_alternative_candidate'}
            return baseline
        started = time.monotonic()
        available = min(self.budget, b.deadline-started-self.config.reserve_s-2)
        if available < 2:
            record['mc_review'] = {'accepted_batch':False, 'reason':'actual_deadline_reserve'}
            return baseline
        self.counters['mc_decisions'] += 1
        try:
            worlds = [self.sampler.sample(b, self.rng) for _ in range(self.worlds)]
        except QuadratureError as exc:
            self.counters['mc_sampling_failures'] += 1
            record['mc_review'] = {'accepted_batch':False, 'reason':'sampling_failed', 'error':str(exc)}
            return baseline
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                                            mp_context=multiprocessing.get_context('spawn'))
        futures=[]
        for i,world in enumerate(worlds):
            for j,action in enumerate(candidates):
                futures.append(self.pool.submit(continue_world, {
                    'belief':b, 'world':world, 'config':self.config, 'points':self.points,
                    'batch':self.action_batch(b, action), 'shared_checked':self.shared_checked,
                    'completion_mode':self.completion_mode, 'action':action,
                    'world_index':i, 'candidate_index':j, 'wall_limit_s':max(1.,available)}))
        done,pending=wait(futures,timeout=max(0.,available-(time.monotonic()-started)))
        results=[]
        for future in done:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({'complete':False,'cost_s':None,'steps':0,'error':str(exc)})
        accepted = not pending and len(results)==self.worlds*len(candidates) and all(r['complete'] for r in results)
        detail={'accepted_batch':accepted, 'worlds_per_action':self.worlds,
                'sampling_prior':getattr(self.sampler,'prior_description','independent_Bernoulli_source_and_type_counts'),
                'sampled_directional_counts':[world.score()['directional_count'] for world in worlds],
                'candidate_count':len(candidates), 'planned_rollouts':len(futures),
                'same_worlds_for_all_candidates':True, 'full_remaining_task':True,
                'terminal_value_estimate':False, 'suffix_cost_cache':False,
                'real_time_s':time.monotonic()-started, 'rollouts':results,
                'pending_rollouts_at_cutoff':len(pending), 'candidates':[asdict(a) for a in candidates]}
        if not accepted:
            self.counters['mc_failed_batches'] += 1
            self.counters['mc_discarded_rollouts'] += len(futures)
            self.counters['mc_discarded_steps'] += sum(r['steps'] for r in results)
            for future in pending:
                future.cancel()
            self.abandoned.extend(pending)
            record['mc_review']=detail
            return baseline
        costs=np.empty((self.worlds,len(candidates)))
        for row in results:
            costs[row['world_index'],row['candidate_index']]=row['cost_s']
        counts=np.array([world.score()['source_count'] for world in worlds])
        ratios=(costs+b.virtual_time)/counts[:,None]
        mean=ratios.mean(axis=0)
        selected=int(np.argmin(mean))
        gain=float(mean[0]-mean[selected])
        differences=(costs[:,0]-costs[:,selected])/counts
        stderr=float(differences.std(ddof=1)/np.sqrt(self.worlds)) if self.worlds>1 else None
        threshold=self.required_gain(stderr)
        if gain <= threshold:
            selected=0
        self.counters['mc_effective_worlds'] += self.worlds
        self.counters['mc_effective_rollouts'] += len(results)
        self.counters['mc_effective_steps'] += sum(r['steps'] for r in results)
        self.counters['mc_changed'] += int(selected!=0)
        detail.update(mean_remaining_s=costs.mean(axis=0).tolist(),
                      mean_total_s_per_sampled_source=mean.tolist(), sampled_source_counts=counts.tolist(),
                      chosen_candidate=selected, predicted_best_gain_s_per_source=gain,
                      paired_standard_error_s_per_source=stderr, minimum_gain_s_per_source=threshold,
                      confidence_note='Finite-sample heuristic; not a strict confidence guarantee.')
        self.batch=self.action_batch(b,candidates[selected])
        if selected==0:
            record['mc_review']=detail
            return baseline
        self.records.pop()
        return self.record(b,candidates[selected],'complete_paired_mc',mc_review=detail,baseline_decision=record)


class StableMonteCarloPolicy(MonteCarloPolicy):
    def __init__(self,config=None,**options):
        super().__init__(config,**options)
        if self.worlds<2:
            raise ValueError('Paired uncertainty screening requires at least two worlds')
        from .joint_sampling import JointWorldSampler
        self.sampler=JointWorldSampler(self.model)
        self.implementation='q4_v7_joint_count_complete_mc_guard_massfix'

    def required_gain(self,standard_error):
        # Selection from several candidates and finite-sample standard errors
        # make this a screening heuristic, not a strict confidence guarantee.
        return 2.+1.5*standard_error
