"""Local-only experiment: paired complete continuations at discovery departures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import multiprocessing
from pathlib import Path
import time

import numpy as np

import run
from bayes_tsp.cached import CachedRefinementPolicy
from bayes_tsp.refinement import RefinementConfig
from bayes_tsp.rollout_review import RolloutJob, ReviewWorldSampler, evaluate_task, capture_continuation_state
from bayes_tsp.shared import Action, distance, load


class ReviewPolicy(CachedRefinementPolicy):
    implementation = 'experimental_paired_complete_v6_review'

    def __init__(self, config, pool, worlds=4, max_reviews=3):
        super().__init__(config)
        self.pool, self.worlds, self.max_reviews = pool, worlds, max_reviews
        self.sampler = ReviewWorldSampler(seed=872961, pool_size=512)
        self.reviewed_discoveries = set()
        self.reviews = []

    def choose(self, b):
        action = super().choose(b)
        record = self.records[-1]
        signature = tuple(c for c, p in b.channels.items() if p.status in ('detected', 'cleared'))
        if (record['status'] != 'route_decision' or distance(b.position, action.position) < 100
                or signature in self.reviewed_discoveries or len(self.reviews) >= self.max_reviews):
            return action
        rows = sorted(record.get('candidates', []), key=lambda row: row['score_s'])
        if not rows:
            return action
        reference = next((row for row in rows if Action(**row['action']) == action), None)
        if reference is None:
            return action
        menu, tasks = [reference], {reference['task']}
        for row in rows:
            if row['task'] not in tasks and row['score_s'] <= reference['score_s']+120:
                menu.append(row)
                tasks.add(row['task'])
            if len(menu) >= 4:
                break
        if len(menu) < 2:
            return action
        self.reviewed_discoveries.add(signature)
        started = time.monotonic()
        review = {'step': b.steps, 'discovered_channels': signature, 'worlds': self.worlds,
                  'candidates': menu, 'selected_index': 0, 'status': 'running'}
        self.reviews.append(review)
        try:
            continuation_state = capture_continuation_state(self)
            review['continuation_state'] = continuation_state
            worlds = self.sampler.sample(b, self.worlds)
            batches = []
            coverage = self._coverage_tasks(b)
            for row in menu:
                key = int(row['task'].split(':')[1])
                batches.append(tuple(coverage[key]['members']) if key < 0 else ())
            jobs = [RolloutJob(b, world, Action(**row['action']), batch, self.config,
                               max_steps=4000, timeout_s=60., continuation_state=continuation_state)
                    for world in worlds for row, batch in zip(menu, batches)]
            results = list(self.pool.map(evaluate_task, jobs))
            review['rollouts'] = results
            if not all(r['completed'] for r in results):
                review['status'] = 'discarded_incomplete_batch'
                return action
            costs = np.asarray([r['cost'] for r in results]).reshape(self.worlds, len(menu))
            means = costs.mean(axis=0)
            winner = int(np.argmin(means))
            # A paired practical margin reduces switching on small sample noise.
            gains = costs[:, 0]-costs[:, winner]
            stderr = float(gains.std(ddof=1)/np.sqrt(self.worlds)) if self.worlds > 1 else 0.
            if means[0]-means[winner] < max(15., .5*stderr):
                winner = 0
            review.update(status='completed', means_s=means.tolist(), selected_index=winner,
                          paired_gain_s=float(means[0]-means[winner]),
                          paired_standard_error_s=stderr)
            chosen = Action(**menu[winner]['action'])
            if winner:
                self.batch_channels = list(batches[winner])
                self.batch_position = chosen.position if self.batch_channels else None
                self.after_clear = (chosen.position, chosen.channel) if chosen.kind == 'clear' else None
                self.active_target = chosen.channel if not batches[winner] else None
                record.update(selected=asdict(chosen), selected_move_m=distance(b.position, chosen.position),
                              selected_task=menu[winner]['task'], active_target=self.active_target,
                              selected_score_s=menu[winner]['score_s'],
                              departure_review='paired_complete_continuations')
            record['completion_review'] = {k: v for k, v in review.items() if k != 'rollouts'}
            return chosen
        except Exception as exc:
            review.update(status='fallback', error=f'{type(exc).__name__}: {exc}')
            return action
        finally:
            review['elapsed_s'] = time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--reviews', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    out = run.new_output()
    config = RefinementConfig(stable_routing=False)
    run.dump(out/'config.json', {**vars(args), 'config': asdict(config), 'code_sha256': run.code_manifest(),
                                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    summaries = []
    manifest = {'status': 'running', 'planned': args.rounds, 'completed': []}
    run.dump(out/'batch.json', manifest)
    simulator = load('simulator')
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for seed in range(args.seed, args.seed+args.rounds):
                env = simulator.LocalSimulator(simulator.generate_world(seed))
                policy = ReviewPolicy(config, pool, args.worlds, args.reviews)
                b = run.make_belief(policy)
                started = time.monotonic()
                b.deadline = started+1200
                actions = []
                error = None
                try:
                    for step in range(4000):
                        if b.done() or time.monotonic() >= b.deadline:
                            break
                        action = policy.choose(b)
                        response = env.execute(action, f'real-{step}')
                        b.apply(action, response, f'real-{step}')
                        actions.append({'action': asdict(action), 'response': response})
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                summary = {'seed': seed, **env._world.score(), 'certified_complete': b.done(),
                    'virtual_time_s': b.virtual_time, 'real_time_s': time.monotonic()-started,
                    'movement_m': env.costs['move_s']*5, 'costs': env.costs,
                    'reviews': len(policy.reviews), 'review_changes': sum(r.get('selected_index', 0)>0 for r in policy.reviews),
                    'error': error}
                summaries.append(summary)
                for suffix, data in [('actions', actions), ('decisions', policy.records), ('reviews', policy.reviews), ('summary', summary)]:
                    run.dump(out/f'{seed}-{suffix}.json', data)
                manifest['completed'].append(seed)
                run.dump(out/'batch.json', manifest)
                run.dump(out/'summary.json', summaries)
                print(summary, flush=True)
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'interrupted'
        raise
    finally:
        run.dump(out/'batch.json', manifest)
        print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
