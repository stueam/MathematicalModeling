"""Q4 candidate-node learning. All execution here uses local simulated worlds."""
import argparse
from pathlib import Path
from nnq4.experiment import new_run, execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'benchmark', 'train', 'cost-labels', 'relative-round'])
    p.add_argument('--seed', type=int, default=280000000)
    p.add_argument('--rounds', type=int, default=32)
    p.add_argument('--train', type=int, default=192)
    p.add_argument('--validation', type=int, default=32)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--checkpoint')
    p.add_argument('--data')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--replay-data', action='append', default=[])
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--value-weight', type=float, default=.1)
    p.add_argument('--bc-weight', type=float, default=1.)
    p.add_argument('--cost-sample-weight', type=float, default=1.)
    p.add_argument('--replay-weight', type=float, default=1.)
    p.add_argument('--q-weight', type=float, default=.1)
    p.add_argument('--pair-weight', type=float, default=.1)
    p.add_argument('--cost-policy-weight', type=float, default=0.)
    p.add_argument('--cost-temperature', type=float, default=100.)
    p.add_argument('--states', type=int, default=128)
    p.add_argument('--worlds', type=int, default=2)
    p.add_argument('--candidates', type=int, default=4)
    p.add_argument('--collect-neural', action='store_true')
    p.add_argument('--selection', choices=['policy', 'q'], default='policy')
    p.add_argument('--menu', choices=['full', 'external'], default='full')
    p.add_argument('--layout', choices=['ring22', 's21', 's25', 'tri25'], default='ring22')
    p.add_argument('--features', choices=['v1', 'v2'], default='v1')
    p.add_argument('--macro-budget', type=int, default=160)
    p.add_argument('--pure', action='store_true')
    p.add_argument('--teacher-control', action='store_true')
    p.add_argument('--scenario', default='uniform')
    p.add_argument('--error-mode', default='iid')
    args = p.parse_args()
    if args.mode == 'relative-round':
        from nnq4.relative_round import launch, DEFAULT_CHECKPOINT, DEFAULT_DATA
        _, control = launch(args.checkpoint or DEFAULT_CHECKPOINT, args.data or DEFAULT_DATA,
                            workers=args.workers, seed=args.seed, device=args.device)
        if control['status'] != 'complete':
            raise SystemExit(1)
        return
    if args.mode in ('train', 'cost-labels') and not args.data:
        p.error(args.mode+' requires --data')
    if args.mode == 'collect' and args.collect_neural and not args.checkpoint:
        p.error('--collect-neural requires --checkpoint')
    if args.mode == 'benchmark' and not args.checkpoint and not args.teacher_control:
        p.error('benchmark requires --checkpoint or --teacher-control')
    out = new_run(args.mode, vars(args))
    print('OUTPUT='+str(out.resolve()), flush=True)
    if args.mode == 'train':
        from nnq4.training import train
        train(args.data, out/'training', epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
              initial=args.checkpoint, device=args.device, replay_data=args.replay_data,
              learning_rate=args.learning_rate, value_weight=args.value_weight, q_weight=args.q_weight,
              pair_weight=args.pair_weight, cost_policy_weight=args.cost_policy_weight,
              cost_temperature=args.cost_temperature, bc_weight=args.bc_weight,
              cost_sample_weight=args.cost_sample_weight, replay_weight=args.replay_weight)
        return
    if args.mode == 'cost-labels':
        from nnq4.cost_learning import collect_cost_labels
        collect_cost_labels(args.data, out/'labels', states=args.states, worlds=args.worlds,
                            candidates=args.candidates, workers=args.workers, seed=args.seed,
                            checkpoint=args.checkpoint)
        return
    common = {'macro_budget': args.macro_budget, 'allow_fallback': not args.pure,
              'scenario': args.scenario, 'error_mode': args.error_mode, 'selection': args.selection,
              'menu': args.menu, 'layout': args.layout, 'features': args.features}
    if args.mode == 'collect':
        mode = 'neural' if args.collect_neural else 'teacher'
        jobs = [{**common, 'seed': args.seed+i, 'mode': mode, 'checkpoint': args.checkpoint, 'collect': True, 'split': 'train'}
                for i in range(args.train)]
        jobs += [{**common, 'seed': args.seed+10000+i, 'mode': mode, 'checkpoint': args.checkpoint, 'collect': True, 'split': 'validation'}
                 for i in range(args.validation)]
    else:
        jobs = [{**common, 'seed': args.seed+i, 'mode': mode, 'checkpoint': args.checkpoint}
                for i in range(args.rounds) for mode in ('probes', 'teacher' if args.teacher_control else 'neural')]
    execute(jobs, out, args.workers)


if __name__ == '__main__':
    main()
