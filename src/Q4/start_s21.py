"""Run the S21 probes policy locally, check coverage, or join Windows practice."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true', help='Check coverage without network access')
    mode.add_argument('--local', action='store_true', help='Complete a local simulated mission')
    mode.add_argument('--connect', action='store_true', help='Connect to the Windows practice simulator')
    parser.add_argument('--seed', type=int, default=800, help='Local world seed')
    parser.add_argument('--rounds', type=int, default=1, help='Number of practice rounds, 1..100')
    parser.add_argument('--resume-ready-practice', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 100:
        parser.error('rounds must be 1..100')
    if (args.local or args.self_test) and (args.rounds != 1 or args.resume_ready_practice):
        parser.error('Practice options require practice mode')
    if not args.local and args.seed != 800:
        parser.error('--seed requires --local')
    if args.self_test:
        from q4.core import Belief
        from q4.sector import ProbePolicy

        policy = ProbePolicy()
        action = policy.choose(Belief())
        if len(policy.points) != 21:
            raise RuntimeError('S21 requires exactly 21 certified stations')
        print(
            json.dumps(
                {
                    'coverage': 'passed',
                    'stations': len(policy.points),
                    'implementation': policy.implementation,
                    'first_action': action.payload(),
                    'network_used': False,
                },
                ensure_ascii=False,
            )
        )
    elif args.local:
        import run

        run.main(['local', '--policy', 'probes', '--seed', str(args.seed)])
    else:
        import practice_windows

        options = ['--policy', 'probes', '--rounds', str(args.rounds)]
        if args.connect:
            options.append('--connect')
        if args.resume_ready_practice:
            options.append('--resume-ready-practice')
        practice_windows.main(options)


if __name__ == '__main__':
    main()
