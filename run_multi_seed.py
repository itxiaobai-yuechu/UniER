"""Run one benchmark command for several seeds and summarize scalar metrics.

Use ``{seed}`` inside a command token when a program has a nonstandard seed
option. Every child also receives UNIER_SEED and PYTHONHASHSEED.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--cwd', default='.')
    parser.add_argument('--seed_arg', default=None,
                        help='standard seed option to append, for example --seed')
    parser.add_argument('--metric', action='append', default=[], metavar='NAME=REGEX',
                        help='repeatable scalar metric extractor; the first capture group is used')
    parser.add_argument('--output_dir', default='multi_seed_results')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return parser.parse_args()


def parse_metric_specs(specs):
    result = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'Metric specification must be NAME=REGEX, got {spec!r}.')
        name, pattern = spec.split('=', 1)
        result[name] = re.compile(pattern)
    return result


def main(args):
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('A command is required after --.')

    metric_patterns = parse_metric_specs(args.metric)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for seed in args.seeds:
        seed_was_substituted = any('{seed}' in token for token in command)
        seeded_command = [token.replace('{seed}', str(seed)) for token in command]
        if args.seed_arg and not seed_was_substituted:
            seeded_command.extend([args.seed_arg, str(seed)])

        env = os.environ.copy()
        env['UNIER_SEED'] = str(seed)
        env['PYTHONHASHSEED'] = str(seed)
        completed = subprocess.run(
            seeded_command,
            cwd=args.cwd,
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = output_dir / f'seed_{seed}.log'
        log_path.write_text(completed.stdout, encoding='utf-8')

        metrics = {}
        for name, pattern in metric_patterns.items():
            matches = pattern.findall(completed.stdout)
            if matches:
                value = matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1]
                metrics[name] = float(value)
        runs.append({
            'seed': seed,
            'returncode': completed.returncode,
            'command': seeded_command,
            'log': str(log_path),
            'metrics': metrics,
        })

    summary = {}
    for name in metric_patterns:
        values = [run['metrics'][name] for run in runs if name in run['metrics']]
        summary[name] = {
            'count': len(values),
            'mean': statistics.fmean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            'values': values,
        }

    report = {'runs': runs, 'summary': summary}
    report_path = output_dir / 'summary.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-seed logs and summary saved under {output_dir}.')
    if any(run['returncode'] != 0 for run in runs):
        raise SystemExit(1)


if __name__ == '__main__':
    main(parse_args())
