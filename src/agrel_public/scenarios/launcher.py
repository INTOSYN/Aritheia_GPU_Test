"""Standalone public scenario entry: local execution only, no protected core."""
import argparse
from pathlib import Path
from ..common import save_json
from .registry import configs, config_id


def main(name):
    p = argparse.ArgumentParser(description=name + ": two frozen scientific task configurations")
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.steps <= 10000:
        p.error('steps must be 1..10000')
    if args.out.exists():
        p.error('new output directory required')
    from .engine import run_config
    args.out.mkdir(parents=True)
    rows = []
    for cfg in configs(name):
        row = run_config(cfg, args.device, args.precision,
                         args.out / config_id(cfg).replace('/', '_'), steps=args.steps)
        rows.append(row); save_json(args.out / 'scenarios.json', rows)
        print(name, cfg, row['status'], row['metric'], flush=True)
