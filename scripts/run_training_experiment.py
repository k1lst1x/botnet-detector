"""Run training and held-out evaluation as separate explicit commands."""
import argparse
import json
from pathlib import Path
from engine.experiment import train, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'evaluate'])
    parser.add_argument('--config', type=Path, default=Path('experiments/structural-shard7-v1.json'))
    parser.add_argument('--output', type=Path, default=Path('media/models/experiments/structural-shard7-v1-run1'))
    args = parser.parse_args()
    result = train(args.config, args.output) if args.phase == 'train' else evaluate(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
