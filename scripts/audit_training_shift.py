"""Inspect only train/val feature distributions and derive train-only class weight."""
import argparse
import json
from pathlib import Path
from engine.shift_audit import audit_arrays
from engine.experiment import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/structural-shard7-v1.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/reports/2026-11-20-training-shift.json'))
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit output already exists')
    config=json.loads(args.config.read_text())
    report=audit_arrays(config['numeric_arrays_directory'],config['arrays_manifest_sha256'])
    save_json(args.output,report)
    print(json.dumps({'train':report['train'],'val':report['val'],'train_positive_weight':report['train_positive_weight']},indent=2))


if __name__=='__main__':
    main()
