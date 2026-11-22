"""Prepare a combined JSON import and provenance report from repeated observations."""
import argparse
from pathlib import Path
from engine.contracts import Dataset
from engine.history import merge_observations
from engine.experiment import save_json
from engine.shard_index import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output directory must be new')
    datasets=[Dataset.model_validate_json(p.read_text()) for p in args.inputs]
    merged,report=merge_observations(datasets)
    report['inputs_sha256']=[{'path':str(p),'sha256':file_hash(p)} for p in args.inputs]
    args.output.mkdir(parents=True)
    save_json(args.output/'dataset.json',merged.model_dump(mode='json'))
    save_json(args.output/'coverage.json',report)
    print(f'Prepared {len(merged.accounts)} accounts and {len(merged.posts)} unique posts; coverage incomplete')


if __name__=='__main__':main()
