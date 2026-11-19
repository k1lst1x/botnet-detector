"""Export the proposed structural feature subset without fitting a model."""

import argparse
import json
from pathlib import Path

from engine.training_arrays import export_training_arrays

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=ROOT / "data/processed/twibot22-shard7-training-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/twibot22-shard7-structural-arrays-v1",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=ROOT / "experiments/structural-shard7-v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.experiment.read_text())
    result = export_training_arrays(args.prepared, args.output, config["base_features"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
