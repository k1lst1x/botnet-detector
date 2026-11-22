"""python -m engine.cli --help. Preparation and validation never train."""

import argparse
import json
from pathlib import Path

from .datasets import prepare_twibot, read_import
from .model import ModelService, evaluate_model, train_model, validate_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument(
        "--observation",
        type=Path,
        help="JSON Observation with verified coverage; optional",
    )
    validate = commands.add_parser("validate")
    validate.add_argument("directory", type=Path)
    train = commands.add_parser(
        "train", help="Explicitly run training; no other command invokes it"
    )
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--feature-set", choices=["profile", "portable"], default="profile"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--min-precision", type=float, default=0.9)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prepared", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--format", choices=["json", "jsonl", "csv"], required=True)
    predict.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .contracts import Observation

            observation = (
                Observation.model_validate_json(args.observation.read_text())
                if args.observation
                else None
            )
            result = prepare_twibot(args.source, args.output, args.limit, observation)
        elif args.command == "validate":
            result = validate_prepared(args.directory)
        elif args.command == "train":
            result = train_model(
                args.prepared,
                args.output,
                args.feature_set,
                args.seed,
                args.min_precision,
            )
        elif args.command == "evaluate":
            result = evaluate_model(args.prepared, args.model)
        else:
            from .inference import analyze_dataset

            data = read_import(args.input.read_text(), args.format)
            result = analyze_dataset(data, ModelService(args.model))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
