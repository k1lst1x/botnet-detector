"""Audit the entire downloaded shard and prepare account features without training."""

import argparse
import json
from pathlib import Path

from engine.shard_prepare import prepare_shard
from engine.model import validate_prepared

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/raw/twibot22-kaggle-shard7/tweet_7-004.json",
    )
    parser.add_argument("--official", type=Path, default=ROOT / "data/raw/twibot22")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data/raw/twibot22-kaggle-shard7/download-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/twibot22-shard7-training-v1",
    )
    parser.add_argument(
        "--work", type=Path, default=ROOT / "data/processed/twibot22-shard7-index-v1"
    )
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--max-posts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-index", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.source_manifest.read_text())["file_sha256"]
    result = prepare_shard(
        args.source,
        args.official,
        args.output,
        args.work,
        args.min_posts,
        args.max_posts,
        args.seed,
        args.resume_index,
        expected,
    )
    validation = validate_prepared(args.output)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "counts": result["counts"],
                "classes_by_split": result["classes_by_split"],
                "audit": result["audit"],
                "selection": result["selection"],
                "peak_rss_mb": result["peak_rss_mb"],
                "validation": validation,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
