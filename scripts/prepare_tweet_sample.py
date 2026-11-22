"""Build a bounded real-tweet subset and preserve official account splits.

python -m scripts.prepare_tweet_sample --tweets 10000
The subset is not the full benchmark and does not establish timeline completeness.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import ijson

from engine.datasets import prepare_twibot
from engine.model import validate_prepared

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets", type=int, default=10000)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/raw/twibot22-kaggle-shard7/tweet_7-004.json",
    )
    parser.add_argument("--profiles", type=Path, default=ROOT / "data/raw/twibot22")
    parser.add_argument(
        "--sample", type=Path, default=ROOT / "data/raw/twibot22-shard7-sample"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/twibot22-shard7-sample"
    )
    args = parser.parse_args()
    if not 1 <= args.tweets <= 100000:
        raise ValueError("Choose between 1 and 100000 tweets for a bounded sample")
    if args.sample.exists() and any(args.sample.iterdir()):
        raise ValueError("Sample directory must be empty")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Prepared directory must be empty")
    with (args.profiles / "label.csv").open(newline="") as stream:
        labels = {r["id"]: r["label"] for r in csv.DictReader(stream)}
    selected = []
    authors = set()
    unknown = 0
    with args.source.open("rb") as stream:
        for index, raw in enumerate(ijson.items(stream, "item", use_float=True)):
            if index >= args.tweets:
                break
            author = str(raw.get("author_id") or "")
            if author not in labels and author.isdigit() and f"u{author}" in labels:
                author = f"u{author}"
            if author not in labels:
                unknown += 1
                continue
            raw["author_id"] = author
            selected.append(raw)
            authors.add(author)
    if not selected:
        raise ValueError("No tweets matched official account identities")
    args.sample.mkdir(parents=True, exist_ok=True)
    with (args.sample / "tweet_7-004.json").open("w") as stream:
        json.dump(selected, stream)
    del selected
    profile_ids = set()
    with (
        (args.profiles / "user.json").open("rb") as source,
        (args.sample / "user.json").open("w") as output,
    ):
        output.write("[")
        first = True
        for raw in ijson.items(source, "item", use_float=True):
            if raw["id"] not in authors:
                continue
            if not first:
                output.write(",")
            json.dump(raw, output)
            first = False
            profile_ids.add(raw["id"])
        output.write("]")
    if profile_ids != authors:
        raise ValueError("Some tweet authors lack official profiles")
    split_counts = Counter()
    classes = Counter()
    for filename, field in [("label.csv", "label"), ("split.csv", "split")]:
        with (
            (args.profiles / filename).open(newline="") as source,
            (args.sample / filename).open("w", newline="") as output,
        ):
            writer = csv.DictWriter(output, fieldnames=["id", field])
            writer.writeheader()
            for row in csv.DictReader(source):
                if row["id"] in authors:
                    writer.writerow(row)
                    if field == "split":
                        split_counts[row[field]] += 1
                    else:
                        classes[row[field]] += 1
    prepared = prepare_twibot(args.sample, args.output)
    validation = validate_prepared(args.output)
    provenance = {
        "scope": "first N tweets from a third-party single shard; not full TwiBot-22",
        "requested_tweets": args.tweets,
        "unmatched_author_tweets": unknown,
        "matched_accounts": len(authors),
        "classes": dict(classes),
        "official_splits_preserved": True,
        "split_counts": dict(split_counts),
        "source_file": str(args.source.relative_to(ROOT))
        if args.source.is_relative_to(ROOT)
        else args.source.name,
        "author_id_normalization": "numeric native IDs mapped only when official u-prefixed label exists",
        "complete_timeline": False,
        "training_performed": False,
        "prepared_rows": prepared["rows"],
        "validation": validation,
    }
    (args.sample / "sample-manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
