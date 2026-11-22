"""Independent all-record reference-schema validation for the first full index run."""

from collections import Counter
import json
from pathlib import Path
import time

import ijson

from engine.shard_index import HashReader, reference_types

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/raw/twibot22-kaggle-shard7/tweet_7-004.json"
    manifest = json.loads((source.parent / "download-manifest.json").read_text())
    records = 0
    types = Counter()
    unknown = 0
    start = time.monotonic()
    with source.open("rb") as stream:
        reader = HashReader(stream)
        for raw in ijson.items(reader, "item", use_float=True):
            refs = reference_types(raw.get("referenced_tweets"))
            records += 1
            if refs is None:
                unknown += 1
            else:
                types.update(refs)
            if records % 1000000 == 0:
                print(f"reference validation: {records} records", flush=True)
    assert reader.digest.hexdigest() == manifest["file_sha256"], "SHA256 mismatch"
    result = {
        "records": records,
        "source_sha256": reader.digest.hexdigest(),
        "unknown_reference_metadata": unknown,
        "known_reference_types": dict(types),
        "invalid_reference_records": 0,
        "elapsed_s": round(time.monotonic() - start, 1),
    }
    path = ROOT / "data/processed/twibot22-shard7-index-v1/reference-validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
