"""Prepare capped, deterministic per-account features from an audited shard."""

from collections import Counter
import json
from pathlib import Path
import resource
import time

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import AccountSnapshot, Observation, Post
from .features import BASE_FEATURES, FEATURE_NAMES, FEATURE_VERSION, extract_features
from .shard_index import HashReader, index_shard


def prepare_shard(
    source,
    official,
    output,
    work,
    min_posts=20,
    max_posts=100,
    seed=42,
    resume=False,
    expected_sha256=None,
):
    if not 1 <= min_posts <= max_posts:
        raise ValueError("Require 1 <= min_posts <= max_posts")
    source, official, output, work = map(Path, (source, official, output, work))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared output must be empty")
    db, audit = index_shard(source, official, work, seed, resume, expected_sha256)
    start = time.monotonic()
    writers = {}
    try:
        output.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                ("external_user_id", pa.string()),
                ("label", pa.int8()),
                ("split", pa.string()),
                ("observed_posts", pa.int64()),
                ("available_posts", pa.int64()),
            ]
            + [(n, pa.float64()) for n in FEATURE_NAMES]
        )
        counts = Counter()
        classes = Counter()
        coverage = Counter()
        buffers = {s: [] for s in ("train", "val", "test")}
        for split in buffers:
            writers[split] = pq.ParquetWriter(
                output / f"{split}.parquet", schema, compression="zstd"
            )

        def flush(split):
            if buffers[split]:
                writers[split].write_table(
                    pa.Table.from_pylist(buffers[split], schema=schema)
                )
                buffers[split].clear()

        expected = db.execute(
            "SELECT count(*) FROM author_counts WHERE n>=?", (min_posts,)
        ).fetchone()[0]
        if not expected:
            raise ValueError("No accounts meet minimum post count")
        db.execute("UPDATE accounts SET profile_seen=0")
        db.commit()
        rows = 0
        profiles = 0
        used_posts = 0
        with (official / "user.json").open("rb") as stream:
            profile_reader = HashReader(stream)
            for raw in ijson.items(profile_reader, "item", use_float=True):
                uid = str(raw["id"])
                meta = db.execute(
                    "SELECT a.label,a.split,a.profile_seen,coalesce(c.n,0) FROM accounts a LEFT JOIN author_counts c ON c.author=a.id WHERE a.id=?",
                    (uid,),
                ).fetchone()
                if meta is None:
                    raise ValueError("Profile absent from official labels/split")
                label, split, seen, available = meta
                if seen:
                    raise ValueError("Duplicate official profile")
                profiles += 1
                db.execute("UPDATE accounts SET profile_seen=1 WHERE id=?", (uid,))
                if available < min_posts:
                    continue
                metrics = raw.get("public_metrics") or {}
                account = AccountSnapshot(
                    platform="twitter",
                    external_user_id=uid,
                    username=raw.get("username"),
                    created_at=raw.get("created_at"),
                    followers=metrics.get("followers_count"),
                    following=metrics.get("following_count"),
                    biography=raw.get("description"),
                    has_avatar=bool(raw["profile_image_url"])
                    if raw.get("profile_image_url") is not None
                    else None,
                )
                posts = [
                    Post.model_validate_json(row[0])
                    for row in db.execute(
                        "SELECT payload FROM tweets WHERE author=? ORDER BY sample_key,id LIMIT ?",
                        (uid, max_posts),
                    )
                ]
                if len(posts) != min(available, max_posts):
                    raise ValueError("Sampling count mismatch")
                observation = Observation(
                    start=min(p.created_at for p in posts),
                    end=max(p.created_at for p in posts),
                    source="third_party_single_shard_sample",
                    complete=False,
                    truncated=available > max_posts,
                )
                features = extract_features(account, posts, observation)
                buffers[split].append(
                    {
                        "external_user_id": uid,
                        "label": label,
                        "split": split,
                        "observed_posts": len(posts),
                        "available_posts": available,
                        **features,
                    }
                )
                rows += 1
                used_posts += len(posts)
                counts[split] += 1
                classes[(split, label)] += 1
                coverage.update(n for n in BASE_FEATURES if features[n] is not None)
                if len(buffers[split]) >= 1024:
                    flush(split)
                    db.commit()
                if rows % 10000 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "features",
                                "rows": rows,
                                "expected": expected,
                                "profiles_scanned": profiles,
                                "elapsed_s": round(time.monotonic() - start, 1),
                                "peak_rss_mb": round(
                                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                    / 1024,
                                    1,
                                ),
                            }
                        ),
                        flush=True,
                    )
        if db.execute("SELECT id FROM accounts WHERE profile_seen=0").fetchone():
            raise ValueError("Official profiles missing from user.json")
        if rows != expected:
            raise ValueError("Eligible author/profile mismatch")
        for split in buffers:
            flush(split)
        db.commit()
    finally:
        for writer in writers.values():
            writer.close()
        db.close()
    index_identity = json.loads((work / "state.json").read_text())["identity"]
    report = {
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "rows": rows,
        "counts": {s: counts[s] for s in buffers},
        "classes_by_split": {
            s: {"human": classes[(s, 0)], "bot": classes[(s, 1)]} for s in buffers
        },
        "used_posts": used_posts,
        "behavior_available": True,
        "observation_complete": False,
        "training_performed": False,
        "scope": "third-party single-shard account subset; not full TwiBot-22",
        "selection": {
            "minimum_unique_posts": min_posts,
            "maximum_sampled_posts": max_posts,
            "seed": seed,
            "policy": "bottom-k BLAKE2b(seed:canonical_post_id), tie-break by post_id; independent of label, split and source order",
        },
        "feature_coverage": {
            n: {"observed": coverage[n], "ratio": coverage[n] / rows}
            for n in BASE_FEATURES
        },
        "source": {
            "sha256": audit["source_sha256"],
            "bytes": audit["source_bytes"],
            "normalization_version": "native-tweet-v1",
            "official_inputs": {
                "label.csv": index_identity["labels_sha256"],
                "split.csv": index_identity["splits_sha256"],
                "user.json": profile_reader.digest.hexdigest(),
            },
        },
        "audit": audit,
        "elapsed_feature_s": round(time.monotonic() - start, 1),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "warnings": [
            "One third-party shard introduces selection bias; not full-benchmark results.",
            "Posting timelines are incomplete; temporal rate and regularity features remain missing.",
            "Randomly capped observed posts require equivalent target sampling; no Threads quality established.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
