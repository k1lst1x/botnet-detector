import hashlib
import json

import pyarrow.parquet as pq
import pytest

from engine.shard_prepare import prepare_shard


def fixtures(tmp_path, reverse=False, conflicting=False):
    official = tmp_path / "official"
    official.mkdir()
    (official / "label.csv").write_text(
        "id,label\n"
        + "".join(f"u{i},{'bot' if i % 2 == 0 else 'human'}\n" for i in range(1, 7))
    )
    (official / "split.csv").write_text(
        "id,split\n"
        + "".join(
            f"u{i},{['train', 'val', 'test'][(i - 1) // 2]}\n" for i in range(1, 7)
        )
    )
    (official / "user.json").write_text(
        json.dumps(
            [
                {
                    "id": f"u{i}",
                    "username": f"name{i}",
                    "public_metrics": {"followers_count": i, "following_count": 1},
                }
                for i in range(1, 7)
            ]
        )
    )
    tweets = [
        {
            "id": f"t{i}-{j}",
            "author_id": str(i),
            "created_at": f"2022-01-01T0{j}:00:00Z",
            "text": f"content {j % 2} https://example.org/{j % 2}",
            "referenced_tweets": [],
        }
        for i in range(1, 7)
        for j in range(4)
    ]
    tweets.append(dict(tweets[0], **({"text": "conflict"} if conflicting else {})))
    tweets.append(
        {
            "id": "t-unknown",
            "author_id": "999",
            "created_at": "2022-01-01T00:00:00Z",
            "text": "unknown author",
        }
    )
    if reverse:
        tweets.reverse()
    source = tmp_path / "tweet_7.json"
    source.write_text(json.dumps(tweets))
    return source, official


def test_full_audit_and_selection_preserve_splits_and_missing_temporal_features(
    tmp_path,
):
    source, official = fixtures(tmp_path)
    report = prepare_shard(
        source,
        official,
        tmp_path / "out",
        tmp_path / "work",
        min_posts=3,
        max_posts=3,
        seed=42,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert report["audit"]["records"] == 26
    assert report["audit"]["duplicate_records"] == 1
    assert report["audit"]["unknown_author_records"] == 1
    assert report["audit"]["unique_known_posts"] == 24
    assert report["rows"] == 6
    assert report["counts"] == {"train": 2, "val": 2, "test": 2}
    for split, ids in [
        ("train", {"u1", "u2"}),
        ("val", {"u3", "u4"}),
        ("test", {"u5", "u6"}),
    ]:
        rows = pq.read_table(tmp_path / f"out/{split}.parquet").to_pylist()
        assert {r["external_user_id"] for r in rows} == ids
        assert {r["label"] for r in rows} == {0, 1}
        assert all(r["available_posts"] == 4 and r["observed_posts"] == 3 for r in rows)
        assert all(
            r["posts_per_day"] is None and r["interval_cv"] is None for r in rows
        )
    assert report["training_performed"] is False


def test_sampling_is_independent_of_input_order(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for root, reverse in [(a, False), (b, True)]:
        source, official = fixtures(root, reverse)
        prepare_shard(
            source,
            official,
            root / "out",
            root / "work",
            min_posts=3,
            max_posts=3,
            seed=42,
        )
    for split in ("train", "val", "test"):
        assert (
            pq.read_table(a / f"out/{split}.parquet").to_pylist()
            == pq.read_table(b / f"out/{split}.parquet").to_pylist()
        )


def test_conflicting_post_id_and_wrong_hash_never_publish_manifest(tmp_path):
    source, official = fixtures(tmp_path, conflicting=True)
    with pytest.raises(ValueError, match="Conflicting tweet"):
        prepare_shard(
            source,
            official,
            tmp_path / "out",
            tmp_path / "work",
            min_posts=3,
            max_posts=3,
        )
    assert not (tmp_path / "out/manifest.json").exists()


def test_source_fingerprint_mismatch_fails(tmp_path):
    source, official = fixtures(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        prepare_shard(
            source,
            official,
            tmp_path / "out",
            tmp_path / "work",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "out/manifest.json").exists()


def test_index_can_be_reused_but_not_with_different_sampling_seed(tmp_path):
    source, official = fixtures(tmp_path)
    prepare_shard(
        source,
        official,
        tmp_path / "out",
        tmp_path / "work",
        min_posts=3,
        max_posts=3,
        seed=42,
    )
    report = prepare_shard(
        source,
        official,
        tmp_path / "second",
        tmp_path / "work",
        min_posts=4,
        max_posts=4,
        seed=42,
        resume=True,
    )
    assert report["rows"] == 6
    with pytest.raises(ValueError, match="identity"):
        prepare_shard(
            source,
            official,
            tmp_path / "third",
            tmp_path / "work",
            seed=99,
            resume=True,
        )


@pytest.mark.parametrize("references", [[{}], [{"type": "replied_t0", "id": "999"}]])
def test_malformed_reference_metadata_is_never_known_negative(references):
    from engine.shard_index import canonical_tweet

    with pytest.raises(ValueError, match="reference"):
        canonical_tweet(
            {
                "id": "t1",
                "created_at": "2022-01-01T00:00:00Z",
                "text": "x",
                "referenced_tweets": references,
            },
            "u1",
        )


def test_deliverable_retains_all_official_input_fingerprints(tmp_path):
    source, official = fixtures(tmp_path)
    report = prepare_shard(
        source, official, tmp_path / "out", tmp_path / "work", min_posts=3, max_posts=3
    )
    assert report["source"]["official_inputs"] == {
        name: hashlib.sha256((official / name).read_bytes()).hexdigest()
        for name in ("label.csv", "split.csv", "user.json")
    }
