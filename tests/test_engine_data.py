import json

import pyarrow.parquet as pq
import pytest

from engine.datasets import prepare_twibot, read_import
from engine.model import ModelService, ModelUnavailable


def twibot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "user.json").write_text(
        json.dumps(
            [
                {
                    "id": "u1",
                    "username": "one",
                    "public_metrics": {"followers_count": 0, "following_count": 2},
                },
                {"id": "u2", "username": "two", "public_metrics": {}},
                {
                    "id": "u3",
                    "username": "three",
                    "public_metrics": {"followers_count": 7},
                },
            ]
        )
    )
    (source / "label.csv").write_text("id,label\nu1,human\nu2,bot\nu3,human\n")
    (source / "split.csv").write_text("id,split\nu1,train\nu2,val\nu3,test\n")
    return source


def test_prepare_preserves_official_splits_and_unknown_behavior(tmp_path):
    source = twibot(tmp_path)
    report = prepare_twibot(source, tmp_path / "out")
    assert report["rows"] == 3
    assert report["counts"] == {"train": 1, "val": 1, "test": 1}
    train = pq.read_table(tmp_path / "out/train.parquet").to_pylist()
    assert train[0]["external_user_id"] == "u1"
    assert train[0]["followers_log1p"] == 0
    assert train[0]["interval_cv"] is None
    assert train[0]["label"] == 0
    assert report["behavior_available"] is False


def test_prepare_rejects_duplicate_label_and_unknown_split(tmp_path):
    source = twibot(tmp_path)
    with (source / "label.csv").open("a") as f:
        f.write("u1,bot\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_twibot(source, tmp_path / "out")
    assert not (tmp_path / "out/manifest.json").exists()


def test_prepare_joins_tweet_author_edges_without_inventing_absent_flags(tmp_path):
    source = twibot(tmp_path)
    (source / "tweet_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "t10",
                    "created_at": "2022-01-01T00:00:00Z",
                    "text": "hi",
                    "author_id": "u1",
                },
                {"id": "t11", "created_at": "2022-01-01T01:00:00Z", "text": "hi"},
            ]
        )
    )
    (source / "edge.csv").write_text("source_id,target_id,relation\nu1,t11,post\n")
    prepare_twibot(source, tmp_path / "out")
    row = pq.read_table(tmp_path / "out/train.parquet").to_pylist()[0]
    assert row["observed_posts"] == 2
    assert row["duplicate_ratio"] == 0.5
    assert row["repost_ratio"] is None
    # A historical archive without declared full window isn't a complete timeline.
    assert row["posts_per_day"] is None


def test_import_csv_and_jsonl_keep_unknown_values_and_reject_invalid_boolean():
    csv_text = "platform,author_id,post_id,created_at,text,is_reply\nthreads,a,p,2026-01-01T00:00:00Z,hello,\n"
    data = read_import(csv_text, "csv")
    assert data.posts[0].is_reply is None
    assert data.accounts[0].followers is None
    obj = {
        "platform": "threads",
        "author_id": "a",
        "post_id": "p",
        "created_at": "2026-01-01T00:00:00Z",
        "text": "hello",
    }
    assert read_import(json.dumps(obj) + "\n", "jsonl").posts[0].external_post_id == "p"
    with pytest.raises(ValueError):
        read_import(csv_text.replace("hello,", "hello,maybe"), "csv")


def test_missing_model_does_not_train_or_return_heuristic(tmp_path):
    service = ModelService(tmp_path / "missing.joblib")
    assert service.status()["status"] == "model_not_ready"
    with pytest.raises(ModelUnavailable):
        service.predict([{}])


def test_malformed_artifact_is_not_reported_ready(tmp_path):
    import joblib

    class_path = tmp_path / "bad.joblib"
    # A valid joblib file with missing metadata is still an invalid model.
    joblib.dump(
        {
            "features": ["followers_log1p"],
            "feature_version": "portable-v1",
            "threshold": 0.5,
            "model": None,
        },
        class_path,
    )
    assert ModelService(class_path).status()["status"] == "model_incompatible"


def test_preparation_unknown_split_is_rejected(tmp_path):
    source = twibot(tmp_path)
    (source / "split.csv").write_text(
        "id,split\nu1,train\nu2,validation_typo\nu3,test\n"
    )
    with pytest.raises(ValueError, match="Invalid"):
        prepare_twibot(source, tmp_path / "out")


def test_unfitted_classifier_artifact_is_rejected_before_readiness(tmp_path):
    import joblib
    from sklearn.tree import DecisionTreeClassifier

    path = tmp_path / "unfitted.joblib"
    joblib.dump(
        {
            "model": DecisionTreeClassifier(),
            "features": ["followers_log1p"],
            "feature_version": "portable-v1",
            "model_version": "fake",
            "feature_set": "profile",
            "required_observed_features": ["followers_log1p"],
            "threshold": 0.5,
        },
        path,
    )
    assert ModelService(path).status()["status"] == "model_incompatible"


def test_prepare_can_use_explicit_complete_window_for_temporal_features(tmp_path):
    from engine.contracts import Observation

    source = twibot(tmp_path)
    (source / "tweet_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "t1",
                    "author_id": "u1",
                    "created_at": "2022-01-01T00:00:00Z",
                    "text": "hi",
                },
                {
                    "id": "t2",
                    "author_id": "u1",
                    "created_at": "2022-01-01T01:00:00Z",
                    "text": "hi",
                },
            ]
        )
    )
    obs = Observation(
        start="2022-01-01T00:00:00Z",
        end="2022-01-02T00:00:00Z",
        source="verified_archive",
        complete=True,
    )
    report = prepare_twibot(source, tmp_path / "out", observation=obs)
    row = pq.read_table(tmp_path / "out/train.parquet").to_pylist()[0]
    assert row["posts_per_day"] == 2
    assert row["interval_median_s"] == 3600
    assert report["observation_complete"] is True


def test_handle_only_import_does_not_collide_with_platform_id():
    by_handle = read_import(
        "handle,post_id,created_at,text\n123,p,2026-01-01T00:00:00Z,hi\n", "csv"
    )
    by_id = read_import(
        "author_id,post_id,created_at,text\n123,p,2026-01-01T00:00:00Z,hi\n", "csv"
    )
    assert by_handle.accounts[0].key != by_id.accounts[0].key
    assert by_handle.accounts[0].identity_source == "username_fallback"
    assert by_handle.posts[0].author_id == "username:123"
