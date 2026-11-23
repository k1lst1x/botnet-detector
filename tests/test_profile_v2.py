import json
from datetime import datetime, timezone

import numpy as np
import pytest
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from engine.contracts import AccountSnapshot, Dataset, Observation
from engine.datasets import read_import
from engine.features import (
    PROFILE_EXTRA_FEATURES,
    PROFILE_V2_FEATURES,
    PROFILE_VERSION,
    name_username_overlap,
    profile_text_features,
)
from engine.inference import analyze_dataset
from engine.model import ModelService

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def test_profile_text_features_hand_computed():
    f = profile_text_features("Hi @a #b 2 https://t.co/x", "Ann_1", "ann_99", False, True)
    assert f["bio_length"] == 25
    assert (f["bio_url_count"], f["bio_mention_count"], f["bio_hashtag_count"]) == (1, 1, 1)
    assert f["bio_digit_ratio"] == pytest.approx(1 / 25)
    assert (f["avatar_default"], f["verified"]) == (0.0, 1.0)
    assert f["name_length"] == 5 and f["name_digit_ratio"] == pytest.approx(0.2)
    assert f["name_non_alnum_ratio"] == pytest.approx(0.2)  # underscore
    assert (f["username_length"], f["username_trailing_digits"], f["username_underscore_count"]) == (6, 2, 1)
    assert f["name_username_overlap"] == 1.0


def test_unknown_fields_stay_missing_not_zero():
    f = profile_text_features(None, None, "x", None, None)
    assert f["username_length"] == 1
    assert all(f[n] is None for n in PROFILE_EXTRA_FEATURES if not n.startswith("username"))
    assert f["name_username_overlap"] is None


def test_overlap_without_latin_letters_is_zero():
    assert name_username_overlap("Анна", "anna") == 0.0


def test_v2_feature_order_is_stable():
    assert len(PROFILE_V2_FEATURES) == 18 and len(set(PROFILE_V2_FEATURES)) == 18
    assert set(PROFILE_EXTRA_FEATURES) < set(PROFILE_V2_FEATURES)


def test_import_reads_optional_profile_columns():
    row = {"author_id": "a", "post_id": "p", "created_at": "2026-01-01T00:00:00Z",
           "name": "Ann", "is_verified": "true", "avatar_default": "0"}
    account = read_import(json.dumps(row), "jsonl").accounts[0]
    assert (account.display_name, account.is_verified, account.avatar_default) == ("Ann", True, False)


def _bundle(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((200, len(PROFILE_V2_FEATURES)))
    y = (X[:, 0] > 0.5).astype(int)
    pipe = Pipeline([("imputer", SimpleImputer(keep_empty_features=True)),
                     ("xgboost", xgb.XGBClassifier(n_estimators=5, max_depth=2))]).fit(X, y)
    path = tmp_path / "model.joblib"
    import joblib
    joblib.dump({
        "model": pipe, "features": PROFILE_V2_FEATURES, "feature_version": PROFILE_VERSION,
        "feature_set": "profile", "model_version": "test-v2", "threshold": 0.5,
        "required_observed_features": ["followers_log1p", "bio_present", "username_length", "name_length"],
        "minimum_observed_ratio": 0.75, "minimum_posts": 0, "decision_policy": "review_queue",
    }, path)
    return ModelService(path)


def test_profile_v2_artifact_scores_threads_profile_without_posts(tmp_path):
    model = _bundle(tmp_path)
    assert model.status()["feature_version"] == PROFILE_VERSION
    full = AccountSnapshot(platform="threads", external_user_id="1", username="ann_99",
                           display_name="Ann", followers=10, biography="hi", has_avatar=True,
                           is_verified=False)
    bare = AccountSnapshot(platform="threads", external_user_id="2", username="bare")
    data = Dataset(accounts=[full, bare], posts=[],
                   observation=Observation(start=NOW, end=NOW, source="manual"))
    first, second = analyze_dataset(data, model)["results"]
    assert first["decision"] in ("priority_review", "standard_review") and first["explanation"]
    assert first["feature_version"] == PROFILE_VERSION
    assert second["decision"] == "insufficient_data" and second["score"] is None


def test_unknown_feature_version_is_rejected(tmp_path):
    model = _bundle(tmp_path)
    import joblib
    bundle = joblib.load(model.path)
    bundle["feature_version"] = "profile-v9"
    joblib.dump(bundle, model.path)
    assert ModelService(model.path).status()["status"] == "model_incompatible"
