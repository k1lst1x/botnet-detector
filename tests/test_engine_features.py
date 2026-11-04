from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from engine.contracts import AccountSnapshot, Dataset, Observation, Post
from engine.features import extract_features


def sample(complete=True):
    account = AccountSnapshot(platform="threads", external_user_id="a", username="same")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    obs = Observation(
        start=start, end=start + timedelta(days=1), complete=complete, source="timeline"
    )
    posts = [
        Post(
            platform="threads",
            external_post_id=str(i),
            author_id="a",
            created_at=start + timedelta(hours=i),
            text="Hello 1!",
            is_reply=False,
        )
        for i in range(3)
    ]
    return account, posts, obs


def test_regular_timeline_has_hand_computed_features():
    account, posts, obs = sample()
    f = extract_features(account, posts, obs)
    assert f["posts_per_day"] == 3
    assert f["interval_median_s"] == 3600
    assert f["interval_cv"] == 0
    assert f["burstiness"] == -1
    assert f["duplicate_ratio"] == pytest.approx(2 / 3)
    assert f["reply_ratio"] == 0
    assert f["repost_ratio"] is None
    assert f["repost_ratio__missing"] == 1
    assert f["followers_log1p"] is None


def test_partial_history_does_not_claim_population_activity():
    account, posts, obs = sample(False)
    f = extract_features(account, posts, obs)
    assert f["posts_per_day"] is None
    assert f["interval_median_s"] is None
    assert f["hour_entropy"] is None
    assert (
        f["reply_ratio"] == 0
    )  # observed-sample statistic, coverage separately reported


def test_utc_equivalent_times_and_unicode_without_language_dictionary():
    account, posts, obs = sample()
    first = extract_features(account, posts, obs)
    moved = [
        p.model_copy(
            update={"created_at": p.created_at.astimezone(timezone(timedelta(hours=5)))}
        )
        for p in posts
    ]
    assert extract_features(account, moved, obs) == first
    translated = [p.model_copy(update={"text": "Приве 1!"}) for p in posts]
    assert extract_features(account, translated, obs)["text_length_median"] == 8


def test_empty_text_and_one_post_do_not_fabricate_regular_behavior():
    account, posts, obs = sample()
    f = extract_features(account, [posts[0].model_copy(update={"text": ""})], obs)
    assert f["duplicate_ratio"] is None
    assert f["interval_cv"] is None
    assert f["text_digit_ratio"] is None


def test_contract_rejects_naive_times_duplicate_posts_wrong_author_and_outside_window():
    account, posts, obs = sample()
    with pytest.raises(ValidationError):
        Post(
            platform="threads",
            external_post_id="p",
            author_id="a",
            created_at="2026-01-01T00:00:00",
        )
    with pytest.raises(ValueError):
        Dataset(accounts=[account], posts=posts + [posts[0]], observation=obs)
    with pytest.raises(ValueError):
        Dataset(
            accounts=[account],
            posts=[posts[0].model_copy(update={"platform": "twitter"})],
            observation=obs,
        )
    with pytest.raises(ValueError):
        Dataset(
            accounts=[account],
            posts=[
                posts[0].model_copy(
                    update={"created_at": obs.end + timedelta(seconds=1)}
                )
            ],
            observation=obs,
        )


def test_same_username_different_platforms_have_distinct_identity():
    account, posts, obs = sample()
    other = AccountSnapshot(platform="twitter", external_user_id="a", username="same")
    data = Dataset(accounts=[account, other], posts=posts, observation=obs)
    assert len(data.accounts) == 2


def test_malformed_link_in_user_text_does_not_abort_feature_extraction():
    account, posts, obs = sample()
    posts[0] = posts[0].model_copy(update={"text": "See https://[broken"})
    f = extract_features(account, posts, obs)
    assert f["unique_domains"] == 0
