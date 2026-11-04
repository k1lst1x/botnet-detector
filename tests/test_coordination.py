from engine.coordination import coordination_summary
from engine.contracts import Dataset
from tests.test_engine_features import sample


def test_coordination_is_separate_from_bot_probability_and_ignores_viral_buckets():
    a, posts, obs = sample(False)
    b = a.model_copy(update={"external_user_id": "b", "username": "b"})
    bs = [
        p.model_copy(
            update={"external_post_id": "b" + p.external_post_id, "author_id": "b"}
        )
        for p in posts
    ]
    data = Dataset(accounts=[a, b], posts=posts + bs, observation=obs)
    summary = coordination_summary(data)
    assert summary["edges"][0]["synchrony"] == 1
    assert summary["edges"][0]["shared_url_ratio"] == 0
    assert summary["edges"][0]["repeated_windows"] == 3
    assert summary["note"] == "coordination_is_not_bot_probability"
    assert coordination_summary(data, max_bucket_accounts=1)["edges"] == []
