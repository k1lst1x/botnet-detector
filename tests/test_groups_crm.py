from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from core.models import EngineAccount, EngineGroup, EngineJob
from engine.coactivity import group_summary
from engine.contracts import AccountSnapshot, Dataset, Observation, Post
from engine.inference import analyze_dataset, review_queue

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _campaign_dataset():
    accounts, posts = [], []
    for i in range(30):
        uid = f"h{i}"
        accounts.append(AccountSnapshot(platform="threads", external_user_id=uid, username=uid))
        for j in range(5):
            posts.append(Post(platform="threads", external_post_id=f"{uid}-{j}", author_id=uid,
                              created_at=T0 + timedelta(minutes=i * 7 + j),
                              text=f"my own thoughts number {i} about topic #t{i}_{j}", targets=[f"friend{i}"]))
    for i in range(6):
        uid = f"b{i}"
        accounts.append(AccountSnapshot(platform="threads", external_user_id=uid, username=uid))
        for j in range(5):
            posts.append(Post(platform="threads", external_post_id=f"{uid}-{j}", author_id=uid,
                              created_at=T0 + timedelta(minutes=j), targets=["brand"],
                              text=f"Best deal ever, buy now {j}! https://shop.example/promo #sale"))
    return Dataset(accounts=accounts, posts=posts,
                   observation=Observation(start=T0, end=T0 + timedelta(hours=5), source="manual"))


def test_group_summary_flags_campaign_with_explanation():
    summary = group_summary(_campaign_dataset())
    flagged = [g for g in summary["groups"] if g["flagged"]]
    assert len(flagged) == 1
    group = flagged[0]
    assert set(group["members"]) == {f"threads:b{i}" for i in range(6)}
    kinds = {o["object"].split(":")[0] for o in group["shared_objects"]}
    assert {"m", "u", "h", "txt"} <= kinds


def test_review_queue_is_account_or_group():
    assert review_queue("priority_review", False) == "priority"
    assert review_queue("standard_review", True) == "priority"
    assert review_queue("insufficient_data", True) == "priority"
    assert review_queue("standard_review", False) == "standard"
    assert review_queue("model_not_ready", False) == "not_scored"


class _NoModel:
    bundle = None

    def status(self):
        return {"status": "model_not_ready", "model_version": None}


def test_inference_attaches_group_and_queue_without_model():
    result = analyze_dataset(_campaign_dataset(), _NoModel())
    rows = {r["external_user_id"]: r for r in result["results"]}
    assert rows["b0"]["queue"] == "priority" and rows["b0"]["group"]["flagged"]
    assert "coordinated_group" in rows["b0"]["warnings"]
    assert rows["h0"]["queue"] == "not_scored"
    assert "account_group" not in result["groups"]


@pytest.mark.django_db
def test_refresh_stores_groups_and_links_accounts(client):
    user = get_user_model().objects.create_user("analyst")
    other = get_user_model().objects.create_user("other")
    client.force_login(user)
    EngineJob.objects.create(owner=user, job_id="j1", kind="analysis", status="completed")
    result = {
        "results": [{"platform": "threads", "external_user_id": "b0", "username": "b0",
                     "score": 0.2, "decision": "standard_review", "model_version": "v",
                     "queue": "priority",
                     "group": {"group_id": "g1", "flagged": True, "coordination_score": 0.8}}],
        "groups": {"groups": [{"group_id": "g1", "size": 6, "density": 1.0, "mean_weight": 0.7,
                               "flagged": True, "members": ["threads:b0"],
                               "shared_objects": [{"object": "h:sale", "accounts": 6}]}]},
    }
    with patch("core.engine_views.EngineClient.get", side_effect=[{"status": "completed"}, result]):
        assert client.post("/engine/jobs/j1/refresh/").status_code == 200
    group = EngineGroup.objects.get(owner=user)
    account = EngineAccount.objects.get(owner=user, external_user_id="b0")
    assert account.group == group and account.queue == "priority"
    assert account.coordination_score == 0.8
    page = client.get(f"/engine/groups/{group.pk}/")
    assert page.status_code == 200 and "#sale" in page.content.decode()
    assert "хэштег" in client.get("/engine/groups/").content.decode()
    assert "группа g1" in client.get("/engine/accounts/").content.decode()
    client.force_login(other)
    assert client.get(f"/engine/groups/{group.pk}/").status_code == 404
