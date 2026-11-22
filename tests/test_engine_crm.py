from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model

from core.models import EngineAccount, EngineJob


@pytest.mark.django_db
def test_engine_pages_require_login_and_never_expose_other_users_jobs(client):
    User = get_user_model()
    a = User.objects.create_user("alice", password="password")
    b = User.objects.create_user("bob", password="password")
    EngineJob.objects.create(owner=a, job_id="job-a", kind="import")
    EngineJob.objects.create(owner=b, job_id="job-b", kind="import")
    assert client.get("/engine/").status_code == 302
    client.force_login(a)
    page = client.get("/engine/")
    assert page.status_code == 200
    assert b"job-a" in page.content
    assert b"job-b" not in page.content
    assert client.get("/engine/jobs/job-b/").status_code == 404


@pytest.mark.django_db
def test_import_sends_owner_scoped_request_to_fastapi(client):
    user = get_user_model().objects.create_user("alice")
    client.force_login(user)
    from django.core.files.uploadedfile import SimpleUploadedFile

    upload = SimpleUploadedFile(
        "posts.csv", b"author_id,post_id,created_at,text\na,p,2026-01-01T00:00:00Z,hi\n"
    )
    with patch(
        "core.engine_views.EngineClient.submit",
        return_value={"job_id": "imported", "status": "queued"},
    ) as submit:
        response = client.post(
            "/engine/import/",
            {"format": "csv", "file": upload, "idempotency_key": "submission"},
        )
    assert response.status_code == 302
    assert EngineJob.objects.get(job_id="imported").owner_id == user.id
    assert submit.call_args.args[0] == "/v1/imports"
    assert submit.call_args.args[1]["format"] == "csv"


@pytest.mark.django_db
def test_same_handle_across_platforms_and_users_remains_distinct(client):
    user = get_user_model().objects.create_user("alice")
    other = get_user_model().objects.create_user("bob")
    for owner, platform in [(user, "twitter"), (user, "threads"), (other, "threads")]:
        EngineAccount.objects.create(
            owner=owner, platform=platform, external_user_id="a", username="same"
        )
    client.force_login(user)
    result = client.get("/engine/accounts/")
    assert result.status_code == 200
    account = EngineAccount.objects.get(owner=other, platform="threads")
    assert (
        client.post(f"/engine/accounts/{account.pk}/", {"label": "bot"}).status_code
        == 404
    )


@pytest.mark.django_db
def test_queue_publish_failure_keeps_persisted_job_visible_to_user(client):
    import httpx
    from django.test import override_settings

    user = get_user_model().objects.create_user("queue-user")
    client.force_login(user)
    from django.core.files.uploadedfile import SimpleUploadedFile

    failure = httpx.Response(
        503,
        request=httpx.Request("POST", "http://engine/v1/imports"),
        json={"detail": {"code": "queue_unavailable", "job_id": "persisted-job"}},
    )
    with (
        override_settings(ENGINE_API_KEY="test-secret"),
        patch("core.engine_client.httpx.request", return_value=failure),
    ):
        response = client.post(
            "/engine/import/",
            {
                "format": "csv",
                "file": SimpleUploadedFile("posts.csv", b"hello"),
                "idempotency_key": "same",
            },
        )
    assert response.url == "/engine/jobs/persisted-job/"
    assert EngineJob.objects.get(job_id="persisted-job").owner_id == user.pk


@pytest.mark.django_db
def test_refreshing_old_analysis_cannot_roll_back_latest_crm_decision(client):
    from datetime import timedelta
    from django.utils import timezone

    user = get_user_model().objects.create_user("analyst")
    client.force_login(user)
    old = EngineJob.objects.create(
        owner=user, job_id="old", kind="analysis", status="completed"
    )
    new = EngineJob.objects.create(
        owner=user, job_id="new", kind="analysis", status="completed"
    )
    EngineJob.objects.filter(pk=old.pk).update(
        created_at=timezone.now() - timedelta(days=1)
    )
    record = EngineAccount.objects.create(
        owner=user,
        platform="threads",
        external_user_id="a",
        score=0.9,
        decision="bot",
        source_job=new,
        label="human",
        notes="reviewed",
    )
    old_result = {
        "results": [
            {
                "platform": "threads",
                "external_user_id": "a",
                "username": "a",
                "score": 0.1,
                "decision": "human",
                "model_version": "old-v",
            }
        ]
    }
    with patch(
        "core.engine_views.EngineClient.get",
        side_effect=[{"status": "completed"}, old_result],
    ):
        response = client.post("/engine/jobs/old/refresh/")
    assert response.status_code == 200
    record.refresh_from_db()
    assert record.score == 0.9
    assert record.source_job_id == new.pk
    assert record.label == "human"
    assert record.notes == "reviewed"
