from fastapi.testclient import TestClient

from service.app import create_app
from service.settings import Settings
from service.worker import execute_job


def setup(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/jobs.sqlite3",
        api_key="test-secret",
        model_path=tmp_path / "absent.joblib",
    )
    dispatches = []
    app = create_app(settings, dispatcher=dispatches.append)
    client = TestClient(app)
    return settings, client, dispatches


def headers(tenant="user:1"):
    return {
        "Authorization": "Bearer test-secret",
        "X-Tenant-ID": tenant,
        "Idempotency-Key": "key1",
    }


def test_import_analysis_state_persistence_and_no_model(tmp_path):
    settings, c, queue = setup(tmp_path)
    assert c.get("/health/live").status_code == 200
    assert c.get("/health/ready").status_code == 503
    assert c.post("/v1/imports", json={}).status_code == 401
    content = "author_id,post_id,created_at,text\na,p,2026-01-01T00:00:00Z,hello\n"
    payload = {"format": "csv", "content": content}
    response = c.post("/v1/imports", json=payload, headers=headers())
    assert response.status_code == 202
    imported = response.json()["job_id"]
    assert (
        c.post("/v1/imports", json=payload, headers=headers()).json()["job_id"]
        == imported
    )
    assert len(queue) == 1
    execute_job(imported, settings)
    assert (
        c.get(f"/v1/jobs/{imported}", headers=headers()).json()["status"] == "completed"
    )
    analysis = c.post(
        "/v1/analyses",
        json={"source_job_id": imported},
        headers={**headers(), "Idempotency-Key": "analysis1"},
    )
    assert analysis.status_code == 202
    job = analysis.json()["job_id"]
    execute_job(job, settings)
    result = c.get(f"/v1/analyses/{job}", headers=headers()).json()
    assert result["results"][0]["decision"] == "model_not_ready"
    assert result["results"][0]["score"] is None
    assert result["results"][0]["coverage"]["complete"] is False
    # Opening a new app instance must retain data; known IDs cannot cross tenants.
    other = TestClient(create_app(settings, dispatcher=lambda _: None))
    assert other.get(f"/v1/jobs/{job}", headers=headers()).status_code == 200
    assert other.get(f"/v1/jobs/{job}", headers=headers("user:2")).status_code == 404
    assert (
        other.post(
            "/v1/analyses", json={"source_job_id": imported}, headers=headers("user:2")
        ).status_code
        == 404
    )
    # Repeated worker delivery is a no-op after success.
    execute_job(job, settings)
    assert other.get(f"/v1/analyses/{job}", headers=headers()).json() == result


def test_idempotency_conflict_and_bad_import_fail_as_explicit_jobs(tmp_path):
    settings, c, queue = setup(tmp_path)
    bad = {"format": "jsonl", "content": "not-json"}
    r = c.post("/v1/imports", json=bad, headers=headers())
    assert r.status_code == 202
    conflict = c.post(
        "/v1/imports", json={**bad, "content": "different"}, headers=headers()
    )
    assert conflict.status_code == 409
    execute_job(r.json()["job_id"], settings)
    status = c.get("/v1/jobs/" + r.json()["job_id"], headers=headers()).json()
    assert status["status"] == "failed"
    assert status["error"] == "invalid_input"


def test_unconfigured_collection_fails_explicitly(tmp_path):
    settings, c, queue = setup(tmp_path)
    r = c.post("/v1/collections/threads", json={"query": "Алматы"}, headers=headers())
    assert r.status_code == 202
    execute_job(r.json()["job_id"], settings)
    assert (
        c.get("/v1/jobs/" + r.json()["job_id"], headers=headers()).json()["error"]
        == "threads_not_configured"
    )


def test_default_celery_dispatch_executes_persisted_import_without_training(
    tmp_path, monkeypatch
):
    from service.worker import celery

    monkeypatch.setenv("ENGINE_DATABASE_URL", f"sqlite:///{tmp_path}/celery.sqlite3")
    monkeypatch.setenv("ENGINE_API_KEY", "test-secret")
    monkeypatch.setenv("ENGINE_MODEL_PATH", str(tmp_path / "absent.joblib"))
    monkeypatch.setitem(celery.conf, "task_always_eager", True)
    monkeypatch.setitem(celery.conf, "broker_url", "memory://")
    with TestClient(create_app()) as c:
        r = c.post(
            "/v1/imports",
            headers=headers(),
            json={
                "format": "csv",
                "content": "author_id,post_id,created_at,text\na,p,2026-01-01T00:00:00Z,hi\n",
            },
        )
        assert r.status_code == 202
        job = c.get("/v1/jobs/" + r.json()["job_id"], headers=headers()).json()
        assert job["status"] == "completed"
        assert job["summary"]["posts"] == 1
