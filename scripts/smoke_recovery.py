"""Fault-injection checks against this project's local Compose stack only.

Temporarily stops this project's worker/Redis, then restores them in finally.
Does not train, remove volumes, or touch other Compose projects.
"""

import json
import subprocess
import time
import uuid

import httpx

from smoke_mvp import ROOT, csrf, env_values, job_id, login


def compose(*args):
    subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def main():
    env = env_values()
    credentials = json.loads((ROOT / "media/dev-login.json").read_text())
    web = f"http://127.0.0.1:{env.get('DJANGO_PORT', '8000')}"
    api = f"http://127.0.0.1:{env.get('ENGINE_PORT', '8001')}"
    headers = {
        "Authorization": f"Bearer {env['ENGINE_API_KEY']}",
        "X-Tenant-ID": f"user:{credentials['primary']['id']}",
    }
    report = {"checks": [], "status": "running"}

    def passed(name):
        report["checks"].append(name)
        print(f"PASS {name}", flush=True)

    run = uuid.uuid4().hex
    payload = (ROOT / "data/threads_example.json").read_bytes()
    with (
        httpx.Client(base_url=web, timeout=45) as client,
        httpx.Client(base_url=api, headers=headers, timeout=45) as engine,
    ):
        login(client, credentials["primary"])

        def upload(key):
            return client.post(
                "/engine/import/",
                data={
                    "csrfmiddlewaretoken": csrf(client),
                    "format": "json",
                    "idempotency_key": key,
                },
                files={"file": ("recovery.json", payload, "application/json")},
            )

        def wait(id):
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                status = engine.get(f"/v1/jobs/{id}").json()
                assert status["status"] != "failed", (
                    f"Job failed: {status.get('error')}"
                )
                if status["status"] == "completed":
                    return status
                time.sleep(0.4)
            raise RuntimeError("Worker did not recover in time")

        previous = json.loads((ROOT / "media/mvp-smoke-report.json").read_text())
        try:
            compose("restart", "worker")
            assert (
                engine.get("/v1/jobs/" + previous["analysis_job_id"]).json()["status"]
                == "completed"
            )
            assert previous["fixture_author_id"] in client.get("/engine/accounts/").text
            passed("completed_sql_jobs_and_crm_survive_worker_restart")
            compose("stop", "worker")
            queued = job_id(upload(run + "-stopped-worker"))
            assert engine.get(f"/v1/jobs/{queued}").json()["status"] == "queued"
            compose("start", "worker")
            wait(queued)
            passed("real_redis_queued_job_consumed_after_worker_restart")
            compose("stop", "redis")
            response = upload(run + "-broker-down")
            recovered = job_id(response)
            assert engine.get(f"/v1/jobs/{recovered}").json()["status"] == "queued"
            page = client.get(response.headers["location"])
            assert recovered in page.text
            passed("broker_outage_preserves_visible_sql_job_in_django")
        finally:
            compose("start", "redis", "worker")
        # Redis must be accepting requests before republishing through Celery.
        for _ in range(30):
            try:
                compose("exec", "-T", "redis", "redis-cli", "ping")
                break
            except subprocess.CalledProcessError:
                time.sleep(0.5)
        compose("exec", "-T", "worker", "python", "-m", "service.worker", "--recover")
        wait(recovered)
        assert job_id(upload(run + "-broker-down")) == recovered
        passed("recover_republishes_persisted_job_without_duplicate_submission")
        report.update(
            {"status": "passed", "queued_job_id": queued, "recovered_job_id": recovered}
        )
    path = ROOT / "media/mvp-recovery-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
