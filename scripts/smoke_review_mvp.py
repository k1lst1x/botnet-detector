"""Exercise a running Django/FastAPI/Celery stack without fitting a model.

Run with .venv/bin/python scripts/smoke_review_mvp.py after creating local users.
Secrets are read locally; the JSON report never includes passwords or API keys.
"""

import argparse
import json
from pathlib import Path
import re
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]


def env_values():
    result = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def csrf(client):
    return client.cookies["csrftoken"]


def login(client, credentials):
    response = client.get("/auth/login/")
    assert response.status_code == 200, "Login page unavailable"
    response = client.post(
        "/auth/login/",
        data={
            "csrfmiddlewaretoken": csrf(client),
            "username": credentials["username"],
            "password": credentials["password"],
            "next": "/engine/",
        },
    )
    assert response.status_code == 302, "Login rejected"


def job_id(response):
    assert response.status_code == 302, "Django did not accept submission"
    match = re.fullmatch(
        r"/engine/jobs/([a-f0-9-]+)/", response.headers.get("location", "")
    )
    assert match, "Submission did not produce a visible CRM job"
    return match[1]


def label_form_for_author(html, author):
    sections = re.findall(r"<section\b[^>]*>.*?</section>", html, re.S)
    matches = [section for section in sections if author in section]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely locate the smoke account")
    match = re.search(r'action="(/engine/accounts/\d+/)"', matches[0])
    if match is None:
        raise ValueError("CRM label form missing for the smoke account")
    return match[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials", type=Path, default=ROOT / "media/dev-login.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "media/mvp-smoke-report.json"
    )
    args = parser.parse_args()
    env = env_values()
    credentials = json.loads(args.credentials.read_text())
    web = f"http://127.0.0.1:{env.get('DJANGO_PORT', '8000')}"
    api = f"http://127.0.0.1:{env.get('ENGINE_PORT', '8001')}"
    report = {
        "mode": "running_stack_without_model",
        "checks": [],
        "web_url": web,
        "api_url": api,
    }
    run = uuid.uuid4().hex

    def passed(name):
        report["checks"].append(name)
        print(f"PASS {name}", flush=True)

    headers = {
        "Authorization": f"Bearer {env['ENGINE_API_KEY']}",
        "X-Tenant-ID": f"user:{credentials['primary']['id']}",
    }
    with (
        httpx.Client(base_url=api, headers=headers, timeout=20) as engine,
        httpx.Client(base_url=web, timeout=20) as client,
        httpx.Client(base_url=web, timeout=20) as other,
    ):
        assert engine.get("/health/live").status_code == 200
        assert engine.get("/health/ready").json()["status"] == "ready"
        passed("live_api_and_ready_model")
        assert client.get("/engine/").status_code == 302
        login(client, credentials["primary"])
        login(other, credentials["secondary"])
        passed("real_django_login_and_anonymous_redirect")
        data = json.loads((ROOT / "data/threads_example.json").read_text())
        author = f"smoke-{run}"
        data["accounts"][0]["external_user_id"] = author
        data["accounts"][0]["username"] = author
        for i, post in enumerate(data["posts"]):
            post["author_id"] = author
            post["external_post_id"] = f"{run}-{i}"
        data["accounts"][0].update(followers=100, following=200, biography="Example profile", has_avatar=True)
        payload = json.dumps(data).encode()

        def upload():
            return client.post(
                "/engine/import/",
                data={
                    "csrfmiddlewaretoken": csrf(client),
                    "format": "json",
                    "idempotency_key": run,
                },
                files={"file": ("smoke.json", payload, "application/json")},
            )

        imported = job_id(upload())
        assert job_id(upload()) == imported
        passed("web_import_and_idempotent_repeat")

        def wait(id):
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                response = engine.get(f"/v1/jobs/{id}")
                response.raise_for_status()
                value = response.json()
                assert value["status"] != "failed", f"Job failed: {value.get('error')}"
                if value["status"] == "completed":
                    return value
                time.sleep(0.3)
            raise RuntimeError("Timed out waiting for real worker")

        imported_status = wait(imported)
        assert imported_status["summary"]["posts"] == 2
        response = client.post(
            f"/engine/jobs/{imported}/refresh/",
            data={"csrfmiddlewaretoken": csrf(client)},
        )
        assert response.status_code == 302
        passed("postgres_redis_worker_completed_import")
        analysis = job_id(
            client.post(
                f"/engine/jobs/{imported}/analyze/",
                data={
                    "csrfmiddlewaretoken": csrf(client),
                    "idempotency_key": f"{run}-analysis",
                },
            )
        )
        wait(analysis)
        result = engine.get(f"/v1/analyses/{analysis}").json()
        assert len(result["results"]) == 1
        account = result["results"][0]
        assert account["score"] is not None and account["decision"] in ("priority_review", "standard_review")
        assert account["explanation"]
        assert account["features"]["text_length_median"] is not None
        assert account["features"]["posts_per_day"] is None
        response = client.post(
            f"/engine/jobs/{analysis}/refresh/",
            data={"csrfmiddlewaretoken": csrf(client)},
        )
        assert response.status_code == 302
        response = client.get(f"/engine/jobs/{analysis}/")
        assert response.status_code == 200 and ("обычная очередь" in response.text or "приоритетная проверка" in response.text)
        passed("real_model_score_explanation_and_review_queue")
        page = client.get("/engine/accounts/")
        assert author in page.text
        label_url = label_form_for_author(page.text, author)
        assert other.get(f"/engine/jobs/{analysis}/").status_code == 404
        assert (
            other.post(
                label_url, data={"csrfmiddlewaretoken": csrf(other), "label": "bot"}
            ).status_code
            == 404
        )
        assert author not in other.get("/engine/accounts/").text
        assert (
            engine.get(
                f"/v1/jobs/{analysis}",
                headers={"X-Tenant-ID": f"user:{credentials['secondary']['id']}"},
            ).status_code
            == 404
        )
        passed("django_and_fastapi_tenant_isolation")
        response = client.post(
            label_url,
            data={
                "csrfmiddlewaretoken": csrf(client),
                "label": "human",
                "notes": f"smoke-note-{run}",
            },
        )
        assert response.status_code == 302
        assert f"smoke-note-{run}" in client.get("/engine/accounts/").text
        client.post(
            f"/engine/jobs/{analysis}/refresh/",
            data={"csrfmiddlewaretoken": csrf(client)},
        )
        assert f"smoke-note-{run}" in client.get("/engine/accounts/").text
        passed("manual_crm_label_survives_result_refresh")
        report.update(
            {
                "status": "passed",
                "import_job_id": imported,
                "analysis_job_id": analysis,
                "fixture_author_id": author,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
