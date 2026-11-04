"""Trusted server-to-server client; tenant never comes from user form input."""

import uuid

from django.conf import settings
import httpx


class EngineClientError(RuntimeError):
    pass


class EngineClient:
    def __init__(self, user):
        self.tenant = f"user:{user.pk}"

    def request(self, method, path, payload=None, key=None):
        if not settings.ENGINE_API_KEY:
            raise EngineClientError("Сервис анализа не настроен")
        headers = {
            "Authorization": f"Bearer {settings.ENGINE_API_KEY}",
            "X-Tenant-ID": self.tenant,
        }
        if method == "POST":
            headers["Idempotency-Key"] = key or str(uuid.uuid4())
        try:
            response = httpx.request(
                method,
                settings.ENGINE_API_URL.rstrip("/") + path,
                json=payload,
                headers=headers,
                timeout=15,
                follow_redirects=False,
            )
            if response.status_code == 503:
                body = response.json()
                detail = body.get("detail")
                if (
                    isinstance(detail, dict)
                    and detail.get("code") == "queue_unavailable"
                    and detail.get("job_id")
                ):
                    return {
                        "job_id": detail["job_id"],
                        "status": "queued",
                        "dispatch_pending": True,
                    }
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            raise EngineClientError(
                "Сервис анализа недоступен или отклонил запрос. Повторите попытку позже."
            ) from None

    def submit(self, path, payload, key=None):
        return self.request("POST", path, payload, key)

    def get(self, path):
        return self.request("GET", path)
