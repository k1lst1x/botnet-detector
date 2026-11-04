"""Authenticated internal API. X-Tenant-ID is set by the trusted Django service."""

from contextlib import asynccontextmanager
import hmac
from typing import Literal
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from engine.model import ModelService
from .settings import Settings
from .store import Conflict, JobStore


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["csv", "jsonl", "json"]
    content: str = Field(min_length=1, max_length=8_000_000)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_job_id: str = Field(min_length=1, max_length=36)


class CollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    max_pages: int = Field(default=5, ge=1, le=20)
    after: str | None = Field(default=None, max_length=2048)


def create_app(config=None, dispatcher=None):
    config = config or Settings()
    store = JobStore(config.database_url)
    model = ModelService(config.model_path)
    if dispatcher is None:
        from .worker import dispatch

        dispatcher = dispatch

    @asynccontextmanager
    async def lifespan(app):
        yield
        store.engine.dispose()

    app = FastAPI(title="BotNet engine", version="0.1.0", lifespan=lifespan)

    def tenant(
        authorization: str | None = Header(default=None),
        x_tenant_id: str | None = Header(default=None),
    ):
        if not config.api_key:
            raise HTTPException(503, "service_not_configured")
        expected = f"Bearer {config.api_key}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "unauthorized")
        if not x_tenant_id or len(x_tenant_id) > 128 or not x_tenant_id.strip():
            raise HTTPException(400, "tenant_required")
        return x_tenant_id

    def own_job(job_id, owner):
        job = store.get(job_id, owner)
        if job is None:
            raise HTTPException(404, "job_not_found")
        return job

    def submit(kind, payload, owner, key):
        try:
            job, new = store.submit(owner, kind, payload, key or str(uuid.uuid4()))
        except Conflict:
            raise HTTPException(409, "idempotency_conflict") from None
        if new:
            try:
                dispatcher(job["id"])
            except Exception:
                # The persisted queued job remains recoverable with --recover.
                raise HTTPException(
                    503, {"code": "queue_unavailable", "job_id": job["id"]}
                ) from None
        return {"job_id": job["id"], "status": job["status"]}

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(response: Response):
        status = model.status()
        if status["status"] != "ready":
            response.status_code = 503
        return status

    @app.get("/v1/models/current")
    def current(owner=Depends(tenant)):
        return model.status()

    @app.post("/v1/imports", status_code=202)
    def imports(
        request: ImportRequest,
        owner=Depends(tenant),
        idempotency_key: str | None = Header(default=None, max_length=128),
    ):
        return submit("import", request.model_dump(), owner, idempotency_key)

    @app.post("/v1/collections/threads", status_code=202)
    def collection(
        request: CollectionRequest,
        owner=Depends(tenant),
        idempotency_key: str | None = Header(default=None, max_length=128),
    ):
        return submit("threads", request.model_dump(), owner, idempotency_key)

    @app.post("/v1/analyses", status_code=202)
    def analyses(
        request: AnalysisRequest,
        owner=Depends(tenant),
        idempotency_key: str | None = Header(default=None, max_length=128),
    ):
        source = own_job(request.source_job_id, owner)
        if (
            source["kind"] not in ("import", "threads")
            or source["status"] != "completed"
        ):
            raise HTTPException(409, "source_not_ready")
        return submit("analysis", request.model_dump(), owner, idempotency_key)

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, owner=Depends(tenant)):
        job = own_job(job_id, owner)
        result = job["result"]
        summary = {
            k: v for k, v in (result or {}).items() if k not in ("dataset", "results")
        }
        return {
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "error": job["error"],
            "summary": summary,
            "next_cursor": (job.get("checkpoint") or {}).get("cursor"),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    @app.get("/v1/analyses/{job_id}")
    def result(job_id: str, owner=Depends(tenant)):
        job = own_job(job_id, owner)
        if job["kind"] != "analysis":
            raise HTTPException(404, "analysis_not_found")
        if job["status"] != "completed":
            raise HTTPException(409, "analysis_not_ready")
        return job["result"]

    return app
