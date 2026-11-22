"""Engine-owned persisted jobs. SQLite for dev, PostgreSQL for deployment."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    insert,
    select,
    update,
    text,
)
from sqlalchemy.exc import IntegrityError

metadata = MetaData()
jobs = Table(
    "engine_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("tenant", String(128), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("idempotency_key", String(128), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("status", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("result", JSON),
    Column("error", String(64)),
    Column("checkpoint", JSON),
    Column("lease_token", String(36)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("tenant", "kind", "idempotency_key"),
)


class Conflict(ValueError):
    pass


class JobStore:
    def __init__(self, url):
        self.engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30}
            if url.startswith("sqlite")
            else {},
            pool_pre_ping=True,
        )
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(78362418)"))
            metadata.create_all(conn)

    def get(self, id, tenant=None):
        statement = select(jobs).where(jobs.c.id == id)
        if tenant is not None:
            statement = statement.where(jobs.c.tenant == tenant)
        with self.engine.connect() as conn:
            row = conn.execute(statement).mappings().first()
        return dict(row) if row else None

    def submit(self, tenant, kind, payload, key):
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        now = datetime.now(timezone.utc)
        record = {
            "id": str(uuid.uuid4()),
            "tenant": tenant,
            "kind": kind,
            "idempotency_key": key,
            "request_hash": digest,
            "status": "queued",
            "payload": payload,
            "created_at": now,
            "updated_at": now,
        }
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(jobs).values(**record))
            return record, True
        except IntegrityError:
            with self.engine.connect() as conn:
                row = (
                    conn.execute(
                        select(jobs).where(
                            jobs.c.tenant == tenant,
                            jobs.c.kind == kind,
                            jobs.c.idempotency_key == key,
                        )
                    )
                    .mappings()
                    .one()
                )
            if row["request_hash"] != digest:
                raise Conflict("Idempotency key reused with another payload")
            return dict(row), False

    def claim(self, id):
        now = datetime.now(timezone.utc)
        token = str(uuid.uuid4())
        with self.engine.begin() as conn:
            result = conn.execute(
                update(jobs)
                .where(jobs.c.id == id, jobs.c.status == "queued")
                .values(status="running", updated_at=now, error=None, lease_token=token)
            )
        return token if result.rowcount == 1 else None

    def finish(self, id, lease_token, result=None, error=None):
        with self.engine.begin() as conn:
            conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == id,
                    jobs.c.status == "running",
                    jobs.c.lease_token == lease_token,
                )
                .values(
                    status="failed" if error else "completed",
                    result=result,
                    error=error,
                    updated_at=datetime.now(timezone.utc),
                )
            )

    def checkpoint(self, id, lease_token, cursor, page):
        with self.engine.begin() as conn:
            current = conn.execute(
                select(jobs.c.checkpoint).where(
                    jobs.c.id == id,
                    jobs.c.status == "running",
                    jobs.c.lease_token == lease_token,
                )
            ).first()
            if current is None:
                return
            pages = (current[0] or {}).get("pages", [])
            pages.append(page)
            conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == id,
                    jobs.c.status == "running",
                    jobs.c.lease_token == lease_token,
                )
                .values(
                    checkpoint={"cursor": cursor, "pages": pages},
                    updated_at=datetime.now(timezone.utc),
                )
            )

    def recover(self):
        """Requeue lost workers after a one-hour lease; queued jobs can be republished."""
        with self.engine.begin() as conn:
            conn.execute(
                update(jobs)
                .where(
                    jobs.c.status == "running",
                    jobs.c.updated_at < datetime.now(timezone.utc) - timedelta(hours=1),
                )
                .values(status="queued", lease_token=None)
            )
            return list(
                conn.execute(
                    select(jobs.c.id).where(jobs.c.status == "queued")
                ).scalars()
            )
