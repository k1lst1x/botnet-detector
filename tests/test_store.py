from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from service.store import JobStore, jobs


def test_expired_worker_cannot_finish_a_new_attempt(tmp_path):
    store = JobStore(f"sqlite:///{tmp_path}/jobs.sqlite3")
    job, _ = store.submit("tenant", "import", {"format": "csv", "content": "x"}, "key")
    old_lease = store.claim(job["id"])
    with store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.id == job["id"])
            .values(updated_at=datetime.now(timezone.utc) - timedelta(hours=2))
        )
    assert job["id"] in store.recover()
    new_lease = store.claim(job["id"])
    assert old_lease != new_lease
    store.finish(job["id"], old_lease, result={"attempt": "old"})
    assert store.get(job["id"])["status"] == "running"
    store.finish(job["id"], new_lease, result={"attempt": "new"})
    assert store.get(job["id"])["result"] == {"attempt": "new"}
