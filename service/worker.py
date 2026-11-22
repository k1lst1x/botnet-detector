"""Worker entry point. Tasks run domain logic, never Django or model training."""

import logging

from celery import Celery

from engine.contracts import Dataset
from engine.datasets import read_import
from engine.inference import analyze_dataset
from engine.model import ModelService
from engine.threads import CollectionError, ThreadsCollector
from .settings import Settings
from .store import JobStore

logger = logging.getLogger(__name__)
settings = Settings()
celery = Celery("botnet", broker=settings.broker_url)
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_connection_timeout=3,
    task_publish_retry_policy={
        "max_retries": 1,
        "interval_start": 0,
        "interval_step": 0,
        "interval_max": 1,
    },
)


def execute_job(job_id, config=None):
    config = config or Settings()
    store = JobStore(config.database_url)
    try:
        lease_token = store.claim(job_id)
        if not lease_token:
            return
        job = store.get(job_id)
        try:
            payload = job["payload"]
            if job["kind"] == "import":
                dataset = read_import(payload["content"], payload["format"])
                result = {
                    "dataset": dataset.model_dump(mode="json"),
                    "accounts": len(dataset.accounts),
                    "posts": len(dataset.posts),
                }
            elif job["kind"] == "threads":
                data, cursor = ThreadsCollector(config.threads_token).collect(
                    payload["query"],
                    payload["max_pages"],
                    payload.get("after"),
                    checkpoint_callback=lambda cursor, page: store.checkpoint(
                        job_id, lease_token, cursor, page
                    ),
                )
                result = {
                    "dataset": data.model_dump(mode="json"),
                    "next_cursor": cursor,
                    "accounts": len(data.accounts),
                    "posts": len(data.posts),
                }
            elif job["kind"] == "analysis":
                source = store.get(payload["source_job_id"], job["tenant"])
                if (
                    not source
                    or source["status"] != "completed"
                    or source["kind"] not in ("import", "threads")
                ):
                    raise ValueError("Source unavailable")
                data = Dataset.model_validate(source["result"]["dataset"])
                result = analyze_dataset(data, ModelService(config.model_path))
            else:
                raise ValueError("Unknown job kind")
            store.finish(job_id, lease_token, result=result)
        except CollectionError as exc:
            store.finish(job_id, lease_token, error=str(exc))
        except (ValueError, KeyError, TypeError):
            store.finish(job_id, lease_token, error="invalid_input")
        except Exception:
            # Avoid logging raw payloads, API responses or credential-bearing exception strings.
            logger.error("Job %s failed with internal error", job_id)
            store.finish(job_id, lease_token, error="internal_error")
    finally:
        store.engine.dispose()


@celery.task(name="botnet.execute_job")
def run_job(job_id):
    execute_job(job_id)


def dispatch(job_id):
    run_job.delay(job_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Republish queued/lost jobs after broker/worker outage",
    )
    args = parser.parse_args()
    if args.recover:
        store = JobStore(Settings().database_url)
        try:
            for job_id in store.recover():
                dispatch(job_id)
        finally:
            store.engine.dispose()
