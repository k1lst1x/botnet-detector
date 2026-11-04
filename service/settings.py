from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "ENGINE_DATABASE_URL", "sqlite:///engine.sqlite3"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("ENGINE_API_KEY", ""))
    model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("ENGINE_MODEL_PATH", "media/models/portable.joblib")
        )
    )
    broker_url: str = field(
        default_factory=lambda: os.getenv(
            "CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"
        )
    )
    threads_token: str = field(
        default_factory=lambda: os.getenv("THREADS_ACCESS_TOKEN", "")
    )
