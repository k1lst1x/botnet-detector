"""Disk-backed audit of every record in one third-party tweet shard."""

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import time

import ijson

from .datasets import _index_csv

NORMALIZATION_VERSION = "native-tweet-v1"


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.bytes += len(chunk)
        return chunk

    def readinto(self, buffer):
        n = self.stream.readinto(buffer)
        if n:
            self.digest.update(memoryview(buffer)[:n])
            self.bytes += n
        return n


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_types(refs):
    if refs is None:
        return None
    allowed = {"replied_to", "retweeted", "quoted"}
    if not isinstance(refs, list) or any(
        not isinstance(ref, dict)
        or not isinstance(ref.get("type"), str)
        or ref["type"] not in allowed
        for ref in refs
    ):
        raise ValueError("Invalid tweet reference type")
    return {ref["type"] for ref in refs}


def canonical_tweet(raw, author):
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError("Missing tweet identity")
    pid = str(raw["id"])
    if pid.isdigit():
        pid = "t" + pid
    stamp = raw.get("created_at")
    if not isinstance(stamp, str):
        raise ValueError("Missing tweet timestamp")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Naive tweet timestamp")
    text = raw.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError("Invalid tweet text")
    refs = raw.get("referenced_tweets")
    types = reference_types(refs)
    post = {
        "platform": "twitter",
        "external_post_id": pid,
        "author_id": author,
        "created_at": parsed.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        ),
        "text": text,
        "is_reply": ("replied_to" in types) if refs is not None else None,
        "is_repost": ("retweeted" in types) if refs is not None else None,
        "is_quote": ("quoted" in types) if refs is not None else None,
    }
    return pid, json.dumps(post, ensure_ascii=False, separators=(",", ":"))


def index_shard(source, official, work, seed=42, resume=False, expected_sha256=None):
    source, official, work = Path(source), Path(official), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SQLITE_TMPDIR", str(work.resolve()))
    identity = {
        "source": str(source.resolve()),
        "bytes": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "labels_sha256": file_hash(official / "label.csv"),
        "splits_sha256": file_hash(official / "split.csv"),
        "profiles_bytes": (official / "user.json").stat().st_size,
        "profiles_mtime_ns": (official / "user.json").stat().st_mtime_ns,
        "seed": seed,
        "normalization_version": NORMALIZATION_VERSION,
        "expected_sha256": expected_sha256,
    }
    state_path = work / "state.json"
    db_path = work / "tweets.sqlite3"
    if db_path.exists():
        if not resume:
            raise ValueError("Index exists; use resume or a new work directory")
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("Index identity mismatch")
        if state.get("phase") != "index_complete":
            raise ValueError(
                "Incomplete index cannot be resumed; choose a new work directory"
            )
        return sqlite3.connect(db_path), json.loads((work / "audit.json").read_text())
    state_path.write_text(
        json.dumps({"identity": identity, "phase": "indexing"}, indent=2) + "\n"
    )
    db = sqlite3.connect(db_path)
    start = time.monotonic()
    try:
        db.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-65536; PRAGMA temp_store=FILE; CREATE TABLE labels(id TEXT PRIMARY KEY,value TEXT); CREATE TABLE splits(id TEXT PRIMARY KEY,value TEXT); CREATE TABLE accounts(id TEXT PRIMARY KEY,label INTEGER NOT NULL,split TEXT NOT NULL,profile_seen INTEGER DEFAULT 0); CREATE TABLE tweets(id TEXT PRIMARY KEY,author TEXT,payload TEXT NOT NULL,sample_key BLOB NOT NULL) WITHOUT ROWID;"
        )
        _index_csv(db, official / "label.csv", "labels", "label", {"bot", "human"})
        _index_csv(
            db, official / "split.csv", "splits", "split", {"train", "val", "test"}
        )
        if (
            db.execute("SELECT id FROM labels EXCEPT SELECT id FROM splits").fetchone()
            or db.execute(
                "SELECT id FROM splits EXCEPT SELECT id FROM labels"
            ).fetchone()
        ):
            raise ValueError("Official label/split identity mismatch")
        db.execute(
            "INSERT INTO accounts(id,label,split) SELECT l.id,CASE l.value WHEN 'bot' THEN 1 ELSE 0 END,s.value FROM labels l JOIN splits s ON s.id=l.id"
        )
        db.commit()

        @lru_cache(maxsize=65536)
        def author_id(value):
            if db.execute("SELECT 1 FROM accounts WHERE id=?", (value,)).fetchone():
                return value
            prefixed = "u" + value
            if (
                value.isdigit()
                and db.execute(
                    "SELECT 1 FROM accounts WHERE id=?", (prefixed,)
                ).fetchone()
            ):
                return prefixed
            return None

        audit = {
            "records": 0,
            "duplicate_records": 0,
            "deduplication_scope": "canonical_model_input; unused raw metadata may differ",
            "unknown_author_records": 0,
            "source_fields": {},
            "source_bytes": identity["bytes"],
        }
        field_counts = Counter()
        batch = []
        min_time = None
        max_time = None

        def flush():
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO tweets VALUES (?,?,?,?)", batch)
            duplicates = len(batch) - (db.total_changes - before)
            if duplicates:
                for pid, author, payload, key in batch:
                    stored = db.execute(
                        "SELECT author,payload FROM tweets WHERE id=?", (pid,)
                    ).fetchone()
                    if stored != (author, payload):
                        raise ValueError(f"Conflicting tweet identity: {pid}")
            audit["duplicate_records"] += duplicates
            db.commit()
            batch.clear()

        with source.open("rb") as stream:
            reader = HashReader(stream)
            try:
                for raw in ijson.items(reader, "item", use_float=True):
                    if not isinstance(raw, dict):
                        raise ValueError("Tweet must be an object")
                    native = str(raw.get("author_id") or "")
                    known = author_id(native) if native else None
                    pid, payload = canonical_tweet(raw, known or native)
                    key = hashlib.blake2b(
                        f"{seed}:{pid}".encode(), digest_size=8
                    ).digest()
                    batch.append((pid, known, payload, key))
                    audit["records"] += 1
                    audit["unknown_author_records"] += known is None
                    for field in (
                        "text",
                        "created_at",
                        "referenced_tweets",
                        "attachments",
                        "entities",
                    ):
                        field_counts[field] += raw.get(field) is not None
                    stamp = json.loads(payload)["created_at"]
                    min_time = min(min_time, stamp) if min_time else stamp
                    max_time = max(max_time, stamp) if max_time else stamp
                    if len(batch) >= 5000:
                        flush()
                    if audit["records"] % 100000 == 0:
                        progress = {
                            "phase": "indexing",
                            "records": audit["records"],
                            "bytes_read": reader.bytes,
                            "percent": round(100 * reader.bytes / identity["bytes"], 2),
                            "elapsed_s": round(time.monotonic() - start, 1),
                            "peak_rss_mb": round(
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                / 1024,
                                1,
                            ),
                        }
                        (work / "progress.json").write_text(json.dumps(progress) + "\n")
                        print(json.dumps(progress), flush=True)
            except ijson.JSONError as exc:
                raise ValueError("Invalid tweet JSON") from exc
            if batch:
                flush()
            actual_hash = reader.digest.hexdigest()
            if reader.bytes != identity["bytes"]:
                raise ValueError("Source size changed during scan")
            if expected_sha256 and actual_hash != expected_sha256:
                raise ValueError("Source SHA256 mismatch")
        print("Building author/sample index", flush=True)
        db.execute("CREATE INDEX tweets_author_sample ON tweets(author,sample_key,id)")
        db.execute(
            "CREATE TABLE author_counts AS SELECT author,count(*) AS n FROM tweets WHERE author IS NOT NULL GROUP BY author"
        )
        db.execute("CREATE UNIQUE INDEX author_counts_id ON author_counts(author)")
        db.commit()
        audit.update(
            {
                "source_sha256": actual_hash,
                "unique_posts": db.execute("SELECT count(*) FROM tweets").fetchone()[0],
                "unique_known_posts": db.execute(
                    "SELECT count(*) FROM tweets WHERE author IS NOT NULL"
                ).fetchone()[0],
                "known_authors": db.execute(
                    "SELECT count(*) FROM author_counts"
                ).fetchone()[0],
                "official_accounts": db.execute(
                    "SELECT count(*) FROM accounts"
                ).fetchone()[0],
                "source_fields": dict(field_counts),
                "min_timestamp": min_time,
                "max_timestamp": max_time,
                "elapsed_s": round(time.monotonic() - start, 1),
                "peak_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                ),
            }
        )
        sensitivity = {}
        for threshold in (1, 5, 10, 20, 50, 100):
            sensitivity[str(threshold)] = [
                {"split": s, "label": int(y), "accounts": n}
                for s, y, n in db.execute(
                    "SELECT a.split,a.label,count(*) FROM author_counts c JOIN accounts a ON a.id=c.author WHERE c.n>=? GROUP BY a.split,a.label ORDER BY a.split,a.label",
                    (threshold,),
                )
            ]
        audit["minimum_posts_sensitivity"] = sensitivity
        (work / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        state_path.write_text(
            json.dumps({"identity": identity, "phase": "index_complete"}, indent=2)
            + "\n"
        )
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return db, audit
    except BaseException:
        db.close()
        raise
