"""Streaming TwiBot preparation and small external imports. No Django dependency."""

from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
import tempfile

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import AccountSnapshot, Dataset, Observation, Post
from .features import FEATURE_NAMES, FEATURE_VERSION, extract_features


def _boolean(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError("Invalid boolean")


def read_import(content: str, format: str) -> Dataset:
    """Accept canonical Dataset JSON, or CSV/JSONL post rows with profile columns."""
    if format == "json":
        return Dataset.model_validate_json(content)
    if format == "csv":
        rows = csv.DictReader(io.StringIO(content))
    elif format == "jsonl":
        rows = (json.loads(line) for line in content.splitlines() if line.strip())
    else:
        raise ValueError("Unsupported import format")
    accounts, posts = {}, []
    for row in rows:
        platform = row.get("platform") or "threads"
        author = str(row.get("author_id") or row.get("external_user_id") or "").strip()
        identity_source = "platform_id"
        if not author:
            handle = (
                str(row.get("handle") or row.get("username") or "")
                .strip()
                .lstrip("@")
                .casefold()
            )
            if not handle:
                raise ValueError("Missing author identity")
            author = f"username:{handle}"
            identity_source = "username_fallback"
        account = AccountSnapshot(
            platform=platform,
            external_user_id=author,
            identity_source=identity_source,
            username=row.get("username") or row.get("handle"),
            followers=row.get("followers")
            if row.get("followers") not in ("", None)
            else None,
            following=row.get("following")
            if row.get("following") not in ("", None)
            else None,
            biography=row.get("biography"),
            has_avatar=_boolean(row.get("has_avatar")),
            display_name=row.get("display_name") or row.get("name") or None,
            is_verified=_boolean(row.get("is_verified")),
            avatar_default=_boolean(row.get("avatar_default")),
        )
        key = account.key
        if key in accounts and accounts[key] != account:
            raise ValueError("Conflicting account snapshots")
        accounts[key] = account
        posts.append(
            Post(
                platform=platform,
                author_id=author,
                external_post_id=str(
                    row.get("post_id") or row.get("external_post_id") or ""
                ),
                created_at=row.get("created_at"),
                text=row.get("text"),
                is_reply=_boolean(row.get("is_reply")),
                is_repost=_boolean(row.get("is_repost")),
                is_quote=_boolean(row.get("is_quote")),
                has_media=_boolean(row.get("has_media")),
            )
        )
        if len(posts) > 100000 or len(accounts) > 10000:
            raise ValueError("Import too large; split into batches")
    if not posts:
        raise ValueError("Empty import")
    observation = Observation(
        start=min(p.created_at for p in posts),
        end=max(p.created_at for p in posts),
        source="file_import",
    )
    return Dataset(
        accounts=list(accounts.values()), posts=posts, observation=observation
    )


def _index_csv(db, path, table, field, allowed):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", field}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid {path.name} schema")
        for row in reader:
            value = row[field]
            if field == "split" and value == "valid":
                value = "val"
            if not row["id"] or value not in allowed:
                raise ValueError(f"Invalid {path.name} row")
            try:
                db.execute(f"INSERT INTO {table} VALUES (?,?)", (row["id"], value))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate id in {path.name}") from exc


def _known_user_id(db, value):
    value = str(value)
    if db.execute("SELECT 1 FROM labels WHERE id=?", (value,)).fetchone():
        return value
    prefixed = f"u{value}"
    if (
        value.isdigit()
        and db.execute("SELECT 1 FROM labels WHERE id=?", (prefixed,)).fetchone()
    ):
        return prefixed
    raise ValueError("Tweet author is absent from official labels")


def _stage_tweets(db, source):
    files = sorted(source.glob("tweet_*.json"))
    edge = source / "edge.csv"
    if files and edge.exists():
        with edge.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("relation") in ("post", "own"):
                    a, b = row["source_id"], row["target_id"]
                    user, tweet = (a, b) if a.startswith("u") else (b, a)
                    existing = db.execute(
                        "SELECT author FROM authors WHERE id=?", (tweet,)
                    ).fetchone()
                    if existing and existing[0] != user:
                        raise ValueError("Conflicting tweet authors")
                    db.execute(
                        "INSERT OR IGNORE INTO authors VALUES (?,?)", (tweet, user)
                    )
    for path in files:
        with path.open("rb") as stream:
            for raw in ijson.items(stream, "item"):
                pid = str(raw["id"])
                author = raw.get("author_id")
                mapped = db.execute(
                    "SELECT author FROM authors WHERE id=?", (pid,)
                ).fetchone()
                if not author:
                    author = mapped[0] if mapped else None
                if not author:
                    raise ValueError("Tweet author unavailable; provide edge.csv")
                author = _known_user_id(db, author)
                if mapped and author != _known_user_id(db, mapped[0]):
                    raise ValueError("Conflicting tweet authors")
                refs = raw.get("referenced_tweets")
                types = {r.get("type") for r in refs or []}
                post = Post(
                    platform="twitter",
                    external_post_id=pid,
                    author_id=str(author),
                    created_at=raw["created_at"],
                    text=raw.get("text"),
                    is_reply=("replied_to" in types) if refs is not None else None,
                    is_repost=("retweeted" in types) if refs is not None else None,
                    is_quote=("quoted" in types) if refs is not None else None,
                )
                try:
                    db.execute(
                        "INSERT INTO posts VALUES (?,?,?)",
                        (pid, str(author), post.model_dump_json()),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate tweet id") from exc
    db.execute("CREATE INDEX posts_author ON posts(author)")
    db.commit()
    return bool(files)


def prepare_twibot(
    source: Path,
    output: Path,
    limit: int | None = None,
    observation: Observation | None = None,
):
    """Disk-backed joins + fixed-schema Parquet; manifest is written last on success."""
    source, output = Path(source), Path(output)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("external_user_id", pa.string()),
            ("label", pa.int8()),
            ("split", pa.string()),
            ("observed_posts", pa.int64()),
        ]
        + [(n, pa.float64()) for n in FEATURE_NAMES]
    )
    declared_observation = observation
    counts = Counter()
    writers = {}
    buffers = {s: [] for s in ("train", "val", "test")}
    report = {
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "rows": 0,
        "counts": {},
        "sample_limit": limit,
        "behavior_available": False,
        "observation_complete": bool(
            declared_observation and declared_observation.complete
        ),
        "observation": declared_observation.model_dump(mode="json")
        if declared_observation
        else None,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="twibot-stage-") as temp:
            db = sqlite3.connect(str(Path(temp) / "index.sqlite3"))
            try:
                db.executescript(
                    "CREATE TABLE labels(id TEXT PRIMARY KEY,value TEXT); CREATE TABLE splits(id TEXT PRIMARY KEY,value TEXT); CREATE TABLE seen(id TEXT PRIMARY KEY); CREATE TABLE authors(id TEXT PRIMARY KEY,author TEXT); CREATE TABLE posts(id TEXT PRIMARY KEY,author TEXT,payload TEXT);"
                )
                _index_csv(
                    db, source / "label.csv", "labels", "label", {"bot", "human"}
                )
                _index_csv(
                    db,
                    source / "split.csv",
                    "splits",
                    "split",
                    {"train", "val", "test"},
                )
                if (
                    db.execute(
                        "SELECT id FROM labels EXCEPT SELECT id FROM splits"
                    ).fetchone()
                    or db.execute(
                        "SELECT id FROM splits EXCEPT SELECT id FROM labels"
                    ).fetchone()
                ):
                    raise ValueError("Label/split ID sets differ")
                report["behavior_available"] = _stage_tweets(db, source)
                for split in buffers:
                    writers[split] = pq.ParquetWriter(
                        output / f"{split}.parquet", schema, compression="zstd"
                    )

                def flush(split):
                    if buffers[split]:
                        writers[split].write_table(
                            pa.Table.from_pylist(buffers[split], schema=schema)
                        )
                        buffers[split].clear()

                with (source / "user.json").open("rb") as stream:
                    for raw in ijson.items(stream, "item"):
                        uid = str(raw["id"])
                        try:
                            db.execute("INSERT INTO seen VALUES (?)", (uid,))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("Duplicate user id") from exc
                        label = db.execute(
                            "SELECT value FROM labels WHERE id=?", (uid,)
                        ).fetchone()
                        split = db.execute(
                            "SELECT value FROM splits WHERE id=?", (uid,)
                        ).fetchone()
                        if not label or not split:
                            raise ValueError("Unlabelled/unsplit user")
                        metrics = raw.get("public_metrics") or {}
                        account = AccountSnapshot(
                            platform="twitter",
                            external_user_id=uid,
                            username=raw.get("username"),
                            created_at=raw.get("created_at"),
                            followers=metrics.get("followers_count"),
                            following=metrics.get("following_count"),
                            biography=raw.get("description"),
                            has_avatar=bool(raw["profile_image_url"])
                            if raw.get("profile_image_url") is not None
                            else None,
                            display_name=raw.get("name"),
                            is_verified=raw.get("verified"),
                            avatar_default="default_profile_images" in raw["profile_image_url"]
                            if raw.get("profile_image_url") is not None
                            else None,
                        )
                        posts = [
                            Post.model_validate_json(row[0])
                            for row in db.execute(
                                "SELECT payload FROM posts WHERE author=?", (uid,)
                            )
                        ]
                        # Dataset collection window is unknown: never infer complete history from min/max tweets.
                        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
                        observation = declared_observation or Observation(
                            start=min((p.created_at for p in posts), default=epoch),
                            end=max((p.created_at for p in posts), default=epoch),
                            source="twibot_archive",
                        )
                        if any(
                            not observation.start <= p.created_at <= observation.end
                            for p in posts
                        ):
                            raise ValueError(
                                "Tweet outside declared observation window"
                            )
                        features = extract_features(account, posts, observation)
                        name = split[0]
                        buffers[name].append(
                            {
                                "external_user_id": uid,
                                "label": int(label[0] == "bot"),
                                "split": name,
                                "observed_posts": len(posts),
                                **features,
                            }
                        )
                        counts[name] += 1
                        report["rows"] += 1
                        if len(buffers[name]) >= 2048:
                            flush(name)
                        if limit and report["rows"] >= limit:
                            break
                if limit is None:
                    if db.execute(
                        "SELECT id FROM labels EXCEPT SELECT id FROM seen"
                    ).fetchone():
                        raise ValueError("Missing users in user.json")
                    if db.execute(
                        "SELECT DISTINCT author FROM posts EXCEPT SELECT id FROM seen"
                    ).fetchone():
                        raise ValueError("Unknown tweet author")
                for name in buffers:
                    flush(name)
            finally:
                db.close()
    finally:
        for writer in writers.values():
            writer.close()
    report["counts"] = {s: counts[s] for s in buffers}
    report["warnings"] = (
        []
        if declared_observation
        else [
            "Historical observation window is unknown; temporal rate features remain missing."
        ]
    )
    if not report["behavior_available"]:
        report["warnings"].append("Profile-only subset: tweets and edges unavailable.")
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
