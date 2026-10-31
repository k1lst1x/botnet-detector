"""CSV / dict import into the database.

accounts.csv: handle, display_name, followers, following, created_at, label, true_group
posts.csv:    handle, post_id, created_at, text, is_reply, is_repost, reply_to
Any extra columns are ignored; only handle (+ post_id, created_at, text for posts) are required.
"""
import csv
import io
from datetime import datetime, timezone

from django.db import transaction
from django.utils.dateparse import parse_datetime

from analysis.text import extract_entities

from .models import Account, Post

TRUE = {"1", "true", "yes", "y", "t", "да"}
LABEL_MAP = {"1": "bot", "bot": "bot", "0": "human", "human": "human", "genuine": "human"}


def _dt(v):
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    v = (v or "").strip()
    if not v:
        return None
    if v.replace(".", "").isdigit():
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    d = parse_datetime(v.replace("Z", "+00:00"))
    if d is None:
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                d = datetime.strptime(v, fmt)
                break
            except ValueError:
                continue
    if d is not None and d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _bool(v):
    return v if isinstance(v, bool) else str(v or "").strip().lower() in TRUE


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


@transaction.atomic
def import_records(accounts, posts, platform="threads"):
    existing = {a.handle: a for a in Account.objects.all()}
    new = []
    for r in accounts:
        h = (r.get("handle") or "").strip().lstrip("@").lower()
        if not h or h in existing:
            continue
        a = Account(handle=h, display_name=r.get("display_name") or "", platform=platform,
                    followers=_int(r.get("followers")), following=_int(r.get("following")),
                    created_at=_dt(r.get("created_at")),
                    label=LABEL_MAP.get(str(r.get("label", "")).strip().lower(), "unknown"),
                    true_group=r.get("true_group") or "")
        existing[h] = a
        new.append(a)
    for r in posts:  # auto-create authors that only appear in posts
        h = (r.get("handle") or "").strip().lstrip("@").lower()
        if h and h not in existing:
            a = Account(handle=h, platform=platform)
            existing[h] = a
            new.append(a)
    Account.objects.bulk_create(new, batch_size=1000)
    ids = dict(Account.objects.values_list("handle", "id"))
    known = set(Post.objects.values_list("external_id", flat=True))
    objs = []
    for r in posts:
        h = (r.get("handle") or "").strip().lstrip("@").lower()
        pid = str(r.get("post_id") or "").strip()
        ts = _dt(r.get("created_at"))
        if not h or not pid or ts is None or pid in known:
            continue
        known.add(pid)
        text = r.get("text") or ""
        urls, tags, mentions = extract_entities(text)
        objs.append(Post(account_id=ids[h], external_id=pid, text=text, created_at=ts,
                         is_reply=_bool(r.get("is_reply")), is_repost=_bool(r.get("is_repost")) or text.startswith("RT @"),
                         reply_to=(r.get("reply_to") or "").lstrip("@").lower(),
                         urls=urls, hashtags=tags, mentions=mentions))
    Post.objects.bulk_create(objs, batch_size=2000)
    return len(new), len(objs)


def read_csv(fileobj):
    data = fileobj.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(data)))
