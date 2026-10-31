"""Collect public Threads posts by keyword via the official Threads API.

Requires THREADS_ACCESS_TOKEN with the `threads_keyword_search` permission.
Docs: https://developers.facebook.com/docs/threads/keyword-search
"""
import json
import urllib.parse
import urllib.request

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.importer import import_records

API = "https://graph.threads.net/v1.0/keyword_search"
FIELDS = "id,text,username,timestamp,permalink,is_reply,media_type"


class Command(BaseCommand):
    help = "Сбор публичных постов Threads по ключевым словам (официальный API)"

    def add_arguments(self, parser):
        parser.add_argument("queries", nargs="+", help="ключевые слова, напр. Алматы Астана")
        parser.add_argument("--type", default="RECENT", choices=["TOP", "RECENT"])
        parser.add_argument("--pages", type=int, default=5)

    def handle(self, *a, **o):
        token = settings.THREADS_ACCESS_TOKEN
        if not token:
            raise CommandError("Задайте THREADS_ACCESS_TOKEN в окружении")
        posts = []
        for q in o["queries"]:
            params = {"q": q, "search_type": o["type"], "fields": FIELDS, "access_token": token}
            url = f"{API}?{urllib.parse.urlencode(params)}"
            for _ in range(o["pages"]):
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.load(r)
                for it in data.get("data", []):
                    posts.append({"handle": it.get("username"), "post_id": f"th_{it['id']}",
                                  "created_at": it.get("timestamp"), "text": it.get("text") or "",
                                  "is_reply": it.get("is_reply", False)})
                url = data.get("paging", {}).get("next")
                if not url:
                    break
            self.stdout.write(f"«{q}»: всего собрано {len(posts)}")
        na, np_ = import_records([], posts, platform="threads")
        self.stdout.write(self.style.SUCCESS(f"Новых аккаунтов: {na}, постов: {np_}"))
