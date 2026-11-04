"""Official API adapter. Transport is injectable for preparation without Meta access."""

from datetime import datetime, timezone
import time

import httpx

from .contracts import AccountSnapshot, Dataset, Observation, Post

API = "https://graph.threads.net/v1.0/keyword_search"
FIELDS = "id,text,username,timestamp,is_reply,is_quote_post,media_type"


class CollectionError(RuntimeError):
    pass


class ThreadsCollector:
    def __init__(self, token, client=None, sleep=time.sleep):
        if not token:
            raise CollectionError("threads_not_configured")
        self.token = token
        self.client = client
        self.sleep = sleep

    def _page(self, client, params):
        for attempt in range(3):
            try:
                response = client.get(
                    API,
                    params=params,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=30,
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise CollectionError("upstream_unavailable") from None
                self.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise CollectionError(
                        "rate_limited"
                        if response.status_code == 429
                        else "upstream_unavailable"
                    )
                try:
                    delay = min(
                        30,
                        max(0, float(response.headers.get("Retry-After", 2**attempt))),
                    )
                except ValueError:
                    delay = 2**attempt
                self.sleep(delay)
                continue
            try:
                page = response.json()
            except ValueError:
                raise CollectionError("invalid_upstream_response") from None
            error = page.get("error") if isinstance(page, dict) else None
            if error or response.is_error:
                code = error.get("code") if isinstance(error, dict) else None
                if code == 190:
                    raise CollectionError("token_expired")
                if response.status_code == 403 or code in (10, 200):
                    raise CollectionError("permission_denied")
                raise CollectionError("upstream_rejected")
            if not isinstance(page, dict) or not isinstance(page.get("data"), list):
                raise CollectionError("invalid_upstream_response")
            return page

    def collect(self, query, max_pages=5, after=None, checkpoint_callback=None):
        if not query.strip() or not 1 <= max_pages <= 20:
            raise ValueError("Invalid collection bounds")
        if self.client is None:
            with httpx.Client(follow_redirects=False) as client:
                return self._collect(
                    client, query, max_pages, after, checkpoint_callback
                )
        return self._collect(self.client, query, max_pages, after, checkpoint_callback)

    def _collect(self, client, query, max_pages, after, checkpoint_callback):
        accounts = {}
        posts = {}
        raw_pages = []
        cursor = after
        for _ in range(max_pages):
            params = {
                "q": query,
                "search_type": "RECENT",
                "fields": FIELDS,
                "limit": 100,
            }
            if cursor:
                params["after"] = cursor
            page = self._page(client, params)
            # Store data and safe cursors only; paging.next can contain access_token.
            raw_pages.append({"data": page["data"]})
            try:
                for row in page["data"]:
                    # Keyword search may expose username, but no stable author id.
                    # A marked fallback cannot silently merge into a platform-id account.
                    username = row["username"]
                    author = f"username:{username.casefold()}"
                    accounts[author] = AccountSnapshot(
                        platform="threads",
                        external_user_id=author,
                        username=username,
                        collected_at=datetime.now(timezone.utc),
                        identity_source="username_fallback",
                    )
                    post = Post(
                        platform="threads",
                        external_post_id=str(row["id"]),
                        author_id=author,
                        created_at=row["timestamp"],
                        text=row.get("text"),
                        is_reply=row.get("is_reply"),
                        is_quote=row.get("is_quote_post"),
                        has_media=(row["media_type"] != "TEXT")
                        if row.get("media_type")
                        else None,
                    )
                    if (
                        post.external_post_id in posts
                        and posts[post.external_post_id] != post
                    ):
                        raise ValueError("Conflicting post")
                    posts[post.external_post_id] = post
            except (KeyError, ValueError, TypeError):
                raise CollectionError("invalid_upstream_response") from None
            paging = page.get("paging") or {}
            next_cursor = (
                (paging.get("cursors") or {}).get("after")
                if paging.get("next")
                else None
            )
            if next_cursor and next_cursor == cursor:
                raise CollectionError("pagination_loop")
            cursor = next_cursor
            if checkpoint_callback:
                checkpoint_callback(cursor, raw_pages[-1])
            if not cursor:
                break
        values = list(posts.values())
        now = datetime.now(timezone.utc)
        obs = Observation(
            start=min((p.created_at for p in values), default=now),
            end=max((p.created_at for p in values), default=now),
            source="keyword_search",
            complete=False,
            truncated=bool(cursor),
            query=query,
        )
        return Dataset(
            accounts=list(accounts.values()), posts=values, observation=obs
        ), cursor
