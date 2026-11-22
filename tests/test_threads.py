import httpx
import pytest

from engine.threads import ThreadsCollector, CollectionError


def test_pagination_deduplicates_and_never_trusts_next_url():
    seen = []

    def endpoint(request):
        seen.append(str(request.url))
        if "after" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "p",
                            "username": "alice",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "text": "hi",
                        }
                    ],
                    "paging": {
                        "next": "https://evil.invalid/?access_token=secret",
                        "cursors": {"after": "cursor1"},
                    },
                },
            )
        assert request.url.params["after"] == "cursor1"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "p",
                        "username": "alice",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "text": "hi",
                    },
                    {
                        "id": "q",
                        "username": "bob",
                        "timestamp": "2026-01-01T01:00:00Z",
                        "text": "hello",
                        "is_reply": False,
                    },
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as client:
        result, checkpoint = ThreadsCollector("secret", client=client).collect(
            "topic", max_pages=2
        )
    assert len(result.posts) == 2
    assert result.posts[0].is_reply is None
    assert result.observation.complete is False
    assert result.observation.source == "keyword_search"
    assert all("evil.invalid" not in url and "secret" not in url for url in seen)
    assert checkpoint is None


def test_retry_exhaustion_expired_token_and_checkpoint():
    attempts = []

    def limited(request):
        attempts.append(1)
        return httpx.Response(
            429, headers={"Retry-After": "0"}, json={"error": {"code": 4}}
        )

    with httpx.Client(transport=httpx.MockTransport(limited)) as client:
        with pytest.raises(CollectionError, match="rate_limited"):
            ThreadsCollector("secret", client=client, sleep=lambda _: None).collect("x")
    assert len(attempts) == 3
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                400, json={"error": {"code": 190, "message": "secret"}}
            )
        )
    ) as client:
        with pytest.raises(CollectionError, match="token_expired") as exc:
            ThreadsCollector("secret", client=client).collect("x")
    assert "secret" not in str(exc.value)
    page = {
        "data": [{"id": "p", "username": "alice", "timestamp": "2026-01-01T00:00:00Z"}],
        "paging": {"cursors": {"after": "resume"}, "next": "unused"},
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=page))
    ) as client:
        data, checkpoint = ThreadsCollector("secret", client=client).collect(
            "x", max_pages=1
        )
    assert checkpoint == "resume"
    assert data.observation.truncated is True


@pytest.mark.parametrize(
    "rows", [[], [{"id": "p", "username": "a", "timestamp": "2026-01-01T00:00:00Z"}]]
)
def test_single_page_without_paging_terminates_normally(rows):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": rows})
        )
    ) as client:
        data, cursor = ThreadsCollector("secret", client=client).collect("topic")
    assert len(data.posts) == len(rows)
    assert cursor is None
