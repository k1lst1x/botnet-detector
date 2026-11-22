"""Versioned, framework-independent boundary shared by training and inference."""

from datetime import timezone
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

Platform = Literal["twitter", "threads"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("*")
    @classmethod
    def utc(cls, value):
        if hasattr(value, "astimezone"):
            return value.astimezone(timezone.utc)
        return value


class AccountSnapshot(Contract):
    platform: Platform
    external_user_id: str = Field(min_length=1, max_length=256)
    username: str | None = None
    collected_at: AwareDatetime | None = None
    created_at: AwareDatetime | None = None
    followers: int | None = Field(default=None, ge=0)
    following: int | None = Field(default=None, ge=0)
    biography: str | None = None
    has_avatar: bool | None = None
    available_fields: list[str] = Field(default_factory=list)
    identity_source: Literal["platform_id", "username_fallback"] = "platform_id"

    @property
    def key(self):
        return self.platform, self.external_user_id


class Post(Contract):
    platform: Platform
    external_post_id: str = Field(min_length=1, max_length=256)
    author_id: str = Field(min_length=1, max_length=256)
    created_at: AwareDatetime
    text: str | None = None
    is_reply: bool | None = None
    is_repost: bool | None = None
    is_quote: bool | None = None
    has_media: bool | None = None
    targets: list[str] | None = None
    urls: list[str] | None = None


class Observation(Contract):
    start: AwareDatetime
    end: AwareDatetime
    source: str = Field(min_length=1)
    complete: bool = False
    truncated: bool = False
    query: str | None = None

    @model_validator(mode="after")
    def valid_window(self):
        if self.end < self.start:
            raise ValueError("Observation end precedes start")
        if self.complete and (
            self.truncated or self.source == "keyword_search" or self.end == self.start
        ):
            raise ValueError("Cannot claim a complete observation")
        return self


class Dataset(Contract):
    schema_version: Literal["1"] = "1"
    accounts: list[AccountSnapshot] = Field(max_length=10000)
    posts: list[Post] = Field(max_length=100000)
    observation: Observation

    @model_validator(mode="after")
    def consistent(self):
        accounts = {a.key for a in self.accounts}
        if len(accounts) != len(self.accounts):
            raise ValueError("Duplicate account identity")
        seen = set()
        for post in self.posts:
            key = post.platform, post.external_post_id
            if key in seen:
                raise ValueError("Duplicate post identity")
            seen.add(key)
            if (post.platform, post.author_id) not in accounts:
                raise ValueError("Post author not present in accounts")
            if not self.observation.start <= post.created_at <= self.observation.end:
                raise ValueError("Post outside observation window")
        return self
