import hashlib
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["arxiv", "wikipedia", "rss", "local"]


def doc_id(url: str) -> str:
    """Stable document id: sha256 of the canonical URL (first 24 hex chars)."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class BronzeDocument(BaseModel):
    """One raw document as stored in /datalake/bronze/<source>/<date>/<id>.json.

    raw_content is never cleaned here: text extraction happens in the Spark silver job.
    Binary content (PDF) is base64-encoded and flagged by content_encoding.
    """

    id: str
    source: Source
    url: str
    title: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: str = Field(default_factory=utc_now)
    raw_content: str
    content_type: str
    content_encoding: Literal["utf-8", "base64"] = "utf-8"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, *, source: Source, url: str, **kwargs: Any) -> "BronzeDocument":
        return cls(id=doc_id(url), source=source, url=url, **kwargs)
