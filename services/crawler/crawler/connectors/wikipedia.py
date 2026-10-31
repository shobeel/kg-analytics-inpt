"""Wikipedia via the MediaWiki Action API. raw_content = parsed article HTML."""

import logging
from typing import Iterator
from urllib.parse import quote

from ..http import PoliteSession
from ..models import BronzeDocument

log = logging.getLogger(__name__)


def article_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,'")


def resolve_titles(http: PoliteSession, endpoint: str, titles: list[str]) -> list[dict]:
    """Resolve redirects and get page ids + last revision timestamps, 50 titles per call."""
    pages = []
    for i in range(0, len(titles), 50):
        resp = http.get_api(endpoint, params={
            "action": "query", "titles": "|".join(titles[i:i + 50]), "redirects": 1,
            "prop": "info|revisions", "rvprop": "timestamp|ids",
            "format": "json", "formatversion": 2})
        data = resp.json()["query"]
        for p in data["pages"]:
            if p.get("missing") or p.get("invalid"):
                log.warning("wikipedia: no page for %r", p.get("title"))
                continue
            pages.append(p)
    return pages


def collect(cfg: dict, http: PoliteSession) -> Iterator[BronzeDocument]:
    endpoint = cfg["endpoint"]
    for page in resolve_titles(http, endpoint, cfg["titles"]):
        title = page["title"]
        rev = (page.get("revisions") or [{}])[0]
        resp = http.get_api(endpoint, params={
            "action": "parse", "pageid": page["pageid"], "prop": "text",
            "disableeditsection": 1, "disabletoc": 1, "disablelimitreport": 1,
            "format": "json", "formatversion": 2})
        html = resp.json()["parse"]["text"]
        yield BronzeDocument.create(
            source="wikipedia",
            url=article_url(title),
            title=title,
            published_at=rev.get("timestamp"),          # last revision time
            raw_content=html,
            content_type="text/html",
            metadata={"pageid": page["pageid"], "revid": rev.get("revid"), "lang": "en"},
        )
