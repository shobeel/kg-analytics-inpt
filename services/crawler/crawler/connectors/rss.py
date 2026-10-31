"""Tech news RSS/Atom feeds. Each item becomes a document; when fetch_full_article is on,
the linked page is fetched (robots.txt checked) and the feed text is the fallback."""

import logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Iterator, Optional

import requests

from ..http import PoliteSession, RobotsDisallowed
from ..models import BronzeDocument

log = logging.getLogger(__name__)
ATOM = "{http://www.w3.org/2005/Atom}"
CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"


def _iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    try:
        return parsedate_to_datetime(value).isoformat()      # RSS: RFC 822
    except (TypeError, ValueError):
        return value                                         # Atom: already ISO 8601


def parse_feed(xml_text: str) -> list[dict]:
    """Return items as dicts: title, link, published, content (HTML)."""
    root = ET.fromstring(xml_text)
    items = []
    for it in root.iter("item"):                              # RSS 2.0
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "link": (it.findtext("link") or it.findtext("guid") or "").strip(),
            "published": _iso_date(it.findtext("pubDate")),
            "content": it.findtext(CONTENT) or it.findtext("description") or "",
            "categories": [c.text for c in it.findall("category") if c.text],
        })
    for it in root.iter(f"{ATOM}entry"):                      # Atom
        link = next((l.get("href") for l in it.findall(f"{ATOM}link")
                     if l.get("rel", "alternate") == "alternate"), "")
        items.append({
            "title": (it.findtext(f"{ATOM}title") or "").strip(),
            "link": (link or "").strip(),
            "published": _iso_date(it.findtext(f"{ATOM}published") or it.findtext(f"{ATOM}updated")),
            "content": it.findtext(f"{ATOM}content") or it.findtext(f"{ATOM}summary") or "",
            "categories": [c.get("term") for c in it.findall(f"{ATOM}category") if c.get("term")],
        })
    return [i for i in items if i["link"]]


def collect(cfg: dict, http: PoliteSession, known=lambda _id: False) -> Iterator[BronzeDocument]:
    max_items = int(cfg.get("max_items_per_feed", 30))
    full = bool(cfg.get("fetch_full_article", False))
    for feed in cfg["feeds"]:
        try:
            resp = http.get_web(feed["url"])
            items = parse_feed(resp.content)
        except (requests.RequestException, RobotsDisallowed, ET.ParseError) as e:
            log.warning("rss %s: feed failed: %s", feed["name"], e)
            continue
        log.info("rss %s: %d items (keeping %d)", feed["name"], len(items), min(len(items), max_items))
        for item in items[:max_items]:
            doc = BronzeDocument.create(
                source="rss", url=item["link"], title=item["title"], published_at=item["published"],
                raw_content=item["content"], content_type="text/html",
                metadata={"feed": feed["name"], "feed_url": feed["url"],
                          "categories": item["categories"], "full_article": False})
            if known(doc.id):
                yield doc                        # store will skip it; avoids a page fetch
                continue
            if full:
                try:
                    page = http.get_web(item["link"])
                    if "html" in page.headers.get("Content-Type", "") and page.text:
                        doc.raw_content = page.text
                        doc.metadata.update(full_article=True, feed_summary=item["content"][:2000])
                except RobotsDisallowed:
                    doc.metadata["full_article_skipped"] = "robots.txt"
                except requests.RequestException as e:
                    doc.metadata["full_article_skipped"] = f"error: {type(e).__name__}"
            yield doc
