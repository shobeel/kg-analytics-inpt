"""arXiv API connector (Atom). raw_content = the paper's <entry> XML; text is built in Spark."""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Iterator

from ..http import PoliteSession
from ..models import BronzeDocument

log = logging.getLogger(__name__)
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def canonical_abs_url(entry_id: str) -> str:
    """http://arxiv.org/abs/2302.13971v1 -> https://arxiv.org/abs/2302.13971 (version-free,
    so a new version of the same paper keeps the same document id)."""
    arxiv_id = entry_id.rsplit("/abs/", 1)[-1]
    return "https://arxiv.org/abs/" + re.sub(r"v\d+$", "", arxiv_id)


def parse_feed(xml_text: str) -> list[BronzeDocument]:
    root = ET.fromstring(xml_text)
    docs = []
    for entry in root.findall("a:entry", NS):
        entry_id = entry.findtext("a:id", default="", namespaces=NS)
        if "/abs/" not in entry_id:            # API returns an error entry for bad ids
            continue
        url = canonical_abs_url(entry_id)
        title = " ".join((entry.findtext("a:title", default="", namespaces=NS)).split())
        pdf = next((l.get("href") for l in entry.findall("a:link", NS) if l.get("title") == "pdf"), None)
        docs.append(BronzeDocument.create(
            source="arxiv",
            url=url,
            title=title,
            published_at=entry.findtext("a:published", namespaces=NS),
            raw_content=ET.tostring(entry, encoding="unicode"),
            content_type="application/atom+xml",
            metadata={
                "arxiv_id": url.rsplit("/", 1)[-1],
                "authors": [a.findtext("a:name", namespaces=NS) for a in entry.findall("a:author", NS)],
                "categories": [c.get("term") for c in entry.findall("a:category", NS)],
                "primary_category": (entry.find("arxiv:primary_category", NS).get("term")
                                     if entry.find("arxiv:primary_category", NS) is not None else None),
                "updated_at": entry.findtext("a:updated", namespaces=NS),
                "pdf_url": pdf,
                "doi": entry.findtext("arxiv:doi", namespaces=NS),
            },
        ))
    return docs


def collect(cfg: dict, http: PoliteSession) -> Iterator[BronzeDocument]:
    endpoint = cfg["endpoint"]
    seed_ids = [str(i) for i in cfg.get("seed_ids", [])]   # keep ids quoted in YAML
    if seed_ids:
        resp = http.get_api(endpoint, params={"id_list": ",".join(seed_ids), "max_results": len(seed_ids)})
        docs = parse_feed(resp.text)
        got = {d.metadata["arxiv_id"] for d in docs}
        missing = [i for i in seed_ids if i not in got]
        log.info("arxiv seeds: %d/%d papers", len(docs), len(seed_ids))
        if missing:
            log.warning("arxiv seeds not returned: %s", missing)
        yield from docs

    total, page = int(cfg.get("max_results", 100)), int(cfg.get("page_size", 50))
    for start in range(0, total, page):
        resp = http.get_api(endpoint, params={
            "search_query": cfg["query"], "start": start, "max_results": min(page, total - start),
            "sortBy": cfg.get("sort_by", "relevance"), "sortOrder": "descending"})
        docs = parse_feed(resp.text)
        log.info("arxiv query start=%d: %d papers", start, len(docs))
        if not docs:
            break
        yield from docs
