"""Local folder connector: PDF (base64) and HTML files from the mounted data/local folder."""

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..models import BronzeDocument

log = logging.getLogger(__name__)
TYPES = {".pdf": "application/pdf", ".html": "text/html", ".htm": "text/html"}


def collect(cfg: dict, http=None) -> Iterator[BronzeDocument]:
    root = Path(cfg.get("path", "/data/local"))
    exts = {e.lower() for e in cfg.get("extensions", TYPES)}
    if not root.is_dir():
        log.warning("local: %s not found", root)
        return
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        ctype = TYPES[path.suffix.lower()]
        is_pdf = ctype == "application/pdf"
        sidecar = path.with_suffix(path.suffix + ".url")          # optional: original source URL
        origin = sidecar.read_text().strip() if sidecar.exists() else None
        yield BronzeDocument.create(
            source="local",
            url=f"local://{rel}",
            title=path.stem.replace("_", " "),
            published_at=None,
            raw_content=base64.b64encode(data).decode("ascii") if is_pdf else data.decode("utf-8", "replace"),
            content_type=ctype,
            content_encoding="base64" if is_pdf else "utf-8",
            metadata={"filename": rel, "size_bytes": len(data), "origin_url": origin,
                      "file_mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()},
        )
