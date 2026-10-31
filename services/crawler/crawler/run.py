"""Run the crawler: every enabled connector -> bronze, skipping ids already stored."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .connectors import CONNECTORS
from .hdfs import WebHDFS
from .http import PoliteSession
from .models import utc_now
from .store import BronzeStore

log = logging.getLogger("crawler")
DEFAULT_CONFIG = os.environ.get("CRAWLER_CONFIG", "/app/config/sources.yaml")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def crawl(sources: Optional[Iterable[str]] = None, config_path: str = DEFAULT_CONFIG,
          dry_run: bool = False) -> dict:
    cfg = load_config(config_path)
    wanted = list(sources) if sources else [s for s in CONNECTORS if cfg.get(s, {}).get("enabled")]
    http = PoliteSession(cfg)
    store = BronzeStore(WebHDFS())
    started, t0 = utc_now(), time.monotonic()
    log.info("bronze index: %d documents already stored", store.load_index())

    errors: dict[str, str] = {}
    for name in wanted:
        src_cfg = cfg[name]
        kwargs = {"known": store.has} if name == "rss" else {}
        try:
            for doc in CONNECTORS[name](src_cfg, http, **kwargs):
                if dry_run:
                    log.info("[dry-run] %s %s %s", doc.source, doc.id, doc.title)
                    continue
                if store.save(doc):
                    log.info("saved %-9s %s %s", doc.source, doc.id, (doc.title or "")[:70])
        except Exception as e:                      # one broken source must not stop the others
            log.exception("connector %s failed", name)
            errors[name] = f"{type(e).__name__}: {e}"

    summary = {
        "started_at": started, "finished_at": utc_now(),
        "duration_s": round(time.monotonic() - t0, 1),
        "sources": wanted, "saved": dict(store.saved), "skipped_existing": dict(store.skipped),
        "total_in_bronze": len(store.known_ids), "errors": errors, "dry_run": dry_run,
    }
    if not dry_run:
        store.fs.write(f"{store.root}/_runs/{started.replace(':', '')}.json",
                       json.dumps(summary).encode("utf-8"))
    return summary
