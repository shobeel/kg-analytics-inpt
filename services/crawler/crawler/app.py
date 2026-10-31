"""HTTP front for the crawler (used by the API service in Phase 7)."""

import logging
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .connectors import CONNECTORS
from .run import crawl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="crawler")
_state: dict = {"running": False, "last_summary": None}
_lock = threading.Lock()


class CrawlRequest(BaseModel):
    sources: Optional[list[str]] = None
    dry_run: bool = False


@app.get("/health")
def health() -> dict:
    return {"service": "crawler", "status": "ok", "running": _state["running"]}


@app.post("/crawl", status_code=202)
def start_crawl(req: CrawlRequest) -> dict:
    unknown = set(req.sources or []) - set(CONNECTORS)
    if unknown:
        raise HTTPException(400, f"unknown sources: {sorted(unknown)}")
    with _lock:
        if _state["running"]:
            raise HTTPException(409, "a crawl is already running")
        _state["running"] = True

    def job():
        try:
            _state["last_summary"] = crawl(req.sources, dry_run=req.dry_run)
        finally:
            _state["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return {"status": "started"}


@app.get("/crawl/last")
def last() -> dict:
    return {"running": _state["running"], "summary": _state["last_summary"]}
