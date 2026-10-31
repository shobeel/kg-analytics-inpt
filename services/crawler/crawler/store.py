"""Bronze layer store: /datalake/bronze/<source>/<YYYY-MM-DD>/<id>.json (raw, never modified)."""

import logging
from collections import Counter

from .hdfs import WebHDFS
from .models import BronzeDocument

log = logging.getLogger(__name__)

BRONZE_ROOT = "/datalake/bronze"


class BronzeStore:
    def __init__(self, fs: WebHDFS, root: str = BRONZE_ROOT):
        self.fs = fs
        self.root = root.rstrip("/")
        self.known_ids: set[str] = set()
        self.saved = Counter()
        self.skipped = Counter()

    def load_index(self) -> int:
        """Collect ids of every document already in bronze (any source, any date)."""
        self.known_ids.clear()
        for src in self.fs.list_dir(self.root):
            if src["type"] != "DIRECTORY":
                continue
            for day in self.fs.list_dir(f"{self.root}/{src['pathSuffix']}"):
                if day["type"] != "DIRECTORY":
                    continue
                for f in self.fs.list_dir(f"{self.root}/{src['pathSuffix']}/{day['pathSuffix']}"):
                    name = f["pathSuffix"]
                    if name.endswith(".json"):
                        self.known_ids.add(name[:-5])
        return len(self.known_ids)

    def has(self, doc_id: str) -> bool:
        return doc_id in self.known_ids

    def save(self, doc: BronzeDocument) -> bool:
        """Save unless already present. Returns True if written."""
        if self.has(doc.id):
            self.skipped[doc.source] += 1
            return False
        day = doc.retrieved_at[:10]
        path = f"{self.root}/{doc.source}/{day}/{doc.id}.json"
        # One JSON object per file, on a single line (Spark reads it without multiLine).
        self.fs.write(path, doc.model_dump_json().encode("utf-8"))
        self.known_ids.add(doc.id)
        self.saved[doc.source] += 1
        return True
