"""Minimal WebHDFS client (namenode HTTP port 9870). Runs inside the kgnet network,
where the datanode hostnames the namenode redirects to are resolvable."""

import os
from typing import Optional

import requests


class WebHDFS:
    def __init__(self, base_url: Optional[str] = None, user: str = "crawler"):
        self.base = (base_url or os.environ.get("HDFS_NAMENODE_HTTP", "http://namenode:9870")).rstrip("/")
        self.user = user
        self.http = requests.Session()

    def _url(self, path: str, op: str, **params) -> tuple[str, dict]:
        params = {"op": op, "user.name": self.user, **params}
        return f"{self.base}/webhdfs/v1/{path.lstrip('/')}", params

    def list_dir(self, path: str) -> list[dict]:
        url, params = self._url(path, "LISTSTATUS")
        r = self.http.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()["FileStatuses"]["FileStatus"]

    def mkdirs(self, path: str) -> None:
        url, params = self._url(path, "MKDIRS")
        self.http.put(url, params=params, timeout=60).raise_for_status()

    def exists(self, path: str) -> bool:
        url, params = self._url(path, "GETFILESTATUS")
        r = self.http.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    def write(self, path: str, data: bytes) -> None:
        """Write atomically: create <path>.tmp, then rename, so a crash never leaves a
        partial file under the final name (re-runs only check final names)."""
        tmp = path + ".tmp"
        url, params = self._url(tmp, "CREATE", overwrite="true")
        r = self.http.put(url, params=params, allow_redirects=False, timeout=60)
        r.raise_for_status()
        location = r.headers["Location"]          # 307 -> datanode
        self.http.put(location, data=data, timeout=120,
                      headers={"Content-Type": "application/octet-stream"}).raise_for_status()
        url, params = self._url(tmp, "RENAME", destination="/" + path.lstrip("/"))
        r = self.http.put(url, params=params, timeout=60)
        r.raise_for_status()
        if not r.json().get("boolean"):
            raise IOError(f"rename {tmp} -> {path} failed (target exists?)")
