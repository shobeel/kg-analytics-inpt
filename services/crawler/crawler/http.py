"""Polite HTTP: one User-Agent, per-host rate limits, robots.txt for web URLs, retries."""

import logging
import os
import threading
import time
from typing import Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)


def user_agent() -> str:
    contact = os.environ.get("CRAWLER_CONTACT", "").strip()
    ua = "KG-Analytics-Research/0.1 (INPT Big Data student project; non-commercial"
    return ua + (f"; {contact})" if contact else ")")


class RobotsDisallowed(Exception):
    pass


class PoliteSession:
    def __init__(self, cfg: dict):
        defaults = cfg.get("defaults", {})
        self.default_interval = float(defaults.get("min_interval_s", 2.0))
        self.timeout = float(defaults.get("timeout_s", 60))
        self.max_retries = int(defaults.get("max_retries", 3))
        self.host_interval = {h: float(v) for h, v in (cfg.get("hosts") or {}).items()}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent()
        self._last: dict[str, float] = {}
        self._robots: dict[str, Optional[RobotFileParser]] = {}
        self._lock = threading.Lock()

    # ---- robots.txt -------------------------------------------------------
    def _robots_for(self, scheme: str, host: str) -> Optional[RobotFileParser]:
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                resp = self._request("GET", f"{scheme}://{host}/robots.txt", check_robots=False)
                if resp.status_code in (401, 403):
                    rp.disallow_all = True
                elif resp.status_code >= 400:
                    rp.allow_all = True          # no robots.txt => everything allowed
                else:
                    rp.parse(resp.text.splitlines())
            except requests.RequestException as e:
                log.warning("robots.txt for %s unreachable (%s): treating host as disallowed", host, e)
                rp.disallow_all = True
            self._robots[host] = rp
        return self._robots[host]

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        rp = self._robots_for(parts.scheme, parts.netloc)
        return rp.can_fetch(self.session.headers["User-Agent"], url)

    def _interval(self, host: str) -> float:
        interval = self.host_interval.get(host, self.default_interval)
        rp = self._robots.get(host)
        delay = rp.crawl_delay(self.session.headers["User-Agent"]) if rp else None
        return max(interval, float(delay or 0))

    # ---- requests ---------------------------------------------------------
    def _wait_turn(self, host: str) -> None:
        with self._lock:
            wait = self._last.get(host, 0) + self._interval(host) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _request(self, method: str, url: str, check_robots: bool, **kwargs) -> requests.Response:
        host = urlsplit(url).netloc
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(url)
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn(host)
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                log.warning("%s %s failed (%s), retry %d", method, url, e, attempt)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After", "")
                sleep = float(retry_after) if retry_after.isdigit() else 5.0 * attempt
                log.warning("%s %s -> %d, sleeping %.0fs", method, url, resp.status_code, sleep)
                time.sleep(sleep)
                continue
            return resp
        raise RuntimeError("unreachable")

    def get_api(self, url: str, **kwargs) -> requests.Response:
        """Official API call: governed by the provider's API terms, not robots.txt."""
        resp = self._request("GET", url, check_robots=False, **kwargs)
        resp.raise_for_status()
        return resp

    def get_web(self, url: str, **kwargs) -> requests.Response:
        """Web page / feed / file: robots.txt is enforced (raises RobotsDisallowed)."""
        resp = self._request("GET", url, check_robots=True, **kwargs)
        resp.raise_for_status()
        return resp
