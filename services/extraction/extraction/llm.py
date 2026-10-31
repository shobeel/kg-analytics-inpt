"""Gemini via its OpenAI-compatible endpoint: throttled, retried, logged.

- Throttle: a global minimum spacing between request starts (free-tier RPM), shared by threads.
- 429 per-minute / 5xx / timeouts: retry with exponential backoff (honours Gemini's retryDelay).
- 429 per-day quota: raise QuotaExhausted so the run stops cleanly (results so far are cached).
- Every attempt is appended to the JSONL call log (RQ4): model, tokens, latency, cost, status.
"""

import json
import logging
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openai
from openai import OpenAI

from .config import Settings

log = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up: stop and resume tomorrow."""


class LLMCallFailed(RuntimeError):
    pass


def _is_daily_quota(text: str) -> bool:
    t = text.lower()
    return "perday" in t or "per day" in t or "requests_per_day" in t or "daily" in t


def _retry_delay(text: str) -> Optional[float]:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", text) or \
        re.search(r"(?:retry|try again) in (\d+(?:\.\d+)?)\s*s", text, re.I)     # Gemini / Groq
    return float(m.group(1)) if m else None


class LLMClient:
    def __init__(self, s: Settings, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 min_interval_s: Optional[float] = None):
        """Defaults to Gemini; base_url/api_key select another OpenAI-compatible provider."""
        self.s = s
        self.min_interval_s = s.min_interval_s if min_interval_s is None else min_interval_s
        self.client = OpenAI(api_key=api_key or s.api_key, base_url=base_url or s.base_url,
                             timeout=s.timeout_s, max_retries=0)
        self._next_start = 0.0
        self._throttle = threading.Lock()
        self._log_lock = threading.Lock()
        self.quota_exhausted = threading.Event()
        Path(s.call_log).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ internals
    def _wait_turn(self) -> None:
        with self._throttle:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.min_interval_s
        if start > now:
            time.sleep(start - now)

    def _log(self, **row) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **row}
        with self._log_lock, open(self.s.call_log, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _cost(self, tin: int, tout: int) -> float:
        return round(tin / 1e6 * self.s.price_in_per_m + tout / 1e6 * self.s.price_out_per_m, 8)

    # ------------------------------------------------------------------ public
    def chat_json(self, *, system: str, user: str, schema: dict, schema_name: str,
                  purpose: str, ref: str = "", model: Optional[str] = None,
                  extra_body: Optional[dict] = None) -> tuple[str, dict]:
        """One structured-output call. Returns (raw JSON text, usage dict). Raises
        QuotaExhausted on a daily-quota 429, LLMCallFailed after max_retries. extra_body: provider-
        specific request fields (e.g. Groq reasoning_effort)."""
        model = model or self.s.extraction_model
        for attempt in range(1, self.s.max_retries + 1):
            if self.quota_exhausted.is_set():
                raise QuotaExhausted("daily quota already reached in this run")
            self._wait_turn()
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": schema_name, "schema": schema}},
                    **({"extra_body": extra_body} if extra_body else {}))
            except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError,
                    openai.APITimeoutError) as e:
                latency = round(time.monotonic() - t0, 3)
                status = getattr(e, "status_code", None)
                text = str(getattr(e, "body", "")) + " " + str(e)
                daily = status == 429 and _is_daily_quota(text)
                self._log(purpose=purpose, ref=ref, model=model, status="quota_exhausted" if daily else "error",
                          http_status=status, attempt=attempt, latency_s=latency, error=text[:2000])
                if daily:
                    self.quota_exhausted.set()
                    raise QuotaExhausted(str(e)[:500]) from e
                retryable = status in (None, 429, 500, 502, 503, 504) or \
                    (status == 400 and "json_validate_failed" in text)   # Groq: sampled invalid JSON
                if not retryable or attempt == self.s.max_retries:
                    raise LLMCallFailed(f"{model} {status}: {str(e)[:300]}") from e
                delay = _retry_delay(text) or min(120.0, 5.0 * 2 ** (attempt - 1))
                delay += random.uniform(0, 1.5)
                log.warning("%s %s -> %s, retry %d in %.0fs", purpose, ref, status, attempt, delay)
                time.sleep(delay)
                continue

            latency = round(time.monotonic() - t0, 3)
            u = resp.usage
            tin, tout = (u.prompt_tokens or 0), (u.completion_tokens or 0)
            total = u.total_tokens or (tin + tout)
            thinking = max(0, total - tin - tout)            # hidden reasoning tokens, if any
            usage = {"input_tokens": tin, "output_tokens": tout, "thinking_tokens": thinking,
                     "total_tokens": total, "latency_s": latency,
                     "cost_usd": self._cost(tin, tout + thinking)}
            self._log(purpose=purpose, ref=ref, model=model, status="ok", http_status=200,
                      attempt=attempt, **usage)
            content = resp.choices[0].message.content or ""
            return content, usage
        raise LLMCallFailed("unreachable")
