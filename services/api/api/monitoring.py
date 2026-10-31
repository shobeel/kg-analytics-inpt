"""Monitoring: one structured JSON log line per request (stdout and API_LOG, default
/app/logs/api.jsonl) and in-process counters exposed by GET /metrics in Prometheus text format:
requests and errors per endpoint, latency histogram, LLM calls and tokens of /ask."""

import json
import os
import sys
import threading
import time
from collections import defaultdict

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.requests = defaultdict(int)            # (method, route, status) -> n
        self.errors = defaultdict(int)              # route -> n (status >= 500)
        self.lat_sum = defaultdict(float)
        self.lat_count = defaultdict(int)
        self.lat_buckets = defaultdict(lambda: [0] * len(BUCKETS))
        self.llm = defaultdict(int)                 # llm_calls, cache_hits, tokens_in, tokens_out per strategy
        self.started = time.time()

    def observe(self, method: str, route: str, status: int, seconds: float):
        with self.lock:
            self.requests[(method, route, status)] += 1
            if status >= 500:
                self.errors[route] += 1
            self.lat_sum[route] += seconds
            self.lat_count[route] += 1
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    self.lat_buckets[route][i] += 1

    def observe_llm(self, strategy: str, out: dict):
        with self.lock:
            for k in ("llm_calls", "cache_hits", "tokens_in", "tokens_out"):
                self.llm[(strategy, k)] += int(out.get(k) or 0)

    def prometheus(self) -> str:
        with self.lock:
            L = ["# HELP api_requests_total HTTP requests by method, route and status.",
                 "# TYPE api_requests_total counter"]
            L += [f'api_requests_total{{method="{m}",route="{r}",status="{s}"}} {n}'
                  for (m, r, s), n in sorted(self.requests.items())]
            L += ["# HELP api_errors_total Responses with status >= 500.", "# TYPE api_errors_total counter"]
            L += [f'api_errors_total{{route="{r}"}} {n}' for r, n in sorted(self.errors.items())]
            L += ["# HELP api_request_seconds Request latency.", "# TYPE api_request_seconds histogram"]
            for r in sorted(self.lat_count):
                for b, n in zip(BUCKETS, self.lat_buckets[r]):
                    L.append(f'api_request_seconds_bucket{{route="{r}",le="{b}"}} {n}')
                L.append(f'api_request_seconds_bucket{{route="{r}",le="+Inf"}} {self.lat_count[r]}')
                L.append(f'api_request_seconds_sum{{route="{r}"}} {self.lat_sum[r]:.6f}')
                L.append(f'api_request_seconds_count{{route="{r}"}} {self.lat_count[r]}')
            L += ["# HELP api_llm_total LLM calls, cache hits and tokens spent by /ask, per strategy.",
                  "# TYPE api_llm_total counter"]
            L += [f'api_llm_total{{strategy="{s}",kind="{k}"}} {n}' for (s, k), n in sorted(self.llm.items())]
            L += ["# HELP api_uptime_seconds Seconds since start.", "# TYPE api_uptime_seconds gauge",
                  f"api_uptime_seconds {time.time() - self.started:.0f}"]
            return "\n".join(L) + "\n"


METRICS = Metrics()
_log_lock = threading.Lock()


def log_event(**fields):
    """One JSON line to stdout and to API_LOG. Never pass secrets (keys are not logged)."""
    line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), **fields}, ensure_ascii=False,
                      default=str)
    print(line, file=sys.stdout, flush=True)
    path = os.environ.get("API_LOG", "/app/logs/api.jsonl")
    try:
        with _log_lock, open(path, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
