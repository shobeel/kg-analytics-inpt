"""Settings from the environment (.env via docker compose). Model names are never hardcoded."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).split(" #")[0].strip().strip("\"'")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"), repr=False)
    base_url: str = field(default_factory=lambda: _env(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    extraction_model: str = field(default_factory=lambda: _env("EXTRACTION_MODEL"))
    answer_model: str = field(default_factory=lambda: _env("ANSWER_MODEL"))
    # Gold pre-fill drafts come from a different provider (Groq), never from EXTRACTION_MODEL.
    prefill_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"), repr=False)
    prefill_base_url: str = field(default_factory=lambda: _env(
        "PREFILL_BASE_URL", "https://api.groq.com/openai/v1"))
    prefill_model: str = field(default_factory=lambda: _env("PREFILL_MODEL"))
    prefill_fallback_model: str = field(default_factory=lambda: _env("PREFILL_FALLBACK_MODEL"))
    prefill_min_interval_s: float = field(default_factory=lambda: float(_env("PREFILL_MIN_INTERVAL_S", "4")))
    # Free-tier throttle: minimum seconds between request starts (all threads) + parallel calls.
    min_interval_s: float = field(default_factory=lambda: float(_env("LLM_MIN_INTERVAL_S", "4.5")))
    concurrency: int = field(default_factory=lambda: int(_env("LLM_CONCURRENCY", "3")))
    max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "6")))
    timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "120")))
    # USD per 1M tokens; 0 on the free tier. Set list prices to estimate a paid run (RQ4).
    price_in_per_m: float = field(default_factory=lambda: float(_env("LLM_PRICE_IN_PER_M", "0")))
    price_out_per_m: float = field(default_factory=lambda: float(_env("LLM_PRICE_OUT_PER_M", "0")))
    call_log: str = field(default_factory=lambda: _env("LLM_CALL_LOG", "/app/logs/llm_calls.jsonl"))
    cache_dir: str = field(default_factory=lambda: _env("EXTRACTION_CACHE_DIR", "/data/cache/extraction"))
    hdfs: str = field(default_factory=lambda: _env("HDFS_URL", "hdfs://namenode:9000"))
    embedding_model: str = field(default_factory=lambda: _env(
        "FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def settings() -> Settings:
    s = Settings()
    if not s.api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    if not s.extraction_model:
        raise RuntimeError("EXTRACTION_MODEL is not set in .env")
    return s
