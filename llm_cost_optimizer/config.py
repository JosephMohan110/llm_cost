"""Environment-backed configuration for the LLM cost optimizer."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .pricing import PriceRates


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _budget_env() -> float | None:
    raw = os.getenv("LLM_DAILY_BUDGET_USD", "5.00").strip().lower()
    if raw in {"", "none", "off", "disabled"}:
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("LLM_DAILY_BUDGET_USD must be finite and >= 0 or 'none'")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    cheap_model: str = "gpt-6-luna"
    strong_model: str = "gpt-6.1-sol"
    cheap_rates: PriceRates = field(default_factory=lambda: PriceRates(0.05, 0.005, 0.0625, 0.25))
    strong_rates: PriceRates = field(default_factory=lambda: PriceRates(1.00, 0.05, 1.25, 5.00))
    max_output_tokens: int = 350
    max_context_tokens: int = 1800
    provider_timeout_seconds: float = 30.0
    max_provider_retries: int = 1
    retry_backoff_seconds: float = 0.25
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True
    daily_budget_usd: float | None = 5.00
    db_path: str = ".llm_cost_optimizer.sqlite3"
    prompt_version: str = "support-answer-v1"
    route_policy_version: str = "support-route-v1"
    routing_enabled: bool = True
    max_query_chars: int = 8000
    simple_query_max_chars: int = 450
    strong_context_threshold_tokens: int = 1200

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings from environment variables; no dotenv dependency required."""
        cheap_rates = PriceRates(
            input_usd_per_1m=_float_env("LLM_CHEAP_INPUT_USD_PER_1M", 0.05),
            cached_input_usd_per_1m=_float_env("LLM_CHEAP_CACHED_INPUT_USD_PER_1M", 0.005),
            cache_write_usd_per_1m=_float_env("LLM_CHEAP_CACHE_WRITE_USD_PER_1M", 0.0625),
            output_usd_per_1m=_float_env("LLM_CHEAP_OUTPUT_USD_PER_1M", 0.25),
        )
        strong_rates = PriceRates(
            input_usd_per_1m=_float_env("LLM_STRONG_INPUT_USD_PER_1M", 1.00),
            cached_input_usd_per_1m=_float_env("LLM_STRONG_CACHED_INPUT_USD_PER_1M", 0.05),
            cache_write_usd_per_1m=_float_env("LLM_STRONG_CACHE_WRITE_USD_PER_1M", 1.25),
            output_usd_per_1m=_float_env("LLM_STRONG_OUTPUT_USD_PER_1M", 5.00),
        )
        return cls(
            api_key=os.getenv("OPENAI_API_KEY") or None,
            cheap_model=os.getenv("LLM_CHEAP_MODEL", "gpt-6-luna").strip(),
            strong_model=os.getenv("LLM_STRONG_MODEL", "gpt-6.1-sol").strip(),
            cheap_rates=cheap_rates,
            strong_rates=strong_rates,
            max_output_tokens=_int_env("LLM_MAX_OUTPUT_TOKENS", 350),
            max_context_tokens=_int_env("LLM_MAX_CONTEXT_TOKENS", 1800),
            provider_timeout_seconds=_float_env("LLM_PROVIDER_TIMEOUT_SECONDS", 30.0),
            max_provider_retries=_nonnegative_int_env("LLM_MAX_PROVIDER_RETRIES", 1),
            retry_backoff_seconds=_float_env("LLM_RETRY_BACKOFF_SECONDS", 0.25),
            cache_ttl_seconds=_int_env("LLM_CACHE_TTL_SECONDS", 3600),
            cache_enabled=os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            daily_budget_usd=_budget_env(),
            db_path=os.getenv("LLM_DB_PATH", ".llm_cost_optimizer.sqlite3"),
            prompt_version=os.getenv("LLM_PROMPT_VERSION", "support-answer-v1").strip(),
            route_policy_version=os.getenv("LLM_ROUTE_POLICY_VERSION", "support-route-v1").strip(),
            routing_enabled=os.getenv("LLM_ROUTING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            max_query_chars=_int_env("LLM_MAX_QUERY_CHARS", 8000),
            simple_query_max_chars=_int_env("LLM_SIMPLE_QUERY_MAX_CHARS", 450),
            strong_context_threshold_tokens=_int_env("LLM_STRONG_CONTEXT_THRESHOLD_TOKENS", 1200),
        )

    def rates_for(self, model: str) -> PriceRates:
        """Use cheap-model rates only for the configured cheap model; otherwise strong rates."""
        return self.cheap_rates if model == self.cheap_model else self.strong_rates

    def validate(self) -> None:
        if not self.cheap_model or not self.strong_model:
            raise ValueError("Both LLM_CHEAP_MODEL and LLM_STRONG_MODEL must be non-empty")
        if not self.prompt_version or not self.route_policy_version:
            raise ValueError("Prompt and route-policy versions must be non-empty")
        if self.cheap_model == self.strong_model:
            raise ValueError("Cheap and strong model names must differ for model routing to work")
        if self.max_output_tokens < 1 or self.max_context_tokens < 1:
            raise ValueError("Token limits must be positive")
        if not math.isfinite(self.provider_timeout_seconds) or self.provider_timeout_seconds <= 0:
            raise ValueError("Provider timeout must be finite and positive")
        if self.max_provider_retries < 0 or self.max_provider_retries > 5:
            raise ValueError("Provider retry count must be between 0 and 5")
        if not math.isfinite(self.retry_backoff_seconds) or self.retry_backoff_seconds < 0:
            raise ValueError("Retry backoff must be finite and non-negative")
        if self.max_query_chars < 1 or self.simple_query_max_chars < 1:
            raise ValueError("Query character limits must be positive")
        if self.cache_ttl_seconds < 1:
            raise ValueError("Cache TTL must be positive")
        if self.daily_budget_usd is not None and (
            not math.isfinite(self.daily_budget_usd) or self.daily_budget_usd < 0
        ):
            raise ValueError("Daily budget must be finite, non-negative, or None")
        if self.db_path != ":memory:":
            parent = Path(self.db_path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
