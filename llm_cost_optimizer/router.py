"""Transparent, deterministic model-routing heuristic; tune it with evaluation data."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Settings
from .tokens import count_tokens


# These are routing hints, not a safety policy or a claim that keywords measure reasoning.
COMPLEX_PATTERNS = (
    r"\b(compare|contrast|trade[- ]?off|root cause|debug|refactor|architecture|design a system)\b",
    r"\b(step[- ]by[- ]step|analy[sz]e|evaluate|derive|prove|forecast|optimi[sz]e)\b",
    r"\b(write|review|explain|fix)\s+(?:a\s+)?(?:python|sql|javascript|typescript|pine|regex|code|query|algorithm)\b",
    r"\b(multiple scenarios|several constraints|edge cases|production ready)\b",
)
HIGHER_RISK_PATTERNS = (
    r"\bmedical diagnosis\b",
    r"\blegal advice\b",
    r"\binvestment advice\b",
    r"\bfinancial recommendation\b",
    r"\bsecurity incident\b",
)


@dataclass(frozen=True)
class RouteDecision:
    model: str
    reason: str
    complexity: str


def choose_model(query: str, context_text: str, settings: Settings) -> RouteDecision:
    normalized = " ".join(query.lower().split())
    if any(re.search(pattern, normalized) for pattern in HIGHER_RISK_PATTERNS):
        return RouteDecision(settings.strong_model, "Sensitive/high-risk topic keyword; review policy separately", "high")
    context_tokens = count_tokens(context_text, settings.cheap_model)
    if context_tokens > settings.strong_context_threshold_tokens:
        return RouteDecision(settings.strong_model, f"Context is long ({context_tokens} estimated tokens)", "high")
    if len(query) > settings.simple_query_max_chars:
        return RouteDecision(settings.strong_model, "Query exceeds simple-query length threshold", "high")
    if query.count("?") >= 2:
        return RouteDecision(settings.strong_model, "Multiple questions in one request", "high")
    if any(re.search(pattern, normalized) for pattern in COMPLEX_PATTERNS):
        return RouteDecision(settings.strong_model, "Complex-task keyword pattern matched", "high")
    return RouteDecision(settings.cheap_model, "Short single-task request with limited context", "low")
