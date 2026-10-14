"""Cost calculations. Rates are USD per one million tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PriceRates:
    input_usd_per_1m: float
    cached_input_usd_per_1m: float
    cache_write_usd_per_1m: float
    output_usd_per_1m: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")


def calculate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    rates: PriceRates,
) -> float:
    """Calculate an estimated token charge, splitting regular/cached/write input.

    Providers may expose cached-read/write usage differently. If cache-write usage
    is not returned, pass zero; compare this estimate with the provider invoice.
    """
    values = (input_tokens, output_tokens, cached_input_tokens, cache_write_input_tokens)
    if any(value < 0 for value in values):
        raise ValueError("Token counts must be >= 0")

    # Input subcategories must never count more tokens than total input.
    cached = min(input_tokens, cached_input_tokens)
    remaining = input_tokens - cached
    cache_write = min(remaining, cache_write_input_tokens)
    regular = remaining - cache_write

    cost = (
        regular * rates.input_usd_per_1m
        + cached * rates.cached_input_usd_per_1m
        + cache_write * rates.cache_write_usd_per_1m
        + output_tokens * rates.output_usd_per_1m
    ) / 1_000_000
    return round(cost, 12)


def estimate_max_cost_usd(*, input_tokens: int, max_output_tokens: int, rates: PriceRates) -> float:
    """Conservative pre-request estimate: assume no cached-input discount."""
    return calculate_cost_usd(
        input_tokens=input_tokens,
        output_tokens=max_output_tokens,
        rates=rates,
    )
