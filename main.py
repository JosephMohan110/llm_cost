"""Run the reference support-answer feature from the command line."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_cost_optimizer import ContextDocument, LLMCostOptimizedFeature, Settings


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Cost-optimized LLM support-answer demo")
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument("--context-file", help="Optional text file containing the trusted reference material")
    parser.add_argument("--cache-scope", default="demo", help="Use a tenant/user-specific value for private data")
    parser.add_argument("--no-cache", action="store_true", help="Bypass application-level exact-response caching")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your environment or .env file before running the live demo")

    docs = []
    if args.context_file:
        path = Path(args.context_file)
        if not path.is_file():
            parser.error(f"Context file not found: {path}")
        docs.append(ContextDocument(text=path.read_text(encoding="utf-8"), source=path.name, score=1.0))

    settings = Settings.from_env()
    feature = LLMCostOptimizedFeature(settings)
    result = feature.answer(args.query, docs, cache_scope=args.cache_scope, allow_cache=not args.no_cache)

    print("\nANSWER\n------")
    print(result.answer)
    print("\nCOST / ROUTING\n--------------")
    print(f"Model: {result.model}")
    print(f"Route reason: {result.route_reason}")
    print(f"Cache hit: {result.cache_hit}")
    print(f"Input/output tokens: {result.input_tokens}/{result.output_tokens}")
    print(f"Cached input tokens: {result.cached_input_tokens}")
    print(f"Estimated token cost (USD): ${result.cost_usd:.8f}")
    print(f"Usage estimated: {result.usage_estimated}")
    print(f"Provider retries: {result.retry_count}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    print(f"Request ID: {result.request_id}")
    print(f"Today's UTC usage: {feature.usage_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
