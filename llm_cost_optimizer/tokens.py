"""Token counting with a character-based fallback for restricted environments."""
from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=32)
def _encoding_for_model(model: str):
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Current and custom model names may not yet be recognized by tiktoken.
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-6-luna") -> int:
    if not text:
        return 0
    encoding = _encoding_for_model(model)
    if encoding is None:
        # Planning estimate only; actual billing must use API-returned usage.
        return max(1, math.ceil(len(text) / 4))
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, token_limit: int, model: str = "gpt-6-luna") -> str:
    if token_limit <= 0 or not text:
        return ""
    encoding = _encoding_for_model(model)
    if encoding is None:
        return text[: token_limit * 4]
    tokens = encoding.encode(text)
    if len(tokens) <= token_limit:
        return text
    return encoding.decode(tokens[:token_limit])
