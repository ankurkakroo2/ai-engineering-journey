"""
Token Counter - Accurate Token Counting for Embedding API Limits

LEARNING CONTEXT:
This module handles Transformation 2 in the chunking pipeline: normalizing
variable-length content to fit within the embedding model's token limits.

PURPOSE:
Functions vary wildly in size (10 lines to 500+ lines). The OpenAI embedding
model has hard limits (~8000 tokens for text-embedding-3-small). You must:
1. Count tokens accurately using tiktoken (matches OpenAI's tokenizer)
2. Validate content fits within limits
3. Enable intelligent truncation when needed

WHY TIKTOKEN:
OpenAI's tokenizer splits text into subword units. Examples:
- "location" → 1 token
- "authenticate_user" → 3 tokens (["auth", "ent", "icate_user"])
- "def calculate_total(items):" → 7 tokens

Simple character/word counting would be inaccurate and cause API errors.

TOKEN LIMITS:
- text-embedding-3-small: 8191 tokens max
- text-embedding-3-large: 8191 tokens max
- We use 8000 as safety margin for metadata overhead

TRUNCATION STRATEGY:
If content exceeds limit:
1. Preserve header (location context)
2. Preserve docstring (semantic meaning)
3. Truncate code from end (implementation details less critical)
4. Mark chunk as truncated for monitoring

DEPENDENCIES:
- tiktoken library (OpenAI's official tokenizer)
- Input: Formatted content string
- Output: Token count (int) and validation status
"""

import tiktoken
from typing import Optional


_encoding_cache: Optional[tiktoken.Encoding] = None


def get_encoding() -> tiktoken.Encoding:
    global _encoding_cache
    if _encoding_cache is None:
        _encoding_cache = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache


def count_tokens(text: str) -> int:
    encoding = get_encoding()
    return len(encoding.encode(text))


def is_within_limit(text: str, max_tokens: int = 8000) -> bool:
    return count_tokens(text) <= max_tokens


def get_token_stats(text: str) -> dict:
    token_count = count_tokens(text)
    return {
        'token_count': token_count,
        'character_count': len(text),
        'line_count': text.count('\n') + 1,
        'within_limit': token_count <= 8000,
        'utilization_pct': round((token_count / 8000) * 100, 2)
    }
