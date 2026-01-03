"""
Cache Management - Content-Based Caching for Embeddings

LEARNING CONTEXT:
Embeddings are deterministic - same input always produces same output.
This enables aggressive caching to avoid redundant API calls and costs.

PURPOSE:
Implements content-based caching using SHA-256 hashes as keys.
When a chunk is embedded, the result is cached. On re-indexing, unchanged
chunks hit the cache (instant, free) instead of calling the API (slow, costly).

KEY CONCEPTS:
- Cache key = SHA-256(chunk.content) - deterministic and collision-resistant
- No manual cache invalidation needed (content change = different key)
- Cache storage = JSON files in ./data/cache/ directory
- Cache hit rate is a key performance metric

WHY THIS WORKS:
From your Test 1 validation: Embeddings are frozen, pre-trained models.
Same input → same output, always. This determinism enables caching.

DEPENDENCIES:
- Input: Chunk content strings
- Output: Cached embedding vectors or None (cache miss)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/cache")


def get_cache_key(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_from_cache(content: str) -> Optional[List[float]]:
    try:
        cache_key = get_cache_key(content)
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['vector']

        return None
    except Exception as e:
        logger.warning(f"Error loading from cache: {e}")
        return None


def save_to_cache(content: str, vector: List[float]) -> bool:
    try:
        ensure_cache_dir()
        cache_key = get_cache_key(content)
        cache_file = CACHE_DIR / f"{cache_key}.json"

        data = {
            'vector': vector,
            'dimension': len(vector)
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f)

        return True
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
        return False


def get_cache_stats() -> Dict[str, Any]:
    try:
        ensure_cache_dir()
        cache_files = list(CACHE_DIR.glob("*.json"))

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'total_entries': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(CACHE_DIR.absolute())
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            'total_entries': 0,
            'total_size_bytes': 0,
            'total_size_mb': 0.0,
            'cache_dir': str(CACHE_DIR.absolute())
        }


def clear_cache() -> int:
    try:
        ensure_cache_dir()
        cache_files = list(CACHE_DIR.glob("*.json"))
        count = 0

        for cache_file in cache_files:
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache entries")
        return count
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return 0
