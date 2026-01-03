"""
Embedder - Main Orchestrator for Embedding Generation

LEARNING CONTEXT:
This is the main entry point for Component 3 of your RAG pipeline.
It orchestrates the cache-first embedding workflow:
1. Check cache for existing embedding
2. If miss, call OpenAI API
3. Save result to cache
4. Return embedding

PURPOSE:
Coordinates the embedding pipeline with intelligent caching to minimize
API calls and costs. Implements both single and batch processing.

KEY CONCEPTS:
- Cache-first strategy: Always check cache before API call
- Batch processing: Group multiple chunks for efficiency
- Statistics tracking: Monitor cache hit rate and API usage
- Graceful degradation: Individual failures don't crash batch

INTEGRATION POINTS:
- Input: Chunk objects from chunker module
- Output: Embedding objects for storage module
- Uses: OpenAI API (via api_client) and cache layer

DEPENDENCIES:
- cache.py: Cache management
- api_client.py: OpenAI API integration
- models.py: Embedding dataclass
"""

import logging
import time
from typing import List, Optional, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..chunker.models import Chunk
from .models import Embedding
from .cache import load_from_cache, save_to_cache, get_cache_stats
from .api_client import create_openai_client, call_embedding_api, call_embedding_api_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_chunk(chunk: Chunk, client=None) -> Optional[Embedding]:
    try:
        cached_vector = load_from_cache(chunk.content)

        if cached_vector is not None:
            logger.debug(f"Cache hit for chunk {chunk.id[:16]}...")
            return Embedding(
                chunk_id=chunk.id,
                vector=cached_vector,
                cached=True,
                metadata=chunk.metadata
            )

        if client is None:
            client = create_openai_client()

        logger.debug(f"Cache miss for chunk {chunk.id[:16]}..., calling API")
        vector = call_embedding_api(client, chunk.content)

        if vector is None:
            logger.error(f"Failed to generate embedding for chunk {chunk.id[:16]}...")
            return None

        save_to_cache(chunk.content, vector)

        return Embedding(
            chunk_id=chunk.id,
            vector=vector,
            cached=False,
            metadata=chunk.metadata
        )

    except Exception as e:
        logger.error(f"Error embedding chunk {chunk.id[:16]}...: {e}")
        return None


def embed_chunks(
    chunks: List[Chunk],
    batch_size: int = 100,
    client=None
) -> List[Embedding]:
    embeddings = []

    if client is None:
        client = create_openai_client()

    cache_hits = 0
    api_calls = 0
    start_time = time.time()

    uncached_chunks = []
    uncached_indices = []

    for i, chunk in enumerate(chunks):
        cached_vector = load_from_cache(chunk.content)

        if cached_vector is not None:
            embeddings.append(Embedding(
                chunk_id=chunk.id,
                vector=cached_vector,
                cached=True,
                metadata=chunk.metadata
            ))
            cache_hits += 1
        else:
            embeddings.append(None)
            uncached_chunks.append(chunk)
            uncached_indices.append(i)

    if uncached_chunks:
        logger.info(
            f"Processing {len(uncached_chunks)} uncached chunks "
            f"({cache_hits} cache hits)"
        )

        for batch_start in range(0, len(uncached_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_chunks))
            batch = uncached_chunks[batch_start:batch_end]

            batch_contents = [chunk.content for chunk in batch]

            logger.info(
                f"Calling API for batch {batch_start//batch_size + 1} "
                f"({len(batch)} chunks)"
            )

            vectors = call_embedding_api_batch(client, batch_contents)

            if vectors is None:
                logger.error(f"Batch API call failed for batch starting at {batch_start}")
                continue

            api_calls += 1

            for j, (chunk, vector) in enumerate(zip(batch, vectors)):
                save_to_cache(chunk.content, vector)

                original_idx = uncached_indices[batch_start + j]
                embeddings[original_idx] = Embedding(
                    chunk_id=chunk.id,
                    vector=vector,
                    cached=False,
                    metadata=chunk.metadata
                )

    embeddings = [e for e in embeddings if e is not None]

    elapsed_time = time.time() - start_time

    logger.info(
        f"Embedded {len(embeddings)}/{len(chunks)} chunks in {elapsed_time:.2f}s "
        f"(cache hits: {cache_hits}, API calls: {api_calls})"
    )

    return embeddings


def get_embedding_stats(embeddings: List[Embedding]) -> Dict[str, Any]:
    if not embeddings:
        return {
            'total_embeddings': 0,
            'cached_count': 0,
            'api_count': 0,
            'cache_hit_rate': 0.0,
            'avg_dimension': 0,
            'all_valid': True
        }

    cached_count = sum(1 for e in embeddings if e.cached)
    api_count = len(embeddings) - cached_count

    return {
        'total_embeddings': len(embeddings),
        'cached_count': cached_count,
        'api_count': api_count,
        'cache_hit_rate': round(cached_count / len(embeddings) * 100, 2),
        'avg_dimension': sum(e.dimension for e in embeddings) / len(embeddings),
        'all_valid': all(e.is_valid for e in embeddings)
    }
