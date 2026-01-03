"""
Embedder Module - Component 3 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is the third component in your 8-week AI Engineering Journey (Week 1).
After parsing functions and chunking them into embeddable units, the embedder
transforms text into high-dimensional vectors for semantic search.

WHY EMBEDDINGS MATTER:
Embeddings map text to vectors where semantic similarity = geometric proximity.
The OpenAI embedding model (text-embedding-3-small) was trained on massive
text corpora to learn these semantic relationships. When you embed code,
similar functions end up close together in 1536-dimensional space.

CORE INSIGHT:
**Embeddings are deterministic** - same input always produces same output.
This enables aggressive caching to avoid redundant API calls.

From your Test 1 validation: Embedding the same chunk twice produces
identical vectors (not just similar - exactly the same). This determinism
is what makes content-based caching possible.

THE EMBEDDING PROCESS:

Input: Chunk content (formatted text string)
    ↓
Cache Check: SHA-256(content) → cache file exists?
    ↓ (miss)
OpenAI API: text-embedding-3-small
    ↓
Output: 1536-dimensional vector
    ↓
Cache Save: Store for future use

CACHE-FIRST STRATEGY:
1. Compute cache key: SHA-256(chunk.content)
2. Check if cache file exists
3. If yes: Load from cache (fast path: <1ms, free)
4. If no: Call API (slow path: 50-200ms, costs money)
5. Save result to cache for next time

WHY THIS WORKS:
- Deterministic: Same content → same embedding → same cache key
- No invalidation needed: Changed content → different cache key
- Efficient: Cache hit rate approaches 100% on re-indexing

BATCH PROCESSING:
The OpenAI API supports batching up to 2048 inputs per request.
Batching 100 chunks into 1 request is ~10x faster than 100 individual calls.

Trade-offs:
- Batching reduces API overhead (fewer HTTP requests)
- But requires more memory (hold batch in memory)
- Optimal batch size: 100-500 chunks

KEY LEARNING OBJECTIVES:
1. Understand deterministic embeddings enable caching
2. Learn content-based caching eliminates manual invalidation
3. Master batch processing to reduce API overhead
4. Handle rate limits with exponential backoff
5. Validate high-dimensional vectors (1536 dimensions)

USAGE:

    from parser import parse_file
    from chunker import chunk_functions
    from embedder import embed_chunk, embed_chunks

    # Single chunk
    functions = parse_file("models.py")
    chunks = chunk_functions(functions)
    embedding = embed_chunk(chunks[0])
    print(f"Dimension: {embedding.dimension}")
    print(f"Cached: {embedding.cached}")

    # Batch processing
    embeddings = embed_chunks(chunks, batch_size=100)
    print(f"Embedded {len(embeddings)} chunks")

NEXT STEPS:
After embedding, you'll pass embeddings to the storage module (Component 4),
which stores them in ChromaDB for similarity search.
"""

from .models import Embedding
from .embedder import embed_chunk, embed_chunks, get_embedding_stats
from .cache import get_cache_stats, clear_cache
from .api_client import create_openai_client

__all__ = [
    'Embedding',
    'embed_chunk',
    'embed_chunks',
    'get_embedding_stats',
    'get_cache_stats',
    'clear_cache',
    'create_openai_client'
]
