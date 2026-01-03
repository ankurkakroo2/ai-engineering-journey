"""
Query Embedder - Embed User Queries for Semantic Search

LEARNING CONTEXT:
This is Component 5 of your RAG-Code-QA pipeline (Week 1, Day 7-8).
To search for semantically similar code chunks, you must first embed the
user's query using the SAME model that was used to embed the indexed chunks.

PURPOSE:
Embeds natural language queries into 1536-dimensional vectors using
OpenAI's text-embedding-3-small model. This ensures query vectors live
in the same semantic space as indexed chunk vectors.

KEY CONCEPTS:
- MUST use text-embedding-3-small (same as indexing)
- Different models create different semantic spaces (incompatible)
- No caching needed (queries are unique, one-time use)
- Reuses embedder.api_client for consistency

WHY SAME MODEL MATTERS:
Embeddings create a semantic space where similar concepts are close together.
Using different models for indexing vs querying is like comparing GPS
coordinates from Earth to Mars coordinates - they're in different spaces!

Example:
  text-embedding-3-small: "authentication" → [0.12, 0.45, ...]
  text-embedding-ada-002:  "authentication" → [0.89, -0.23, ...]
                                               ↑ Different space!

INTEGRATION POINTS:
- Input: Natural language query string
- Uses: embedder.api_client (create_openai_client, call_embedding_api)
- Output: 1536-dimensional vector for ChromaDB search
"""

import logging
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..embedder.api_client import create_openai_client, call_embedding_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_query(query: str, client=None) -> List[float]:
    """
    Embed a query string using the same model as indexing.

    This is the primary function for query embedding. It MUST use
    text-embedding-3-small to ensure query vectors live in the same
    semantic space as indexed chunk vectors.

    Args:
        query: Natural language query string
        client: Optional OpenAI client (created if not provided)

    Returns:
        1536-dimensional embedding vector

    Raises:
        ValueError: If query is empty or embedding fails

    Example:
        >>> vector = embed_query("How does authentication work?")
        >>> len(vector)
        1536
        >>> type(vector[0])
        <class 'float'>
    """
    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    logger.info(f"Embedding query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

    # Create client if not provided
    if client is None:
        client = create_openai_client()

    # Call OpenAI API (reuses embedder module's implementation)
    vector = call_embedding_api(client, query)

    if vector is None:
        raise ValueError(
            "Failed to embed query. Check API key and network connection."
        )

    # Validate output
    if len(vector) != 1536:
        raise ValueError(
            f"Expected 1536-dimensional vector, got {len(vector)}. "
            "This indicates a model mismatch."
        )

    logger.info(f"Successfully embedded query (dimension: {len(vector)})")

    return vector


def embed_queries_batch(queries: List[str], client=None) -> List[List[float]]:
    """
    Embed multiple queries in batch.

    This is useful for batch processing or evaluation scenarios where
    you need to embed multiple queries at once. Uses the same API client
    for all queries.

    Args:
        queries: List of query strings
        client: Optional OpenAI client (created if not provided)

    Returns:
        List of 1536-dimensional embedding vectors

    Raises:
        ValueError: If any query is empty or embedding fails

    Example:
        >>> queries = [
        ...     "How does authentication work?",
        ...     "Where is error handling implemented?"
        ... ]
        >>> vectors = embed_queries_batch(queries)
        >>> len(vectors)
        2
        >>> all(len(v) == 1536 for v in vectors)
        True
    """
    if not queries:
        raise ValueError("Queries list cannot be empty")

    logger.info(f"Embedding {len(queries)} queries in batch")

    # Create client if not provided
    if client is None:
        client = create_openai_client()

    # Embed each query
    vectors = []
    for i, query in enumerate(queries, 1):
        try:
            vector = embed_query(query, client)
            vectors.append(vector)
            logger.debug(f"Embedded query {i}/{len(queries)}")
        except Exception as e:
            logger.error(f"Failed to embed query {i}: {e}")
            raise

    logger.info(f"Successfully embedded {len(vectors)} queries")

    return vectors


def validate_query_embedding(vector: List[float]) -> bool:
    """
    Validate that a query embedding is correct.

    Checks:
    - Vector has 1536 dimensions
    - All values are floats
    - No NaN or infinite values

    Args:
        vector: Embedding vector to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> vector = embed_query("test query")
        >>> validate_query_embedding(vector)
        True
    """
    # Check dimension
    if len(vector) != 1536:
        logger.error(f"Invalid dimension: {len(vector)} (expected 1536)")
        return False

    # Check types and values
    for i, val in enumerate(vector):
        if not isinstance(val, (int, float)):
            logger.error(f"Invalid type at index {i}: {type(val)}")
            return False

        # Check for NaN or infinite
        if val != val:  # NaN check
            logger.error(f"NaN value at index {i}")
            return False

        if abs(val) == float('inf'):
            logger.error(f"Infinite value at index {i}")
            return False

    return True
