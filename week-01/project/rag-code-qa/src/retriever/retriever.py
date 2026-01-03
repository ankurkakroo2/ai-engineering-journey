"""
Retriever - Main Orchestrator for Semantic Code Search

LEARNING CONTEXT:
This is the main entry point for Component 5 of your RAG-Code-QA pipeline.
It orchestrates the complete retrieval workflow:
1. Embed the query using the same model as indexing
2. Search ChromaDB for similar vectors
3. Filter results by score threshold
4. Rank and limit to top-k
5. Return RetrievalResult objects

PURPOSE:
Provides a simple, high-level API for semantic code search. This is the
facade that ties together all retriever submodules and provides the
interface used by the generator module.

KEY CONCEPTS:
- Semantic search: Find code by meaning, not keywords
- Same model requirement: Query and index embeddings must use same model
- Score thresholds: Filter noise (default: 0.3)
- Top-k results: Return most relevant chunks (default: 5)

ARCHITECTURE:
This module follows the facade pattern, coordinating:
- query_embedder: Embed the query
- searcher: Search ChromaDB
- result_processor: Filter, rank, convert

INTEGRATION POINTS:
- Input: Natural language query string
- Output: List of RetrievalResult objects (for generator module)
- Uses: embedder, storage modules
"""

import logging
from typing import List, Optional, Dict, Any
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .query_embedder import embed_query
from .searcher import search_similar
from .result_processor import process_results
from .models import RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global statistics for monitoring
_retrieval_stats = {
    'total_queries': 0,
    'total_results': 0,
    'avg_query_time': 0.0,
    'cache_hits': 0
}


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.3,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[RetrievalResult]:
    """
    Retrieve semantically similar code chunks for a natural language query.

    This is the main entry point for semantic code search. It handles the
    complete retrieval pipeline from query to ranked results.

    The retrieval process:
    1. Embed query using text-embedding-3-small (same as indexing)
    2. Search ChromaDB for similar vectors using cosine similarity
    3. Filter results below min_score threshold
    4. Rank by similarity score (highest first)
    5. Limit to top-k results
    6. Convert to RetrievalResult objects

    Args:
        query: Natural language query (e.g., "How does authentication work?")
        top_k: Number of results to return (default: 5)
        min_score: Minimum similarity score 0-1 (default: 0.3)
        filter_metadata: Optional metadata filter (e.g., {"language": "python"})

    Returns:
        List of RetrievalResult objects, ranked by similarity (highest first)
        Empty list if no results pass the threshold

    Raises:
        ValueError: If query is empty or embedding fails

    Example:
        >>> results = retrieve("How does authentication work?", top_k=3)
        >>> for result in results:
        ...     print(f"{result.function_name}: {result.score:.3f}")
        ...     print(f"  {result.location}")
        login_handler: 0.872
          src/auth/handlers.py:45-67
        verify_token: 0.819
          src/auth/tokens.py:23-41
        authenticate_user: 0.785
          src/auth/auth.py:12-34

    Example with filtering:
        >>> # Search only Python functions
        >>> results = retrieve(
        ...     "error handling",
        ...     top_k=5,
        ...     min_score=0.5,
        ...     filter_metadata={"language": "python"}
        ... )
    """
    global _retrieval_stats

    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    logger.info(f"Retrieving results for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    logger.info(f"Parameters: top_k={top_k}, min_score={min_score}")

    start_time = time.time()

    try:
        # Step 1: Embed the query
        logger.debug("Step 1: Embedding query...")
        query_vector = embed_query(query)
        embed_time = time.time() - start_time
        logger.debug(f"Query embedded in {embed_time:.3f}s")

        # Step 2: Search ChromaDB
        # Request more results than needed to account for filtering
        # (top_k * 2 ensures we have enough after score filtering)
        logger.debug("Step 2: Searching ChromaDB...")
        search_start = time.time()
        raw_results = search_similar(
            query_vector=query_vector,
            top_k=top_k * 2,  # Over-fetch for filtering
            filter_metadata=filter_metadata
        )
        search_time = time.time() - search_start
        logger.debug(f"Search completed in {search_time:.3f}s ({len(raw_results)} results)")

        # Step 3: Process results (filter, rank, convert, limit)
        logger.debug("Step 3: Processing results...")
        process_start = time.time()
        results = process_results(
            storage_entries=raw_results,
            min_score=min_score,
            top_k=top_k
        )
        process_time = time.time() - process_start
        logger.debug(f"Processing completed in {process_time:.3f}s ({len(results)} final)")

        # Total time
        total_time = time.time() - start_time

        # Update statistics
        _retrieval_stats['total_queries'] += 1
        _retrieval_stats['total_results'] += len(results)
        _retrieval_stats['avg_query_time'] = (
            (_retrieval_stats['avg_query_time'] * (_retrieval_stats['total_queries'] - 1) + total_time)
            / _retrieval_stats['total_queries']
        )

        # Log summary
        logger.info(
            f"Retrieved {len(results)} results in {total_time:.3f}s "
            f"(embed: {embed_time:.3f}s, search: {search_time:.3f}s, process: {process_time:.3f}s)"
        )

        if results:
            logger.info(
                f"Score range: {results[-1].score:.3f} - {results[0].score:.3f}"
            )
        else:
            logger.warning(
                f"No results found above threshold {min_score:.2f}. "
                "Try lowering min_score or rephrasing the query."
            )

        return results

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise


def retrieve_by_language(
    query: str,
    language: str,
    top_k: int = 5,
    min_score: float = 0.3
) -> List[RetrievalResult]:
    """
    Retrieve results filtered by programming language.

    Convenience function for language-specific searches.

    Args:
        query: Natural language query
        language: Programming language ("python", "javascript", "typescript")
        top_k: Number of results to return (default: 5)
        min_score: Minimum similarity score (default: 0.3)

    Returns:
        List of RetrievalResult objects from the specified language

    Example:
        >>> results = retrieve_by_language(
        ...     "authentication",
        ...     language="python",
        ...     top_k=3
        ... )
        >>> all(r.language == "python" for r in results)
        True
    """
    logger.info(f"Retrieving {language} functions for query: '{query}'")

    return retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        filter_metadata={"language": language}
    )


def retrieve_from_file(
    query: str,
    file_path: str,
    top_k: int = 5,
    min_score: float = 0.3
) -> List[RetrievalResult]:
    """
    Retrieve results filtered by file path.

    Useful for searching within a specific file or directory.

    Args:
        query: Natural language query
        file_path: File path filter (exact match)
        top_k: Number of results to return (default: 5)
        min_score: Minimum similarity score (default: 0.3)

    Returns:
        List of RetrievalResult objects from the specified file

    Example:
        >>> results = retrieve_from_file(
        ...     "visitor pattern",
        ...     file_path="src/parser/python_parser.py",
        ...     top_k=3
        ... )
    """
    logger.info(f"Retrieving from file '{file_path}' for query: '{query}'")

    return retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        filter_metadata={"file_path": file_path}
    )


def get_retrieval_stats() -> Dict[str, Any]:
    """
    Get retrieval statistics.

    Returns statistics about all retrieval operations since startup.
    Useful for monitoring and debugging.

    Returns:
        Dictionary with statistics:
        - total_queries: Total number of queries
        - total_results: Total results returned
        - avg_query_time: Average query time in seconds
        - avg_results_per_query: Average results per query

    Example:
        >>> stats = get_retrieval_stats()
        >>> print(f"Total queries: {stats['total_queries']}")
        >>> print(f"Avg time: {stats['avg_query_time']:.3f}s")
    """
    avg_results = (
        _retrieval_stats['total_results'] / _retrieval_stats['total_queries']
        if _retrieval_stats['total_queries'] > 0
        else 0.0
    )

    return {
        'total_queries': _retrieval_stats['total_queries'],
        'total_results': _retrieval_stats['total_results'],
        'avg_query_time': _retrieval_stats['avg_query_time'],
        'avg_results_per_query': avg_results
    }


def reset_retrieval_stats():
    """
    Reset retrieval statistics.

    Useful for testing or when starting a new monitoring period.
    """
    global _retrieval_stats
    _retrieval_stats = {
        'total_queries': 0,
        'total_results': 0,
        'avg_query_time': 0.0,
        'cache_hits': 0
    }
    logger.info("Retrieval statistics reset")
