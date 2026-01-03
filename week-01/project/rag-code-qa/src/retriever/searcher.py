"""
Searcher - ChromaDB Similarity Search Operations

LEARNING CONTEXT:
This is Component 5 of your RAG-Code-QA pipeline (Week 1, Day 7-8).
After embedding the query, you need to search ChromaDB for similar vectors
using cosine similarity to find semantically relevant code chunks.

PURPOSE:
Wraps the storage module's query operations to provide a retriever-specific
interface for similarity search. Handles ChromaDB interactions and returns
StorageEntry objects with similarity scores.

KEY CONCEPTS:
- Uses ChromaDB's HNSW index for fast approximate nearest neighbor search
- Cosine similarity measures semantic relevance (0-1 scale)
- Returns top-k results ranked by similarity
- Supports metadata filtering (language, file path, etc.)

HOW SIMILARITY SEARCH WORKS:
1. Query vector is compared against ALL stored vectors in ChromaDB
2. ChromaDB uses HNSW (Hierarchical Navigable Small World) for efficiency
3. Returns distances (lower = more similar)
4. Storage module converts distance to similarity: similarity = 1 - distance

INTEGRATION POINTS:
- Input: Query embedding vector (1536 dimensions)
- Uses: storage.StorageManager (wraps operations.query_similar)
- Output: List of StorageEntry objects with scores
"""

import logging
from typing import List, Optional, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..storage import StorageManager
from ..storage.models import StorageEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_similar(
    query_vector: List[float],
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    storage_manager: Optional[StorageManager] = None
) -> List[StorageEntry]:
    """
    Search ChromaDB for similar chunks using vector similarity.

    This is the core search function. It takes a query embedding and finds
    the top-k most similar chunks in the vector database using cosine
    similarity.

    Args:
        query_vector: 1536-dimensional query embedding
        top_k: Number of results to return (default: 5)
        filter_metadata: Optional metadata filter (e.g., {"language": "python"})
        storage_manager: Optional StorageManager instance (created if not provided)

    Returns:
        List of StorageEntry objects, sorted by similarity (highest first)

    Raises:
        ValueError: If query_vector is invalid or search fails

    Example:
        >>> from retriever.query_embedder import embed_query
        >>> query_vector = embed_query("How does authentication work?")
        >>> results = search_similar(query_vector, top_k=3)
        >>> for result in results:
        ...     print(f"{result.function_name}: {result.score:.3f}")
        login_handler: 0.872
        verify_token: 0.819
        authenticate_user: 0.785
    """
    # Validate input
    if not query_vector:
        raise ValueError("Query vector cannot be empty")

    if len(query_vector) != 1536:
        raise ValueError(
            f"Query vector must be 1536-dimensional, got {len(query_vector)}"
        )

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    logger.info(
        f"Searching for top {top_k} similar chunks"
        f"{' with metadata filter' if filter_metadata else ''}"
    )

    # Create storage manager if not provided
    if storage_manager is None:
        storage_manager = StorageManager()

    try:
        # Delegate to storage module's search_similar method
        # Note: We don't filter by score here because filtering is done
        # in result_processor.py for better separation of concerns
        results = storage_manager.search_similar(
            query_embedding=query_vector,
            top_k=top_k,
            filter_by=filter_metadata  # Pass metadata filter
        )

        logger.info(f"Found {len(results)} results from ChromaDB")

        # Log score distribution for debugging
        if results:
            scores = [r.score for r in results if r.score is not None]
            if scores:
                logger.debug(
                    f"Score range: {min(scores):.3f} - {max(scores):.3f}"
                )

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise ValueError(f"Failed to search ChromaDB: {e}")


def search_by_language(
    query_vector: List[float],
    language: str,
    top_k: int = 5,
    storage_manager: Optional[StorageManager] = None
) -> List[StorageEntry]:
    """
    Search for similar chunks filtered by programming language.

    This is a convenience function for language-specific searches.
    Useful when you want results only from Python, JavaScript, or TypeScript.

    Args:
        query_vector: 1536-dimensional query embedding
        language: Programming language filter ("python", "javascript", "typescript")
        top_k: Number of results to return (default: 5)
        storage_manager: Optional StorageManager instance

    Returns:
        List of StorageEntry objects from the specified language

    Example:
        >>> query_vector = embed_query("authentication")
        >>> python_results = search_by_language(query_vector, "python", top_k=5)
        >>> all(r.language == "python" for r in python_results)
        True
    """
    logger.info(f"Searching for {language} functions only")

    filter_metadata = {"language": language}

    return search_similar(
        query_vector=query_vector,
        top_k=top_k,
        filter_metadata=filter_metadata,
        storage_manager=storage_manager
    )


def search_by_file(
    query_vector: List[float],
    file_path: str,
    top_k: int = 5,
    storage_manager: Optional[StorageManager] = None
) -> List[StorageEntry]:
    """
    Search for similar chunks filtered by file path.

    This is useful when you want to search within a specific file or
    directory. The file_path can be a full path or a partial path.

    Args:
        query_vector: 1536-dimensional query embedding
        file_path: File path filter (exact match)
        top_k: Number of results to return (default: 5)
        storage_manager: Optional StorageManager instance

    Returns:
        List of StorageEntry objects from the specified file

    Example:
        >>> query_vector = embed_query("visitor pattern")
        >>> results = search_by_file(query_vector, "src/parser/python_parser.py")
        >>> all(file_path in r.file_path for r in results)
        True
    """
    logger.info(f"Searching in file: {file_path}")

    filter_metadata = {"file_path": file_path}

    return search_similar(
        query_vector=query_vector,
        top_k=top_k,
        filter_metadata=filter_metadata,
        storage_manager=storage_manager
    )


def get_search_stats(results: List[StorageEntry]) -> Dict[str, Any]:
    """
    Get statistics about search results.

    Provides insights into the quality and distribution of search results.
    Useful for debugging and understanding retrieval performance.

    Args:
        results: List of StorageEntry objects from search

    Returns:
        Dictionary with statistics:
        - total_results: Number of results
        - avg_score: Average similarity score
        - min_score: Minimum score
        - max_score: Maximum score
        - languages: List of unique languages
        - files: List of unique files

    Example:
        >>> results = search_similar(query_vector, top_k=5)
        >>> stats = get_search_stats(results)
        >>> print(f"Average score: {stats['avg_score']:.3f}")
        Average score: 0.742
    """
    if not results:
        return {
            'total_results': 0,
            'avg_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'languages': [],
            'files': []
        }

    scores = [r.score for r in results if r.score is not None]
    languages = list(set(r.language for r in results))
    files = list(set(r.file_path for r in results))

    return {
        'total_results': len(results),
        'avg_score': sum(scores) / len(scores) if scores else 0.0,
        'min_score': min(scores) if scores else 0.0,
        'max_score': max(scores) if scores else 0.0,
        'languages': languages,
        'files': files,
        'score_distribution': {
            'highly_relevant': len([s for s in scores if s >= 0.7]),
            'moderately_relevant': len([s for s in scores if 0.5 <= s < 0.7]),
            'possibly_relevant': len([s for s in scores if 0.3 <= s < 0.5]),
            'low_relevance': len([s for s in scores if s < 0.3])
        }
    }
