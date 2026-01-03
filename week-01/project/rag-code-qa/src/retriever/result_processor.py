"""
Result Processor - Filter, Rank, and Convert Search Results

LEARNING CONTEXT:
This is Component 5 of your RAG-Code-QA pipeline (Week 1, Day 7-8).
After getting raw results from ChromaDB, you need to filter out noise,
rank by relevance, and convert to retrieval-specific data structures.

PURPOSE:
Processes raw search results from ChromaDB:
- Filters results below minimum score threshold (removes noise)
- Ranks results by similarity score (highest first)
- Converts StorageEntry to RetrievalResult objects
- Provides statistics and quality metrics

KEY CONCEPTS:
- Score thresholds filter noise (default: 0.3)
- Results should always be sorted by score (descending)
- Conversion adds retrieval-specific context
- Statistics help understand retrieval quality

SCORE THRESHOLDS:
- 0.7+ : Highly relevant (strong semantic match)
- 0.5-0.7 : Moderately relevant (good match)
- 0.3-0.5 : Possibly relevant (weak match)
- <0.3 : Likely noise (unrelated)

INTEGRATION POINTS:
- Input: List of StorageEntry objects from searcher
- Output: List of RetrievalResult objects for generator
"""

import logging
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..storage.models import StorageEntry
from .models import RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_by_score(
    results: List[StorageEntry],
    min_score: float = 0.3
) -> List[StorageEntry]:
    """
    Filter results below minimum similarity threshold.

    This removes noise and low-relevance results. The default threshold
    of 0.3 is based on validation testing (Experiment 1, Test 7) and
    filters out clearly unrelated results while keeping potentially
    relevant ones.

    Args:
        results: List of StorageEntry objects from search
        min_score: Minimum similarity score (0-1, default: 0.3)

    Returns:
        Filtered list of StorageEntry objects

    Example:
        >>> results = search_similar(query_vector, top_k=10)
        >>> filtered = filter_by_score(results, min_score=0.5)
        >>> all(r.score >= 0.5 for r in filtered)
        True
    """
    if not results:
        return []

    if min_score < 0.0 or min_score > 1.0:
        logger.warning(
            f"min_score should be between 0 and 1, got {min_score}. "
            "Using default 0.3"
        )
        min_score = 0.3

    filtered = [r for r in results if r.score is not None and r.score >= min_score]

    if len(filtered) < len(results):
        logger.info(
            f"Filtered {len(results) - len(filtered)} results below "
            f"threshold {min_score:.2f} ({len(filtered)} remaining)"
        )

    return filtered


def rank_by_score(results: List[StorageEntry]) -> List[StorageEntry]:
    """
    Ensure results are sorted by similarity score (highest first).

    ChromaDB should return results sorted, but this ensures consistent
    ordering regardless of the source. Always call this before returning
    results to the user.

    Args:
        results: List of StorageEntry objects

    Returns:
        Sorted list (descending by score)

    Example:
        >>> results = rank_by_score(results)
        >>> scores = [r.score for r in results]
        >>> scores == sorted(scores, reverse=True)
        True
    """
    if not results:
        return []

    # Sort by score (descending), handling None scores
    sorted_results = sorted(
        results,
        key=lambda r: r.score if r.score is not None else 0.0,
        reverse=True
    )

    logger.debug(f"Ranked {len(sorted_results)} results by score")

    return sorted_results


def convert_to_retrieval_results(
    entries: List[StorageEntry]
) -> List[RetrievalResult]:
    """
    Convert StorageEntry objects to RetrievalResult objects.

    This adds retrieval-specific context and properties to the results.
    RetrievalResult includes convenience methods for citations, relevance
    categorization, and display formatting.

    Args:
        entries: List of StorageEntry objects

    Returns:
        List of RetrievalResult objects

    Raises:
        ValueError: If any entry lacks a score

    Example:
        >>> storage_entries = search_similar(query_vector, top_k=5)
        >>> retrieval_results = convert_to_retrieval_results(storage_entries)
        >>> type(retrieval_results[0])
        <class 'retriever.models.RetrievalResult'>
    """
    if not entries:
        return []

    results = []
    for entry in entries:
        try:
            result = RetrievalResult.from_storage_entry(entry)
            results.append(result)
        except ValueError as e:
            logger.error(f"Failed to convert entry {entry.id[:16]}...: {e}")
            # Skip entries without scores
            continue

    logger.info(f"Converted {len(results)} storage entries to retrieval results")

    return results


def limit_results(results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
    """
    Limit results to top-k.

    This is typically called after filtering to ensure we return exactly
    the requested number of results (or fewer if not enough pass the filter).

    Args:
        results: List of RetrievalResult objects
        top_k: Maximum number of results to return

    Returns:
        Limited list of RetrievalResult objects

    Example:
        >>> results = limit_results(results, top_k=3)
        >>> len(results) <= 3
        True
    """
    if not results:
        return []

    if top_k <= 0:
        logger.warning(f"top_k must be positive, got {top_k}. Returning all results.")
        return results

    limited = results[:top_k]

    if len(limited) < len(results):
        logger.debug(f"Limited results from {len(results)} to {len(limited)}")

    return limited


def get_result_statistics(results: List[RetrievalResult]) -> Dict[str, Any]:
    """
    Calculate statistics about retrieval results.

    Provides insights into retrieval quality, score distribution, and
    content diversity. Useful for debugging, monitoring, and evaluation.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Dictionary with statistics:
        - total_results: Number of results
        - avg_score: Average similarity score
        - min_score: Minimum score
        - max_score: Maximum score
        - score_distribution: Count by relevance category
        - languages: Unique languages in results
        - files: Unique files in results
        - highly_relevant_count: Results with score >= 0.7

    Example:
        >>> stats = get_result_statistics(results)
        >>> print(f"Average score: {stats['avg_score']:.3f}")
        >>> print(f"Highly relevant: {stats['highly_relevant_count']}")
        Average score: 0.742
        Highly relevant: 3
    """
    if not results:
        return {
            'total_results': 0,
            'avg_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'score_distribution': {
                'highly_relevant': 0,
                'moderately_relevant': 0,
                'possibly_relevant': 0,
                'low_relevance': 0
            },
            'languages': [],
            'files': [],
            'highly_relevant_count': 0,
            'relevant_count': 0
        }

    scores = [r.score for r in results]
    languages = list(set(r.language for r in results))
    files = list(set(r.file_path for r in results))

    # Count by relevance category
    distribution = {
        'highly_relevant': len([r for r in results if r.is_highly_relevant]),
        'moderately_relevant': len([r for r in results if r.is_relevant and not r.is_highly_relevant]),
        'possibly_relevant': len([r for r in results if 0.3 <= r.score < 0.5]),
        'low_relevance': len([r for r in results if r.score < 0.3])
    }

    return {
        'total_results': len(results),
        'avg_score': sum(scores) / len(scores),
        'min_score': min(scores),
        'max_score': max(scores),
        'score_distribution': distribution,
        'languages': languages,
        'files': files,
        'highly_relevant_count': distribution['highly_relevant'],
        'relevant_count': distribution['highly_relevant'] + distribution['moderately_relevant']
    }


def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Remove duplicate results based on chunk_id.

    In rare cases, ChromaDB might return duplicates (e.g., if the same
    chunk was indexed multiple times). This ensures each chunk appears
    only once, keeping the highest-scored instance.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Deduplicated list of RetrievalResult objects

    Example:
        >>> results = deduplicate_results(results)
        >>> chunk_ids = [r.chunk_id for r in results]
        >>> len(chunk_ids) == len(set(chunk_ids))
        True
    """
    if not results:
        return []

    seen_ids = set()
    deduplicated = []

    for result in results:
        if result.chunk_id not in seen_ids:
            seen_ids.add(result.chunk_id)
            deduplicated.append(result)

    if len(deduplicated) < len(results):
        logger.info(
            f"Removed {len(results) - len(deduplicated)} duplicate results"
        )

    return deduplicated


def process_results(
    storage_entries: List[StorageEntry],
    min_score: float = 0.3,
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    Complete result processing pipeline.

    This is a convenience function that applies all processing steps:
    1. Filter by score
    2. Rank by score
    3. Convert to RetrievalResult
    4. Deduplicate
    5. Limit to top-k

    Args:
        storage_entries: Raw results from searcher
        min_score: Minimum similarity score (default: 0.3)
        top_k: Maximum number of results (default: 5)

    Returns:
        Processed list of RetrievalResult objects

    Example:
        >>> storage_entries = search_similar(query_vector, top_k=10)
        >>> results = process_results(storage_entries, min_score=0.5, top_k=3)
        >>> len(results) <= 3
        True
        >>> all(r.score >= 0.5 for r in results)
        True
    """
    logger.info(
        f"Processing {len(storage_entries)} results "
        f"(min_score={min_score}, top_k={top_k})"
    )

    # Step 1: Filter by score
    filtered = filter_by_score(storage_entries, min_score)

    # Step 2: Rank by score
    ranked = rank_by_score(filtered)

    # Step 3: Convert to RetrievalResult
    results = convert_to_retrieval_results(ranked)

    # Step 4: Deduplicate
    deduplicated = deduplicate_results(results)

    # Step 5: Limit to top-k
    final = limit_results(deduplicated, top_k)

    logger.info(f"Processed results: {len(storage_entries)} → {len(final)}")

    return final
