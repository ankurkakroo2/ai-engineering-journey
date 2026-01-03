"""
Retriever Models - Data Structures for Query Results

LEARNING CONTEXT:
This is Component 5 of your RAG-Code-QA pipeline (Week 1, Day 7-8).
After searching ChromaDB for similar chunks, you need a data structure to
represent query results with similarity scores and metadata.

PURPOSE:
Defines the RetrievalResult dataclass, which represents a code chunk
retrieved from ChromaDB in response to a user query. It includes the
similarity score, content, and metadata needed for citations.

KEY CONCEPTS:
- RetrievalResult extends StorageEntry with query-specific context
- Similarity score (0-1) indicates relevance to the query
- Distance (from ChromaDB) is converted to similarity: similarity = 1 - distance
- Metadata enables citations (file paths, line numbers, function names)

DEPENDENCIES:
- Input: StorageEntry objects from storage module
- Output: RetrievalResult objects consumed by generator module
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..storage.models import StorageEntry


@dataclass
class RetrievalResult:
    """
    Represents a code chunk retrieved from ChromaDB in response to a query.

    This is the primary data structure for query results. It combines the
    original chunk content, its embedding, metadata, and most importantly,
    the similarity score indicating relevance to the query.

    Attributes:
        chunk_id: Unique chunk identifier (deterministic hash)
        content: Original chunk text (formatted function code)
        metadata: Rich metadata (file_path, function_name, lines, language, etc.)
        score: Similarity score (0-1, higher = more relevant)
        distance: ChromaDB distance (for debugging, lower = more similar)
        embedding: 1536-dimensional vector (optional, for advanced use)
    """
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: float
    embedding: Optional[List[float]] = None

    @property
    def location(self) -> str:
        """
        Returns citation-friendly location string.

        Format: file_path:start_line-end_line
        Example: src/parser/python_parser.py:54-65
        """
        file_path = self.metadata.get('file_path', 'unknown')
        start_line = self.metadata.get('start_line', 0)
        end_line = self.metadata.get('end_line', 0)
        return f"{file_path}:{start_line}-{end_line}"

    @property
    def function_name(self) -> str:
        """Returns the function name from metadata."""
        return self.metadata.get('name', 'unknown')

    @property
    def language(self) -> str:
        """Returns the programming language from metadata."""
        return self.metadata.get('language', 'unknown')

    @property
    def file_path(self) -> str:
        """Returns the file path from metadata."""
        return self.metadata.get('file_path', 'unknown')

    @property
    def start_line(self) -> int:
        """Returns the starting line number."""
        return self.metadata.get('start_line', 0)

    @property
    def end_line(self) -> int:
        """Returns the ending line number."""
        return self.metadata.get('end_line', 0)

    @property
    def is_relevant(self) -> bool:
        """
        Check if this result is considered relevant.

        Uses a threshold of 0.5 (moderately relevant).
        Scores above 0.7 are highly relevant.
        Scores below 0.3 are likely noise.
        """
        return self.score >= 0.5

    @property
    def is_highly_relevant(self) -> bool:
        """Check if this result is highly relevant (score >= 0.7)."""
        return self.score >= 0.7

    @property
    def relevance_category(self) -> str:
        """
        Categorize relevance based on score.

        Returns:
            'highly_relevant' (0.7+)
            'moderately_relevant' (0.5-0.7)
            'possibly_relevant' (0.3-0.5)
            'low_relevance' (<0.3)
        """
        if self.score >= 0.7:
            return 'highly_relevant'
        elif self.score >= 0.5:
            return 'moderately_relevant'
        elif self.score >= 0.3:
            return 'possibly_relevant'
        else:
            return 'low_relevance'

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Useful for debugging, logging, or API responses.
        """
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score,
            'distance': self.distance,
            'location': self.location,
            'function_name': self.function_name,
            'language': self.language,
            'relevance_category': self.relevance_category
        }

    @classmethod
    def from_storage_entry(cls, entry: StorageEntry) -> 'RetrievalResult':
        """
        Create a RetrievalResult from a StorageEntry.

        This is the primary conversion method used by the retriever module
        to transform storage results into retrieval results.

        Args:
            entry: StorageEntry from storage.operations.query_similar()

        Returns:
            RetrievalResult with score and distance populated

        Raises:
            ValueError: If entry doesn't have a score (not from a query)
        """
        if entry.score is None:
            raise ValueError(
                "Cannot create RetrievalResult from StorageEntry without score. "
                "Entry must come from a query operation."
            )

        # Calculate distance from score (reverse of: score = 1 - distance)
        distance = 1.0 - entry.score

        return cls(
            chunk_id=entry.id,
            content=entry.content,
            metadata=entry.metadata,
            score=entry.score,
            distance=distance,
            embedding=entry.embedding if hasattr(entry, 'embedding') else None
        )


def format_results_for_display(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results for terminal display.

    Creates a human-readable summary of results with scores and locations.
    Useful for debugging and manual testing.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Formatted string for display
    """
    if not results:
        return "No results found."

    lines = [f"\nFound {len(results)} results:\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"[{i}] {result.function_name}")
        lines.append(f"    Score: {result.score:.3f} ({result.relevance_category})")
        lines.append(f"    Location: {result.location}")
        lines.append(f"    Language: {result.language}")

        # Show content preview (first 2 lines)
        content_lines = result.content.split('\n')[:2]
        if content_lines:
            lines.append(f"    Preview: {content_lines[0][:60]}...")
        lines.append("")

    return "\n".join(lines)
