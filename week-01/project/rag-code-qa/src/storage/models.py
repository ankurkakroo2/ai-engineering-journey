"""
Storage Models - Data Structures for Vector Database Operations

LEARNING CONTEXT:
This is Component 4 of your RAG-Code-QA pipeline (Week 1, Day 6-7).
After generating embeddings from chunks, you need to persist them in a vector
database (ChromaDB) for efficient similarity search and retrieval.

PURPOSE:
Defines data structures for storage operations:
- StorageEntry: Represents a stored chunk with its embedding and metadata
- StorageStats: Collection statistics for monitoring and debugging

KEY CONCEPTS:
- Storage entries combine chunks, embeddings, and similarity scores
- Metadata preservation enables citations and filtering
- Statistics provide insights into indexed content
- Score field enables ranking of query results

DEPENDENCIES:
- Input: Chunk objects (from chunker) + Embedding objects (from embedder)
- Output: StorageEntry objects consumed by retriever module
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class StorageEntry:
    """
    Represents a stored chunk with its embedding and metadata.

    This is the unified representation of a code chunk after it has been
    stored in ChromaDB. It combines the original chunk content, its vector
    embedding, and all metadata needed for citations and filtering.

    Attributes:
        id: Unique chunk identifier (deterministic hash from chunker)
        content: Original chunk text (formatted function code)
        embedding: 1536-dimensional vector from OpenAI
        metadata: Rich metadata (file_path, function_name, lines, language, etc.)
        score: Similarity score (0-1) when retrieved from query (None when stored)
    """
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None

    @property
    def location(self) -> str:
        """Returns citation-friendly location string (file:start-end)."""
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
    def has_score(self) -> bool:
        """Check if this entry has a similarity score (from query)."""
        return self.score is not None

    @property
    def embedding_dimension(self) -> int:
        """Returns the dimensionality of the embedding vector."""
        return len(self.embedding)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'score': self.score
        }

    @classmethod
    def from_chromadb_result(
        cls,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        distance: Optional[float] = None
    ) -> 'StorageEntry':
        """
        Create StorageEntry from ChromaDB query result.

        ChromaDB returns distance (0 = identical, higher = more different).
        We convert to similarity score (1 = identical, 0 = completely different).
        For cosine distance: similarity = 1 - distance

        Args:
            chunk_id: Unique chunk identifier
            content: Chunk text content
            embedding: Vector embedding
            metadata: Chunk metadata
            distance: ChromaDB distance (optional, for query results)

        Returns:
            StorageEntry with similarity score if distance provided
        """
        score = None
        if distance is not None:
            # Convert distance to similarity score
            # Cosine distance: 0 (identical) to 2 (opposite)
            # Similarity: 1 (identical) to 0 (orthogonal) to -1 (opposite)
            score = 1.0 - distance

        return cls(
            id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            score=score
        )


@dataclass
class StorageStats:
    """
    Collection statistics for monitoring and debugging.

    Provides insights into what's stored in the vector database:
    - How many chunks are indexed
    - Which files and languages are represented
    - Collection configuration (distance metric, dimensions)
    - Timestamps for tracking updates

    Attributes:
        total_entries: Total number of stored chunks
        total_files: Number of unique source files
        languages: List of programming languages in collection
        collection_name: Name of the ChromaDB collection
        dimension: Embedding vector dimension (should be 1536)
        distance_metric: Similarity metric (should be 'cosine')
        created_at: Collection creation timestamp (if available)
        last_updated: Last modification timestamp (if available)
    """
    total_entries: int
    total_files: int
    languages: List[str]
    collection_name: str
    dimension: int
    distance_metric: str
    created_at: Optional[str] = None
    last_updated: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        """Check if collection has no entries."""
        return self.total_entries == 0

    @property
    def avg_entries_per_file(self) -> float:
        """Calculate average number of functions per file."""
        if self.total_files == 0:
            return 0.0
        return self.total_entries / self.total_files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_entries': self.total_entries,
            'total_files': self.total_files,
            'languages': self.languages,
            'collection_name': self.collection_name,
            'dimension': self.dimension,
            'distance_metric': self.distance_metric,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Collection: {self.collection_name}\n"
            f"  Entries: {self.total_entries}\n"
            f"  Files: {self.total_files}\n"
            f"  Languages: {', '.join(self.languages) if self.languages else 'none'}\n"
            f"  Dimension: {self.dimension}\n"
            f"  Distance: {self.distance_metric}\n"
            f"  Avg per file: {self.avg_entries_per_file:.1f}"
        )
