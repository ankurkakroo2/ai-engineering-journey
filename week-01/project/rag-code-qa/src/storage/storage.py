"""
Storage Manager - High-Level Facade for Vector Database Operations

LEARNING CONTEXT:
This is the main entry point for storage operations. It implements the Facade
design pattern: providing a simple, high-level API that hides the complexity
of client initialization, collection management, and CRUD operations.

PURPOSE:
Provides a clean, user-friendly interface for:
- Storing chunks with embeddings (upsert operation)
- Searching for similar chunks
- Getting collection statistics
- Listing stored entries
- Managing collections

KEY DESIGN PATTERNS:
- Facade: Simple interface over complex subsystem
- Singleton: Reuses client instance
- Dependency Injection: Accepts custom client/collection

WHY THIS ARCHITECTURE:
- Separation of concerns (client, collection, operations)
- Testability (can mock dependencies)
- Flexibility (can customize configuration)
- Simplicity (users don't need to understand ChromaDB internals)

USAGE:
    >>> from src.storage import StorageManager
    >>>
    >>> # Initialize manager
    >>> manager = StorageManager()
    >>>
    >>> # Store chunks
    >>> count = manager.store_chunks(chunks, embeddings)
    >>>
    >>> # Search similar
    >>> results = manager.search_similar(query_embedding, top_k=5)
    >>>
    >>> # Get stats
    >>> stats = manager.get_stats()
    >>> print(stats)

DEPENDENCIES:
- All other storage submodules (client, collection, operations, models)
- Chunk model (from chunker)
- Embedding model (from embedder)
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .client import get_client, create_chromadb_client
from .collection import (
    get_or_create_collection,
    delete_collection,
    get_collection_stats,
    list_all_collections
)
from .operations import (
    upsert_entries,
    query_similar,
    get_by_id,
    list_all,
    count_entries,
    delete_by_ids
)
from .models import StorageEntry, StorageStats
from ..chunker.models import Chunk
from ..embedder.models import Embedding

logger = logging.getLogger(__name__)


class StorageManager:
    """
    High-level interface for vector database operations.

    This class provides a simple API for storing and retrieving code embeddings.
    It handles all the complexity of ChromaDB initialization, collection
    management, and CRUD operations.

    Attributes:
        persist_directory: Path where ChromaDB stores data
        collection_name: Name of the collection to use
        distance_metric: Similarity metric (cosine, l2, or ip)

    Example:
        >>> manager = StorageManager()
        >>>
        >>> # Store chunks
        >>> manager.store_chunks(chunks, embeddings)
        >>>
        >>> # Search
        >>> results = manager.search_similar(query_embedding)
        >>> for result in results:
        ...     print(f"{result.function_name}: {result.score:.3f}")
    """

    def __init__(
        self,
        persist_directory: str = "./data/indexed",
        collection_name: str = "code-chunks",
        distance_metric: str = "cosine"
    ):
        """
        Initialize the StorageManager.

        Args:
            persist_directory: Path for ChromaDB data storage
            collection_name: Name of the collection
            distance_metric: Similarity metric (cosine recommended)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Initialize client and collection
        self._client = None
        self._collection = None

        logger.info(
            f"StorageManager initialized "
            f"(collection: {collection_name}, metric: {distance_metric})"
        )

    def _ensure_initialized(self):
        """Lazy initialization of client and collection."""
        if self._client is None:
            self._client = create_chromadb_client(self.persist_directory)

        if self._collection is None:
            self._collection = get_or_create_collection(
                self._client,
                name=self.collection_name,
                distance_metric=self.distance_metric
            )

    def store_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[Embedding]
    ) -> int:
        """
        Store chunks with their embeddings (upsert operation).

        This is the primary storage operation. It uses upsert, so:
        - New chunks are inserted
        - Existing chunks (same ID) are updated

        This enables incremental updates when re-indexing modified files.

        Args:
            chunks: List of Chunk objects to store
            embeddings: List of corresponding Embedding objects

        Returns:
            Number of entries stored

        Raises:
            ValueError: If chunks and embeddings don't match

        Example:
            >>> manager = StorageManager()
            >>> count = manager.store_chunks(chunks, embeddings)
            >>> print(f"Stored {count} chunks")
        """
        self._ensure_initialized()

        logger.info(f"Storing {len(chunks)} chunks...")
        count = upsert_entries(self._collection, chunks, embeddings)
        logger.info(f"Successfully stored {count} chunks")

        return count

    def update_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[Embedding]
    ) -> int:
        """
        Alias for store_chunks (both use upsert).

        Provided for semantic clarity when updating existing chunks.
        """
        return self.store_chunks(chunks, embeddings)

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[StorageEntry]:
        """
        Search for semantically similar chunks.

        This is the core retrieval operation for RAG. It finds the top-k
        chunks most similar to the query embedding.

        Args:
            query_embedding: Query vector (1536 dimensions)
            top_k: Number of results to return
            filter_by: Optional metadata filter (e.g., {"language": "python"})

        Returns:
            List of StorageEntry objects with similarity scores

        Example:
            >>> # Get query embedding
            >>> query_emb = embedder.embed_chunk(query_chunk)
            >>>
            >>> # Search for similar chunks
            >>> results = manager.search_similar(query_emb.vector, top_k=5)
            >>>
            >>> # Print results
            >>> for result in results:
            ...     print(f"{result.location}: {result.score:.3f}")
        """
        self._ensure_initialized()

        logger.info(f"Searching for top {top_k} similar chunks...")
        results = query_similar(
            self._collection,
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_by
        )

        logger.info(f"Found {len(results)} results")
        return results

    def get_by_id(self, chunk_id: str) -> Optional[StorageEntry]:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            StorageEntry if found, None otherwise

        Example:
            >>> entry = manager.get_by_id("abc123...")
            >>> if entry:
            ...     print(entry.function_name)
        """
        self._ensure_initialized()
        return get_by_id(self._collection, chunk_id)

    def get_stats(self) -> StorageStats:
        """
        Get collection statistics.

        Returns:
            StorageStats object with collection info

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Total entries: {stats.total_entries}")
            >>> print(f"Files: {stats.total_files}")
            >>> print(f"Languages: {', '.join(stats.languages)}")
        """
        self._ensure_initialized()
        return get_collection_stats(self._collection)

    def list_entries(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[StorageEntry]:
        """
        List all entries with pagination.

        Args:
            limit: Maximum number of entries (None = all)
            offset: Number of entries to skip

        Returns:
            List of StorageEntry objects

        Example:
            >>> # Get first 10 entries
            >>> entries = manager.list_entries(limit=10)
            >>> for entry in entries:
            ...     print(f"{entry.location}: {entry.function_name}")
        """
        self._ensure_initialized()
        return list_all(self._collection, limit=limit, offset=offset)

    def count(self) -> int:
        """
        Get total number of stored chunks.

        Returns:
            Total entry count

        Example:
            >>> count = manager.count()
            >>> print(f"Collection has {count} entries")
        """
        self._ensure_initialized()
        return count_entries(self._collection)

    def clear_collection(self) -> bool:
        """
        Delete the collection and all its data.

        This is a destructive operation. Use for complete re-indexing.

        Returns:
            True if collection was deleted

        Example:
            >>> if manager.clear_collection():
            ...     print("Collection cleared")
        """
        if self._client is None:
            self._client = create_chromadb_client(self.persist_directory)

        result = delete_collection(self._client, self.collection_name)

        # Reset collection reference
        self._collection = None

        return result

    def delete_entries(self, chunk_ids: List[str]) -> int:
        """
        Delete specific entries by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of entries deleted

        Example:
            >>> deleted = manager.delete_entries(["id1", "id2"])
            >>> print(f"Deleted {deleted} entries")
        """
        self._ensure_initialized()
        return delete_by_ids(self._collection, chunk_ids)

    def list_collections(self) -> List[str]:
        """
        List all collections in the database.

        Returns:
            List of collection names

        Example:
            >>> collections = manager.list_collections()
            >>> print(f"Collections: {', '.join(collections)}")
        """
        if self._client is None:
            self._client = create_chromadb_client(self.persist_directory)

        return list_all_collections(self._client)

    def __repr__(self) -> str:
        """String representation of StorageManager."""
        return (
            f"StorageManager(collection='{self.collection_name}', "
            f"metric='{self.distance_metric}', "
            f"path='{self.persist_directory}')"
        )
