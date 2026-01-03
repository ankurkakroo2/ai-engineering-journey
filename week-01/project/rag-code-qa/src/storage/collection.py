"""
Collection Management - ChromaDB Collection Lifecycle Operations

LEARNING CONTEXT:
A ChromaDB collection is like a table in a traditional database. It stores
vectors with their metadata and provides similarity search capabilities.

This module manages collection lifecycle:
- Creation with specific distance metrics (cosine similarity)
- Deletion for re-indexing
- Existence checks
- Statistics gathering

PURPOSE:
Provides collection management operations:
- get_or_create_collection(): Initialize or connect to collection
- delete_collection(): Remove collection and all its data
- collection_exists(): Check if collection exists
- get_collection_stats(): Gather statistics about stored data

KEY CONCEPTS:
- Collection = namespace for vectors (one per project/codebase)
- Distance metric = how similarity is calculated (cosine for us)
- Metadata = collection-level configuration
- Statistics = insights into indexed content

WHY COSINE SIMILARITY:
Validated by Experiment 1, Test 4:
- Measures angle between vectors, not distance
- Handles length variation (short vs verbose code)
- Normalized: always returns score between 0 and 1
- Better than Euclidean for high-dimensional semantic vectors

DEPENDENCIES:
- ChromaDB client (from client.py)
- StorageStats model (from models.py)
"""

import logging
from typing import Optional, List
from datetime import datetime
import chromadb
from chromadb.api.models.Collection import Collection

from .models import StorageStats

logger = logging.getLogger(__name__)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = "code-chunks",
    distance_metric: str = "cosine"
) -> Collection:
    """
    Get existing collection or create new one with specified configuration.

    This is the primary way to access a collection. It's idempotent: calling
    multiple times with same parameters returns the same collection.

    Args:
        client: ChromaDB client instance
        name: Collection name (default: "code-chunks")
        distance_metric: Similarity metric ("cosine", "l2", or "ip")
                        Default: "cosine" (recommended for semantic search)

    Returns:
        ChromaDB Collection object

    Raises:
        ValueError: If invalid distance metric specified
        Exception: If collection creation fails

    Example:
        >>> from .client import get_client
        >>> client = get_client()
        >>> collection = get_or_create_collection(client)
        >>> print(f"Collection: {collection.name}")
    """
    valid_metrics = ["cosine", "l2", "ip"]
    if distance_metric not in valid_metrics:
        raise ValueError(
            f"Invalid distance metric: {distance_metric}. "
            f"Must be one of: {', '.join(valid_metrics)}"
        )

    try:
        # ChromaDB's get_or_create_collection is idempotent
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_metric}  # Set distance metric
        )

        logger.info(
            f"Collection '{name}' ready (metric: {distance_metric}, "
            f"entries: {collection.count()})"
        )

        return collection

    except Exception as e:
        logger.error(f"Failed to get/create collection '{name}': {e}")
        raise


def delete_collection(
    client: chromadb.PersistentClient,
    name: str = "code-chunks"
) -> bool:
    """
    Delete a collection and all its data.

    This is a destructive operation - all vectors, documents, and metadata
    in the collection will be permanently deleted.

    Use cases:
    - Complete re-indexing from scratch
    - Cleaning up test data
    - Removing old/unused collections

    Args:
        client: ChromaDB client instance
        name: Collection name to delete

    Returns:
        True if collection was deleted, False if it didn't exist

    Example:
        >>> from .client import get_client
        >>> client = get_client()
        >>> deleted = delete_collection(client, "old-collection")
        >>> if deleted:
        ...     print("Collection deleted successfully")
    """
    try:
        # Check if collection exists first
        if not collection_exists(client, name):
            logger.warning(f"Collection '{name}' does not exist, nothing to delete")
            return False

        client.delete_collection(name=name)
        logger.info(f"Collection '{name}' deleted successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to delete collection '{name}': {e}")
        raise


def collection_exists(
    client: chromadb.PersistentClient,
    name: str
) -> bool:
    """
    Check if a collection exists.

    Args:
        client: ChromaDB client instance
        name: Collection name to check

    Returns:
        True if collection exists, False otherwise

    Example:
        >>> from .client import get_client
        >>> client = get_client()
        >>> if collection_exists(client, "code-chunks"):
        ...     print("Collection exists")
    """
    try:
        collections = client.list_collections()
        return any(col.name == name for col in collections)
    except Exception as e:
        logger.error(f"Failed to check collection existence: {e}")
        return False


def get_collection_stats(
    collection: Collection
) -> StorageStats:
    """
    Gather statistics about a collection.

    Analyzes the collection's metadata to provide insights:
    - Total number of entries
    - Number of unique files
    - Programming languages represented
    - Collection configuration

    Args:
        collection: ChromaDB Collection object

    Returns:
        StorageStats object with collection statistics

    Example:
        >>> collection = get_or_create_collection(client)
        >>> stats = get_collection_stats(collection)
        >>> print(f"Total entries: {stats.total_entries}")
        >>> print(f"Languages: {', '.join(stats.languages)}")
    """
    try:
        # Get total count
        total_entries = collection.count()

        if total_entries == 0:
            # Empty collection
            return StorageStats(
                total_entries=0,
                total_files=0,
                languages=[],
                collection_name=collection.name,
                dimension=1536,  # Default expected dimension
                distance_metric=collection.metadata.get("hnsw:space", "cosine"),
                created_at=None,
                last_updated=None
            )

        # Get all entries to analyze metadata
        # Note: For large collections, this could be expensive
        # In production, you'd want to maintain these stats separately
        results = collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        # Extract unique files and languages
        unique_files = set()
        unique_languages = set()

        for metadata in metadatas:
            if metadata:
                file_path = metadata.get("file_path")
                if file_path:
                    unique_files.add(file_path)

                language = metadata.get("language")
                if language:
                    unique_languages.add(language)

        # Get collection metadata
        distance_metric = collection.metadata.get("hnsw:space", "cosine")

        return StorageStats(
            total_entries=total_entries,
            total_files=len(unique_files),
            languages=sorted(list(unique_languages)),
            collection_name=collection.name,
            dimension=1536,  # Expected dimension for text-embedding-3-small
            distance_metric=distance_metric,
            created_at=None,  # ChromaDB doesn't track this by default
            last_updated=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        # Return minimal stats on error
        return StorageStats(
            total_entries=0,
            total_files=0,
            languages=[],
            collection_name=collection.name,
            dimension=1536,
            distance_metric="cosine",
            created_at=None,
            last_updated=None
        )


def list_all_collections(
    client: chromadb.PersistentClient
) -> List[str]:
    """
    List all collection names in the database.

    Useful for debugging and understanding what's stored.

    Args:
        client: ChromaDB client instance

    Returns:
        List of collection names

    Example:
        >>> from .client import get_client
        >>> client = get_client()
        >>> collections = list_all_collections(client)
        >>> print(f"Collections: {', '.join(collections)}")
    """
    try:
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []
