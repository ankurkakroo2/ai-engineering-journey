"""
Storage Operations - CRUD Operations for Vector Database

LEARNING CONTEXT:
This module implements the core database operations for storing and retrieving
code embeddings. It bridges the gap between your application's data structures
(Chunk, Embedding) and ChromaDB's API.

PURPOSE:
Provides CRUD (Create, Read, Update, Delete) operations:
- add_entries(): Insert new chunks (fails if ID exists)
- upsert_entries(): Insert or update chunks (primary operation)
- query_similar(): Find semantically similar chunks
- get_by_id(): Retrieve specific chunk by ID
- list_all(): Get all chunks (with pagination)
- count_entries(): Get total count

KEY CONCEPTS:
- Upsert = Update if exists, Insert if new (enables incremental updates)
- Batch operations = Process multiple items efficiently
- Similarity search = Find nearest neighbors in vector space
- Metadata filtering = Narrow results by file, language, etc.

WHY UPSERT IS PRIMARY:
With deterministic chunk IDs (from chunker), upsert enables:
- Incremental updates (only re-index changed files)
- Deduplication (same content = same ID)
- Idempotent operations (safe to run multiple times)

DEPENDENCIES:
- ChromaDB Collection (from collection.py)
- Chunk and Embedding models (from chunker and embedder)
- StorageEntry model (from models.py)
"""

import logging
from typing import List, Optional, Dict, Any
from chromadb.api.models.Collection import Collection

from ..chunker.models import Chunk
from ..embedder.models import Embedding
from .models import StorageEntry

logger = logging.getLogger(__name__)


def add_entries(
    collection: Collection,
    chunks: List[Chunk],
    embeddings: List[Embedding]
) -> int:
    """
    Add new entries to the collection.

    This operation will FAIL if any chunk ID already exists in the collection.
    For incremental updates, use upsert_entries() instead.

    Args:
        collection: ChromaDB Collection object
        chunks: List of Chunk objects to store
        embeddings: List of corresponding Embedding objects

    Returns:
        Number of entries successfully added

    Raises:
        ValueError: If chunks and embeddings lists don't match
        Exception: If ChromaDB add operation fails

    Example:
        >>> collection = get_or_create_collection(client)
        >>> count = add_entries(collection, chunks, embeddings)
        >>> print(f"Added {count} entries")
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "count mismatch"
        )

    if not chunks:
        logger.warning("No chunks to add")
        return 0

    try:
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        vectors = [emb.vector for emb in embeddings]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Successfully added {len(chunks)} entries to collection")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to add entries: {e}")
        raise


def upsert_entries(
    collection: Collection,
    chunks: List[Chunk],
    embeddings: List[Embedding]
) -> int:
    """
    Insert or update entries in the collection.

    This is the PRIMARY operation for storing chunks. It's idempotent:
    - If chunk ID exists: updates the entry
    - If chunk ID is new: inserts the entry

    This enables incremental updates: when you re-index a modified file,
    only the changed functions get updated, others remain unchanged.

    Args:
        collection: ChromaDB Collection object
        chunks: List of Chunk objects to store
        embeddings: List of corresponding Embedding objects

    Returns:
        Number of entries successfully upserted

    Raises:
        ValueError: If chunks and embeddings lists don't match
        Exception: If ChromaDB upsert operation fails

    Example:
        >>> collection = get_or_create_collection(client)
        >>> count = upsert_entries(collection, chunks, embeddings)
        >>> print(f"Upserted {count} entries")
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            "count mismatch"
        )

    if not chunks:
        logger.warning("No chunks to upsert")
        return 0

    try:
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        vectors = [emb.vector for emb in embeddings]
        metadatas = [chunk.metadata for chunk in chunks]

        # Upsert to collection
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Successfully upserted {len(chunks)} entries to collection")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to upsert entries: {e}")
        raise


def query_similar(
    collection: Collection,
    query_embedding: List[float],
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[StorageEntry]:
    """
    Find semantically similar chunks using vector similarity search.

    This is the core retrieval operation for RAG:
    1. Takes a query embedding (from embedder)
    2. Finds top-k nearest neighbors in vector space
    3. Returns results with similarity scores

    ChromaDB uses HNSW indexing for fast approximate nearest neighbor search.

    Args:
        collection: ChromaDB Collection object
        query_embedding: Query vector (1536 dimensions)
        top_k: Number of results to return (default: 5)
        filter_metadata: Optional metadata filter (e.g., {"language": "python"})

    Returns:
        List of StorageEntry objects, sorted by similarity (highest first)

    Example:
        >>> # Get query embedding
        >>> query_emb = embed_chunk(client, query_chunk)
        >>> # Search for similar chunks
        >>> results = query_similar(collection, query_emb.vector, top_k=5)
        >>> for result in results:
        ...     print(f"{result.function_name}: {result.score:.3f}")
    """
    try:
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,  # Optional metadata filter
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        # ChromaDB returns nested lists (for batch queries)
        # We're doing single query, so extract first element
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]

        # Convert to StorageEntry objects
        entries = []
        for i in range(len(ids)):
            entry = StorageEntry.from_chromadb_result(
                chunk_id=ids[i],
                content=documents[i],
                embedding=embeddings[i],
                metadata=metadatas[i],
                distance=distances[i]
            )
            entries.append(entry)

        logger.info(
            f"Query returned {len(entries)} results "
            f"(requested top_k={top_k})"
        )

        return entries

    except Exception as e:
        logger.error(f"Failed to query similar entries: {e}")
        raise


def get_by_id(
    collection: Collection,
    chunk_id: str
) -> Optional[StorageEntry]:
    """
    Retrieve a specific chunk by its ID.

    Args:
        collection: ChromaDB Collection object
        chunk_id: Unique chunk identifier

    Returns:
        StorageEntry if found, None otherwise

    Example:
        >>> entry = get_by_id(collection, "abc123...")
        >>> if entry:
        ...     print(f"Found: {entry.function_name}")
    """
    try:
        results = collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"]
        )

        ids = results.get("ids", [])
        if not ids:
            logger.debug(f"Chunk ID not found: {chunk_id}")
            return None

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])

        entry = StorageEntry(
            id=ids[0],
            content=documents[0],
            embedding=embeddings[0],
            metadata=metadatas[0],
            score=None  # No score for direct retrieval
        )

        return entry

    except Exception as e:
        logger.error(f"Failed to get entry by ID: {e}")
        return None


def list_all(
    collection: Collection,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[StorageEntry]:
    """
    List all entries in the collection with pagination.

    Useful for debugging and inspecting what's stored.

    Args:
        collection: ChromaDB Collection object
        limit: Maximum number of entries to return (None = all)
        offset: Number of entries to skip (for pagination)

    Returns:
        List of StorageEntry objects

    Example:
        >>> # Get first 10 entries
        >>> entries = list_all(collection, limit=10, offset=0)
        >>> for entry in entries:
        ...     print(f"{entry.location}: {entry.function_name}")
    """
    try:
        # Get entries with pagination
        results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        embeddings = results.get("embeddings", [])

        entries = []
        for i in range(len(ids)):
            entry = StorageEntry(
                id=ids[i],
                content=documents[i],
                embedding=embeddings[i],
                metadata=metadatas[i],
                score=None
            )
            entries.append(entry)

        logger.info(f"Listed {len(entries)} entries (limit={limit}, offset={offset})")
        return entries

    except Exception as e:
        logger.error(f"Failed to list entries: {e}")
        return []


def count_entries(
    collection: Collection
) -> int:
    """
    Get total number of entries in the collection.

    Args:
        collection: ChromaDB Collection object

    Returns:
        Total entry count

    Example:
        >>> count = count_entries(collection)
        >>> print(f"Collection has {count} entries")
    """
    try:
        return collection.count()
    except Exception as e:
        logger.error(f"Failed to count entries: {e}")
        return 0


def delete_by_ids(
    collection: Collection,
    chunk_ids: List[str]
) -> int:
    """
    Delete specific entries by their IDs.

    Useful for removing outdated or deleted functions.

    Args:
        collection: ChromaDB Collection object
        chunk_ids: List of chunk IDs to delete

    Returns:
        Number of entries deleted

    Example:
        >>> deleted = delete_by_ids(collection, ["id1", "id2"])
        >>> print(f"Deleted {deleted} entries")
    """
    if not chunk_ids:
        logger.warning("No IDs provided for deletion")
        return 0

    try:
        collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} entries")
        return len(chunk_ids)

    except Exception as e:
        logger.error(f"Failed to delete entries: {e}")
        raise
