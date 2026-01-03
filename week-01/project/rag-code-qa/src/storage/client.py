"""
ChromaDB Client - Database Connection Management

LEARNING CONTEXT:
ChromaDB can run in two modes:
1. Embedded mode (PersistentClient) - What we're using for learning
2. Server mode (HttpClient) - For production multi-user scenarios

This module uses embedded mode: ChromaDB runs as a library within your Python
process, storing data as files in ./data/indexed/. No separate server required.

PURPOSE:
Provides a singleton ChromaDB client for the application:
- Initializes PersistentClient with local file storage
- Configures settings (disable telemetry, set persistence path)
- Implements singleton pattern (one client per process)

KEY CONCEPTS:
- Embedded vs Server mode trade-offs
- Persistent storage (data survives application restarts)
- Singleton pattern (reuse connection, avoid overhead)
- Configuration management

WHY EMBEDDED MODE:
- Zero setup (no server to manage)
- Perfect for learning and local development
- Works offline
- Easy upgrade path to server mode (just change client initialization)

PRODUCTION NOTES:
For production deployments with multiple users or distributed systems,
switch to HttpClient and run ChromaDB as a separate server.

DEPENDENCIES:
- chromadb library (already in requirements.txt)
"""

import os
import logging
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Global singleton client instance
_client_instance: Optional[chromadb.PersistentClient] = None


def create_chromadb_client(
    persist_directory: str = "./data/indexed",
    reset: bool = False
) -> chromadb.PersistentClient:
    """
    Create and configure a ChromaDB persistent client.

    This initializes ChromaDB in embedded mode, storing data as files in the
    specified directory. The client is NOT a separate process - it's a library
    that reads/writes files directly.

    Args:
        persist_directory: Path where ChromaDB will store data files
                          Default: ./data/indexed/
        reset: If True, clear existing singleton (for testing)

    Returns:
        Configured ChromaDB PersistentClient

    Raises:
        Exception: If client initialization fails

    Example:
        >>> client = create_chromadb_client()
        >>> collection = client.get_or_create_collection("my-collection")
    """
    global _client_instance

    # Return existing instance (singleton pattern)
    if _client_instance is not None and not reset:
        logger.debug("Reusing existing ChromaDB client instance")
        return _client_instance

    try:
        # Ensure persistence directory exists
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB client with persistence at: {persist_path.absolute()}")

        # Create client with settings
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,  # Disable usage tracking
                allow_reset=True  # Allow collection deletion (useful for development)
            )
        )

        # Store singleton instance
        _client_instance = client

        logger.info("ChromaDB client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        raise


def get_client() -> chromadb.PersistentClient:
    """
    Get the singleton ChromaDB client instance.

    If no client exists, creates one with default settings.
    This is the primary way to access the ChromaDB client throughout the app.

    Returns:
        Singleton ChromaDB PersistentClient

    Example:
        >>> client = get_client()
        >>> collection = client.get_or_create_collection("code-chunks")
    """
    global _client_instance

    if _client_instance is None:
        _client_instance = create_chromadb_client()

    return _client_instance


def reset_client() -> None:
    """
    Reset the singleton client instance.

    Useful for testing or when you need to reinitialize with different settings.
    The next call to get_client() will create a new instance.

    Warning:
        This does NOT delete the persisted data, it only resets the in-memory
        client instance. To delete data, use delete_collection().
    """
    global _client_instance
    _client_instance = None
    logger.info("ChromaDB client instance reset")


def get_client_info() -> dict:
    """
    Get information about the current client configuration.

    Returns:
        Dictionary with client configuration details

    Example:
        >>> info = get_client_info()
        >>> print(f"Storage path: {info['persist_directory']}")
    """
    client = get_client()

    # ChromaDB doesn't expose all settings directly, so we return what we know
    return {
        'client_type': 'PersistentClient',
        'mode': 'embedded',
        'persist_directory': './data/indexed/',  # Default path
        'telemetry_enabled': False,
        'collections': [col.name for col in client.list_collections()]
    }
