"""
Storage Module - Component 4 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is Component 4 of your RAG-Code-QA pipeline (Week 1, Day 6-7).
After generating embeddings from chunks, you need to persist them in a vector
database for efficient similarity search and retrieval.

The storage module manages ChromaDB (embedded mode) for:
- Storing code embeddings with metadata
- Similarity search using cosine distance
- Incremental updates via upsert operations
- Collection management and statistics

PURPOSE:
Provides a complete vector database solution:
- StorageManager: High-level facade for all operations
- StorageEntry: Represents stored chunks with scores
- StorageStats: Collection statistics and insights

KEY LEARNING OBJECTIVES:
1. Understand vector database operations (insert, query, upsert)
2. Learn why cosine similarity works for semantic search
3. Implement incremental updates using deterministic IDs
4. Build collection management utilities

ARCHITECTURE:
```
storage/
├── __init__.py          # Public API (you are here)
├── models.py            # StorageEntry, StorageStats
├── client.py            # ChromaDB client (singleton)
├── collection.py        # Collection lifecycle management
├── operations.py        # CRUD operations
└── storage.py           # StorageManager facade
```

PUBLIC API:
The module exposes a simple, high-level interface:

```python
from src.storage import StorageManager, StorageEntry, StorageStats

# Initialize manager
manager = StorageManager()

# Store chunks (upsert)
manager.store_chunks(chunks, embeddings)

# Search similar
results = manager.search_similar(query_embedding, top_k=5)

# Get statistics
stats = manager.get_stats()
```

DESIGN DECISIONS:
- ChromaDB in embedded mode (no server required)
- Cosine similarity (validated by experiments)
- Upsert as primary operation (enables incremental updates)
- Facade pattern (simple API over complex subsystem)

PRODUCTION NOTES:
This implementation uses ChromaDB in embedded mode for learning purposes.
Production deployments should consider:
- ChromaDB server mode for multi-user access
- Managed vector database services (Pinecone, Weaviate, Qdrant)
- Distributed systems for large-scale deployments

For this learning project, embedded mode is perfect: zero setup, works offline,
sufficient for personal use, and teaches core vector database concepts.

DEPENDENCIES:
- chromadb>=0.4.0 (already in requirements.txt)
- Chunk model from chunker module
- Embedding model from embedder module
"""

from .storage import StorageManager
from .models import StorageEntry, StorageStats
from .client import get_client, create_chromadb_client

# Public API
__all__ = [
    'StorageManager',
    'StorageEntry',
    'StorageStats',
    'get_client',
    'create_chromadb_client'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'RAG-Code-QA Project'
__description__ = 'Vector database storage for code embeddings'
