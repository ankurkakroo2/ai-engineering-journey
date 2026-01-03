"""
Retriever Module - Component 5 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is Component 5 of your RAG-Code-QA pipeline (Week 1, Day 7-8).
After indexing code (Parser → Chunker → Embedder → Storage), you now build
the query side of RAG. The retriever takes natural language queries, embeds
them, and searches ChromaDB for semantically similar code chunks.

PURPOSE:
Enables semantic search over indexed code:
- Embed user queries using the same model as indexing
- Search ChromaDB for similar vectors using cosine similarity
- Filter and rank results by relevance score
- Return top-k most relevant code chunks

KEY CONCEPTS:
- Query embeddings must use the same model as indexing (text-embedding-3-small)
- Cosine similarity measures semantic relevance (0-1 scale)
- Score thresholds filter noise (default: 0.3)
- Metadata filtering enables language/file-specific searches

ARCHITECTURE:
This module follows the facade pattern:
- retriever.py: Main orchestrator (public API)
- query_embedder.py: Embed queries (reuses embedder module)
- searcher.py: ChromaDB search (wraps storage module)
- result_processor.py: Filter, rank, convert results
- models.py: RetrievalResult dataclass

INTEGRATION POINTS:
- Input: Natural language query string
- Uses: embedder.api_client (for query embedding)
- Uses: storage.operations (for similarity search)
- Output: List of RetrievalResult objects (for generator module)

NEXT STEPS:
After retrieval, the generator module will use Claude to generate answers
grounded in the retrieved code chunks.
"""

from .models import RetrievalResult
from .retriever import retrieve

# Public API
__all__ = [
    'RetrievalResult',
    'retrieve'
]

