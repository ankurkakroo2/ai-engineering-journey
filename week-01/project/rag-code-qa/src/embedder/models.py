"""
Embedder Models - Data Structures for Vector Embeddings

LEARNING CONTEXT:
This is Component 3 of your RAG-Code-QA pipeline (Week 1, Day 5-6).
After chunking functions into embeddable units, the embedder transforms text
into high-dimensional vectors (1536 dimensions) for semantic search.

PURPOSE:
Defines the Embedding dataclass - the vector representation of a chunk.
Each embedding represents one chunk as a point in 1536-dimensional space where
semantic similarity = geometric proximity (cosine similarity).

KEY CONCEPTS:
- Embeddings are deterministic (same input = same output)
- 1536 dimensions capture nuanced semantic relationships
- Vectors enable similarity search via cosine distance
- Cached flag tracks whether embedding came from cache or API

DEPENDENCIES:
- Input: Chunk objects from chunker module
- Output: Embedding objects consumed by storage module
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Embedding:
    chunk_id: str
    vector: List[float]
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None

    @property
    def dimension(self) -> int:
        return len(self.vector)

    @property
    def is_valid(self) -> bool:
        return (
            self.dimension == 1536 and
            len(self.chunk_id) > 0 and
            all(isinstance(x, (int, float)) for x in self.vector)
        )

    @property
    def as_numpy(self) -> np.ndarray:
        return np.array(self.vector, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'vector': self.vector,
            'cached': self.cached,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Embedding':
        return cls(
            chunk_id=data['chunk_id'],
            vector=data['vector'],
            cached=data.get('cached', False),
            metadata=data.get('metadata')
        )
