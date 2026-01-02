"""
Chunker Models - Data Structures for Embeddable Units

LEARNING CONTEXT:
This is Component 2 of your RAG-Code-QA pipeline (Week 1, Day 4-5).
After parsing functions into structured objects (Component 1), you need to
transform them into embeddable units that fit within token limits and preserve
all metadata needed for semantic search and citations.

PURPOSE:
Defines the Chunk dataclass - the fundamental unit of your RAG system.
Each chunk represents one function, formatted as embeddable text with:
- Deterministic ID (for caching and incremental updates)
- Content string (ready for embedding API)
- Metadata (for citations and filtering)
- Token count (for validation and monitoring)

KEY CONCEPTS:
- One function = One chunk (semantic chunking strategy)
- Deterministic IDs enable change detection and deduplication
- Metadata preservation enables precise citations in generated answers
- Token tracking ensures compliance with embedding model limits

DEPENDENCIES:
- Input: ParsedFunction from parser module
- Output: Chunk objects consumed by embedder module
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    token_count: int
    truncated: bool = False

    @property
    def is_valid(self) -> bool:
        return self.token_count > 0 and len(self.content.strip()) > 0

    @property
    def location(self) -> str:
        return f"{self.metadata.get('file_path', 'unknown')}:{self.metadata.get('start_line', 0)}-{self.metadata.get('end_line', 0)}"

    @property
    def function_name(self) -> str:
        return self.metadata.get('name', 'unknown')

    @property
    def has_docstring(self) -> bool:
        return self.metadata.get('docstring') is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'token_count': self.token_count,
            'truncated': self.truncated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        return cls(
            id=data['id'],
            content=data['content'],
            metadata=data['metadata'],
            token_count=data['token_count'],
            truncated=data.get('truncated', False)
        )
