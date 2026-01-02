"""
Chunker Module - Component 2 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is the second component in your 8-week AI Engineering Journey (Week 1).
After parsing functions from source code (Component 1), the chunker transforms
them into embeddable units that fit within token limits and preserve metadata.

WHY CHUNKING MATTERS:
The embedding API has hard limits (~8000 tokens). Functions vary wildly in size
(10 lines to 500+ lines). You need to:
1. Transform structured objects into embeddable text strings
2. Normalize variable-length content to fit within limits
3. Assign deterministic IDs for caching and incremental updates

CORE INVARIANT:
**One function = One chunk** (semantic chunking strategy)

This ensures complete semantic units, precise retrieval, and clean boundaries.
Unlike fixed-size chunking (arbitrary 512-token windows), semantic chunking
respects natural code boundaries.

THE THREE TRANSFORMATIONS:
1. Content Formatting (formatter.py)
   ParsedFunction → Formatted text string
   Adds location header, preserves docstring, includes code

2. Token Normalization (token_counter.py)
   Variable-length text → Token-constrained text
   Counts tokens accurately, validates limits, enables truncation

3. Identity Assignment (hasher.py)
   Content + Metadata → Deterministic ID
   SHA-256 hash enables change detection and deduplication

CHUNK ANATOMY:

    # /path/to/file.py:70-73          <- Location (spatial context)
    \"\"\"Returns location string\"\"\"      <- Docstring (semantic meaning)
    @property                          <- Code (implementation)
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}"

This three-layer format optimizes embedding quality:
- Layer 1: Helps queries like "functions in models.py"
- Layer 2: Matches user queries (English, not code syntax)
- Layer 3: Captures technical details and patterns

INCREMENTAL UPDATES:
Deterministic IDs enable efficient re-indexing:
- Unchanged function → same ID → cache hit (skip embedding)
- Modified function → different ID → new embedding needed
- Deleted function → old ID not in new set → remove from DB

KEY LEARNING OBJECTIVES:
1. Understand why embedding models need text strings, not objects
2. Learn how token counting differs from character/word counting
3. See how content-based hashing enables change detection
4. Recognize the trade-offs between semantic vs fixed-size chunking
5. Appreciate how chunk format impacts embedding quality

USAGE:

    from parser import parse_file
    from chunker import chunk_function, chunk_functions

    # Single function
    functions = parse_file("models.py")
    chunk = chunk_function(functions[0])
    print(f"Chunk ID: {chunk.id}")
    print(f"Tokens: {chunk.token_count}")
    print(f"Truncated: {chunk.truncated}")

    # Batch processing
    chunks = chunk_functions(functions)
    print(f"Created {len(chunks)} chunks")

NEXT STEPS:
After chunking, you'll pass chunks to the embedder (Component 3), which calls
the OpenAI API to generate vector representations for semantic search.
"""

from .models import Chunk
from .chunker import chunk_function, chunk_functions, get_chunking_stats
from .formatter import format_function_for_embedding
from .token_counter import count_tokens, is_within_limit, get_token_stats
from .hasher import generate_chunk_id

__all__ = [
    'Chunk',
    'chunk_function',
    'chunk_functions',
    'get_chunking_stats',
    'format_function_for_embedding',
    'count_tokens',
    'is_within_limit',
    'get_token_stats',
    'generate_chunk_id'
]
