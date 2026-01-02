"""
Chunker - Main Orchestrator for ParsedFunction → Chunk Transformation

LEARNING CONTEXT:
This is the main entry point for Component 2 of your RAG pipeline.
It orchestrates the three transformations that convert parsed functions
into embeddable chunks:
1. Format content (formatter.py)
2. Count & validate tokens (token_counter.py)
3. Generate deterministic ID (hasher.py)

PURPOSE:
Coordinates the chunking pipeline and enforces the core invariant:
**One function = One chunk** (semantic chunking strategy)

This ensures:
- Complete semantic units (no broken context)
- Precise retrieval (query matches whole function, not fragment)
- Clean boundaries (no arbitrary cutoffs mid-thought)

PIPELINE FLOW:
ParsedFunction → format → count tokens → truncate if needed → hash → Chunk

ERROR HANDLING:
- Graceful degradation: Log warnings, continue processing
- Invalid chunks (empty content, zero tokens) are filtered out
- Truncation is tracked for quality monitoring

INTEGRATION POINTS:
- Input: ParsedFunction from parser module
- Output: Chunk objects for embedder module
- The embedder will use chunk.content for API calls
- The storage will use chunk.id for database keys
- The retriever will use chunk.metadata for citations

DEPENDENCIES:
- formatter: Content serialization
- token_counter: Token validation
- hasher: ID generation
- models: Chunk dataclass
"""

import logging
from typing import List, Optional

from ..parser.models import ParsedFunction
from .models import Chunk
from .formatter import format_function_for_embedding, truncate_content
from .token_counter import count_tokens, is_within_limit
from .hasher import generate_chunk_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_function(func: ParsedFunction, max_tokens: int = 8000) -> Optional[Chunk]:
    try:
        content = format_function_for_embedding(func)

        token_count = count_tokens(content)
        truncated = False

        if token_count > max_tokens:
            logger.warning(
                f"Function {func.name} at {func.location} exceeds {max_tokens} tokens "
                f"({token_count} tokens). Truncating..."
            )
            content, truncated = truncate_content(
                content, max_tokens, token_count, count_tokens
            )
            token_count = count_tokens(content)

        chunk_id = generate_chunk_id(content, func.file_path, func.start_line)

        metadata = {
            'name': func.name,
            'file_path': func.file_path,
            'start_line': func.start_line,
            'end_line': func.end_line,
            'language': func.language,
            'docstring': func.docstring,
            'has_docstring': func.has_docstring
        }

        chunk = Chunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            token_count=token_count,
            truncated=truncated
        )

        if not chunk.is_valid:
            logger.warning(f"Invalid chunk generated for {func.name} at {func.location}")
            return None

        return chunk

    except Exception as e:
        logger.error(f"Error chunking function {func.name} at {func.location}: {e}")
        return None


def chunk_functions(functions: List[ParsedFunction], max_tokens: int = 8000) -> List[Chunk]:
    chunks = []

    for func in functions:
        chunk = chunk_function(func, max_tokens)
        if chunk:
            chunks.append(chunk)

    logger.info(f"Successfully chunked {len(chunks)}/{len(functions)} functions")

    truncated_count = sum(1 for c in chunks if c.truncated)
    if truncated_count > 0:
        logger.warning(f"{truncated_count} chunks were truncated")

    return chunks


def get_chunking_stats(chunks: List[Chunk]) -> dict:
    if not chunks:
        return {
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens': 0,
            'max_tokens': 0,
            'min_tokens': 0,
            'truncated_count': 0,
            'with_docstring': 0
        }

    token_counts = [c.token_count for c in chunks]

    return {
        'total_chunks': len(chunks),
        'total_tokens': sum(token_counts),
        'avg_tokens': round(sum(token_counts) / len(token_counts), 2),
        'max_tokens': max(token_counts),
        'min_tokens': min(token_counts),
        'truncated_count': sum(1 for c in chunks if c.truncated),
        'with_docstring': sum(1 for c in chunks if c.has_docstring)
    }
