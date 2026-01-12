"""
Context Builder - Format Retrieved Chunks for Claude

LEARNING CONTEXT:
This is Component 6 of your RAG-Code-QA pipeline (Week 1, Day 8-9).
Before calling Claude, you need to format retrieved code chunks into a
context string that Claude can understand and reference.

PURPOSE:
Transforms retrieval results into Claude-friendly context:
- Formats each chunk with file path and line numbers
- Adds numbered references ([1], [2], etc.) for citations
- Counts tokens to respect context window limits
- Truncates if needed to fit token budget

KEY CONCEPTS:
- Context formatting enables Claude to cite sources
- Numbered references make citations easy
- Token counting prevents context overflow
- Truncation prioritizes highest-scored chunks

INTEGRATION POINTS:
- Input: List[RetrievalResult] from retriever
- Uses: tiktoken for token counting
- Output: Formatted context string for Claude
"""

import logging
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..retriever.models import RetrievalResult
from ..chunker.token_counter import count_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_chunk(result: RetrievalResult, index: int) -> str:
    """
    Format a single retrieval result as context for Claude.

    Creates a numbered reference with file location and code content.
    The format makes it easy for Claude to cite sources.

    Args:
        result: RetrievalResult from retriever
        index: Reference number (1-based)

    Returns:
        Formatted chunk string

    Example output:
        [1] src/auth/handlers.py:45-67
        ```python
        def login_handler(username, password):
            # Validate credentials
            if not validate_user(username, password):
                return False
            # Create session
            session = create_session(username)
            return session
        ```
    """
    # Extract language from metadata or file extension
    language = result.language

    # Build formatted chunk
    lines = [
        f"[{index}] {result.location}",
        f"```{language}",
        result.content,
        "```"
    ]

    return "\n".join(lines)


def build_context(
    results: List[RetrievalResult],
    max_tokens: int = 6000
) -> str:
    """
    Build complete context from retrieval results.

    Formats all chunks with numbered references, checks token count,
    and truncates if needed to fit within the token budget.

    Args:
        results: List of RetrievalResult objects from retriever
        max_tokens: Maximum tokens for context (default: 6000)

    Returns:
        Formatted context string ready for Claude

    Example:
        >>> results = retrieve("authentication", top_k=3)
        >>> context = build_context(results, max_tokens=6000)
        >>> print(context)
        [1] auth/handlers.py:45-67
        ```python
        def login_handler(...):
            ...
        ```

        [2] auth/tokens.py:23-41
        ```python
        def verify_token(...):
            ...
        ```
    """
    if not results:
        logger.warning("No retrieval results to build context from")
        return ""

    logger.info(f"Building context from {len(results)} retrieval results")

    # Format each chunk
    formatted_chunks = []
    for i, result in enumerate(results, 1):
        formatted = format_chunk(result, i)
        formatted_chunks.append(formatted)

    # Join with double newlines for separation
    context = "\n\n".join(formatted_chunks)

    # Check token count
    token_count = count_tokens(context)
    logger.info(f"Context token count: {token_count}")

    # Truncate if needed
    if token_count > max_tokens:
        logger.warning(
            f"Context exceeds max_tokens ({token_count} > {max_tokens}). "
            "Truncating..."
        )
        context = truncate_context_string(context, max_tokens)
        final_token_count = count_tokens(context)
        logger.info(f"Truncated context token count: {final_token_count}")

    return context


def count_context_tokens(context: str) -> int:
    """
    Count tokens in context string.

    Uses tiktoken with the same encoding as OpenAI/Anthropic models.

    Args:
        context: Formatted context string

    Returns:
        Token count

    Example:
        >>> context = build_context(results)
        >>> tokens = count_context_tokens(context)
        >>> print(f"Context uses {tokens} tokens")
    """
    return count_tokens(context)


def truncate_context_string(context: str, max_tokens: int) -> str:
    """
    Truncate context string to fit within token budget.

    Tries to preserve complete chunks by truncating from the end.
    Adds a note about truncation.

    Args:
        context: Full context string
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated context string

    Note:
        This is a simple truncation strategy. A more sophisticated
        approach would re-rank chunks or truncate individual chunks.
    """
    # Start with full context
    current_context = context

    # Iteratively remove chunks from the end until we fit
    chunks = context.split("\n\n")

    while count_tokens(current_context) > max_tokens and len(chunks) > 1:
        # Remove last chunk
        chunks = chunks[:-1]
        current_context = "\n\n".join(chunks)

    # If still too long, truncate the text directly
    if count_tokens(current_context) > max_tokens:
        # Estimate characters per token (~4)
        max_chars = max_tokens * 4
        current_context = current_context[:max_chars]

        # Add truncation note
        current_context += "\n\n[Context truncated due to token limit]"

    return current_context


def truncate_results(
    results: List[RetrievalResult],
    max_tokens: int
) -> List[RetrievalResult]:
    """
    Truncate retrieval results to fit token budget.

    This is an alternative to truncate_context_string that works
    at the chunk level before formatting. Preserves complete chunks.

    Args:
        results: List of RetrievalResult objects
        max_tokens: Maximum tokens for all chunks

    Returns:
        Truncated list of RetrievalResult objects

    Strategy:
        - Results are already sorted by score (highest first)
        - Keep adding chunks until we exceed max_tokens
        - Return the chunks that fit

    Example:
        >>> results = retrieve("auth", top_k=10)
        >>> truncated = truncate_results(results, max_tokens=6000)
        >>> print(f"Kept {len(truncated)}/{len(results)} chunks")
    """
    if not results:
        return []

    logger.info(f"Truncating {len(results)} results to fit {max_tokens} tokens")

    kept_results = []
    cumulative_tokens = 0

    for result in results:
        # Format this chunk to count its tokens
        formatted = format_chunk(result, len(kept_results) + 1)
        chunk_tokens = count_tokens(formatted)

        # Check if adding this chunk would exceed limit
        if cumulative_tokens + chunk_tokens > max_tokens:
            logger.info(
                f"Stopping at {len(kept_results)} chunks "
                f"({cumulative_tokens} tokens)"
            )
            break

        # Add this chunk
        kept_results.append(result)
        cumulative_tokens += chunk_tokens

    if len(kept_results) < len(results):
        logger.warning(
            f"Truncated from {len(results)} to {len(kept_results)} chunks "
            f"to fit {max_tokens} token budget"
        )

    return kept_results


def estimate_context_tokens(results: List[RetrievalResult]) -> int:
    """
    Estimate total tokens for a list of results without formatting.

    Quick estimation based on content length. Useful for pre-checking
    before formatting.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Estimated token count

    Note:
        This is an approximation. Use count_context_tokens() for
        accurate counts after formatting.
    """
    total_chars = sum(len(r.content) for r in results)
    # Rough estimate: 4 characters per token
    # Add overhead for formatting (~20% more)
    estimated_tokens = int((total_chars / 4) * 1.2)
    return estimated_tokens


def validate_context(context: str, max_tokens: int) -> Tuple[bool, int]:
    """
    Validate that context fits within token budget.

    Args:
        context: Formatted context string
        max_tokens: Maximum tokens allowed

    Returns:
        Tuple of (is_valid, token_count)

    Example:
        >>> context = build_context(results)
        >>> is_valid, tokens = validate_context(context, 6000)
        >>> if not is_valid:
        ...     print(f"Context too long: {tokens} tokens")
    """
    token_count = count_tokens(context)
    is_valid = token_count <= max_tokens

    if not is_valid:
        logger.warning(
            f"Context validation failed: {token_count} > {max_tokens} tokens"
        )

    return is_valid, token_count


def get_context_stats(context: str) -> dict:
    """
    Get statistics about the context.

    Useful for debugging and monitoring.

    Args:
        context: Formatted context string

    Returns:
        Dictionary with statistics:
        - token_count: Number of tokens
        - char_count: Number of characters
        - line_count: Number of lines
        - chunk_count: Number of chunks (numbered references)

    Example:
        >>> context = build_context(results)
        >>> stats = get_context_stats(context)
        >>> print(f"Context has {stats['chunk_count']} chunks")
    """
    token_count = count_tokens(context)
    char_count = len(context)
    line_count = context.count('\n') + 1

    # Count chunks by counting numbered references [1], [2], etc.
    import re
    chunk_matches = re.findall(r'\[\d+\]', context)
    chunk_count = len(chunk_matches)

    return {
        'token_count': token_count,
        'char_count': char_count,
        'line_count': line_count,
        'chunk_count': chunk_count,
        'avg_tokens_per_chunk': token_count / chunk_count if chunk_count > 0 else 0
    }
