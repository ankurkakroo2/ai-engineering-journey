"""
Generator - Main Orchestrator for Answer Generation

LEARNING CONTEXT:
This is the main entry point for Component 6 of your RAG-Code-QA pipeline.
It orchestrates the complete generation workflow:
1. Build context from retrieval results
2. Build prompt with system and user messages
3. Call Claude API
4. Create GenerationResult with metadata

PURPOSE:
Provides a simple, high-level API for answer generation. This is the
facade that ties together all generator submodules and provides the
interface used by the CLI and testing.

KEY CONCEPTS:
- Facade pattern: Simple API hides complexity
- RAG pattern: Retrieved chunks prevent hallucination
- Citation-based answers: Enable verification
- Token budget management: Balance context and answer

ARCHITECTURE:
This module coordinates:
- context_builder: Format chunks
- prompt_builder: Build prompts
- claude_client: Call API
- models: Create result

INTEGRATION POINTS:
- Input: Query string + List[RetrievalResult]
- Output: GenerationResult with answer and metadata
- Uses: retriever, claude_client modules
"""

import logging
import time
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..retriever.models import RetrievalResult
from ..retriever import retrieve
from .models import GenerationResult, check_citations
from .context_builder import build_context, truncate_results
from .prompt_builder import build_messages, build_system_prompt, build_no_context_prompt
from .claude_client import create_claude_client, call_with_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global statistics for monitoring
_generation_stats = {
    'total_generations': 0,
    'total_tokens': 0,
    'total_time': 0.0,
    'with_citations': 0
}


def generate_answer(
    query: str,
    retrieval_results: List[RetrievalResult],
    max_context_tokens: int = 6000,
    max_answer_tokens: int = 2048
) -> GenerationResult:
    """
    Generate an answer using Claude with retrieved code as context.

    This is the main entry point for answer generation. It handles the
    complete pipeline from retrieval results to generated answer.

    The generation process:
    1. Build context from retrieval results (format + truncate)
    2. Build prompt (system + user messages)
    3. Call Claude API with retry logic
    4. Create GenerationResult with metadata

    Args:
        query: User's question
        retrieval_results: Retrieved code chunks from retriever
        max_context_tokens: Max tokens for context (default: 6000)
        max_answer_tokens: Max tokens for answer (default: 2048)

    Returns:
        GenerationResult with answer and metadata

    Raises:
        ValueError: If query is empty or API call fails

    Example:
        >>> results = retrieve("How does authentication work?", top_k=5)
        >>> answer = generate_answer("How does authentication work?", results)
        >>> print(answer.answer)
        >>> print(f"Sources: {answer.sources_cited}")

    Example with no results:
        >>> results = retrieve("nonexistent", top_k=5)
        >>> answer = generate_answer("nonexistent", results)
        >>> print(answer.answer)  # Helpful message about no code found
    """
    global _generation_stats

    # Validate input
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    logger.info(f"Generating answer for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    logger.info(f"Parameters: max_context_tokens={max_context_tokens}, max_answer_tokens={max_answer_tokens}")

    start_time = time.time()

    try:
        # Handle empty results
        if not retrieval_results:
            logger.warning("No retrieval results provided")
            answer_text = build_no_context_prompt(query)

            # Create result with no context
            result = GenerationResult(
                query=query,
                answer=answer_text,
                context_chunks=[],
                model="none",
                tokens_used={'prompt': 0, 'completion': 0, 'total': 0},
                generation_time=time.time() - start_time,
                has_citations=False
            )

            return result

        # Step 1: Build context
        logger.debug(f"Step 1: Building context from {len(retrieval_results)} results")
        context_start = time.time()

        # Truncate results if needed to fit token budget
        truncated_results = truncate_results(retrieval_results, max_context_tokens)
        context = build_context(truncated_results, max_context_tokens)

        context_time = time.time() - context_start
        logger.debug(f"Context built in {context_time:.3f}s")

        # Step 2: Build prompt
        logger.debug("Step 2: Building prompt")
        prompt_start = time.time()

        system_prompt = build_system_prompt()
        messages = build_messages(query, context)

        prompt_time = time.time() - prompt_start
        logger.debug(f"Prompt built in {prompt_time:.3f}s")

        # Step 3: Call Claude
        logger.debug("Step 3: Calling Claude API")
        api_start = time.time()

        client = create_claude_client()
        response = call_with_retry(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_answer_tokens,
            temperature=0.0
        )

        api_time = time.time() - api_start
        logger.debug(f"Claude API call completed in {api_time:.3f}s")

        # Step 4: Create result
        logger.debug("Step 4: Creating GenerationResult")

        total_time = time.time() - start_time
        has_citations = check_citations(response['answer'])

        result = GenerationResult(
            query=query,
            answer=response['answer'],
            context_chunks=truncated_results,
            model=response['model'],
            tokens_used=response['tokens'],
            generation_time=total_time,
            has_citations=has_citations
        )

        # Update statistics
        _generation_stats['total_generations'] += 1
        _generation_stats['total_tokens'] += result.total_tokens
        _generation_stats['total_time'] += total_time
        if has_citations:
            _generation_stats['with_citations'] += 1

        # Log summary
        logger.info(
            f"Generated answer in {total_time:.2f}s "
            f"(context: {context_time:.2f}s, prompt: {prompt_time:.2f}s, api: {api_time:.2f}s)"
        )
        logger.info(
            f"Tokens: {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})"
        )
        logger.info(f"Citations: {'Yes' if has_citations else 'No'} ({result.citation_count} found)")

        return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


def generate_with_retrieval(
    query: str,
    top_k: int = 5,
    min_score: float = 0.3,
    max_context_tokens: int = 6000,
    max_answer_tokens: int = 2048
) -> GenerationResult:
    """
    Complete RAG pipeline: retrieve + generate.

    Convenience function that calls retriever then generator.
    Useful for one-shot question answering.

    Args:
        query: User's question
        top_k: Number of chunks to retrieve (default: 5)
        min_score: Minimum similarity score (default: 0.3)
        max_context_tokens: Max tokens for context (default: 6000)
        max_answer_tokens: Max tokens for answer (default: 2048)

    Returns:
        GenerationResult with answer and metadata

    Example:
        >>> answer = generate_with_retrieval("How does auth work?")
        >>> print(answer.answer)
        >>> print(f"Used {answer.context_count} code chunks")
    """
    logger.info(f"Running complete RAG pipeline for query: '{query}'")

    # Step 1: Retrieve
    logger.info(f"Retrieving top {top_k} chunks (min_score={min_score})")
    retrieval_results = retrieve(query, top_k=top_k, min_score=min_score)

    logger.info(f"Retrieved {len(retrieval_results)} chunks")

    # Step 2: Generate
    result = generate_answer(
        query=query,
        retrieval_results=retrieval_results,
        max_context_tokens=max_context_tokens,
        max_answer_tokens=max_answer_tokens
    )

    return result


def get_generation_stats() -> dict:
    """
    Get generation statistics.

    Returns statistics about all generation operations since startup.
    Useful for monitoring and debugging.

    Returns:
        Dictionary with statistics:
        - total_generations: Total number of generations
        - total_tokens: Total tokens used
        - total_time: Total time spent
        - avg_tokens_per_generation: Average tokens per generation
        - avg_time_per_generation: Average time per generation
        - citation_rate: Percentage with citations

    Example:
        >>> stats = get_generation_stats()
        >>> print(f"Total generations: {stats['total_generations']}")
        >>> print(f"Citation rate: {stats['citation_rate']:.1f}%")
    """
    total_gens = _generation_stats['total_generations']

    if total_gens == 0:
        return {
            'total_generations': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_tokens_per_generation': 0.0,
            'avg_time_per_generation': 0.0,
            'citation_rate': 0.0
        }

    return {
        'total_generations': total_gens,
        'total_tokens': _generation_stats['total_tokens'],
        'total_time': _generation_stats['total_time'],
        'avg_tokens_per_generation': _generation_stats['total_tokens'] / total_gens,
        'avg_time_per_generation': _generation_stats['total_time'] / total_gens,
        'citation_rate': (_generation_stats['with_citations'] / total_gens) * 100
    }


def reset_generation_stats():
    """
    Reset generation statistics.

    Useful for testing or when starting a new monitoring period.
    """
    global _generation_stats
    _generation_stats = {
        'total_generations': 0,
        'total_tokens': 0,
        'total_time': 0.0,
        'with_citations': 0
    }
    logger.info("Generation statistics reset")
