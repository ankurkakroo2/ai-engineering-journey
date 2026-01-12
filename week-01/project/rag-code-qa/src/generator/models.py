"""
Generator Models - Data Structures for Answer Generation

LEARNING CONTEXT:
This is Component 6 of your RAG-Code-QA pipeline (Week 1, Day 8-9).
After Claude generates an answer, you need a data structure to represent
the result with metadata about tokens, timing, and citations.

PURPOSE:
Defines the GenerationResult dataclass, which represents a generated answer
with all associated metadata. This enables tracking, debugging, and validation
of the generation process.

KEY CONCEPTS:
- GenerationResult stores answer + metadata
- Token usage tracking for cost monitoring
- Citation detection for quality assurance
- Generation timing for performance monitoring

DEPENDENCIES:
- Input: Answer text from Claude API
- Input: RetrievalResult list (context used)
- Output: GenerationResult consumed by CLI/testing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..retriever.models import RetrievalResult


@dataclass
class GenerationResult:
    """
    Represents a generated answer with metadata.

    This is the primary data structure for generation results. It combines
    the answer text, the query, the context used, and metadata about the
    generation process (tokens, timing, citations).

    Attributes:
        query: Original user question
        answer: Generated answer from Claude
        context_chunks: Retrieved code chunks used as context
        model: Claude model used (e.g., "claude-sonnet-4-20250514")
        tokens_used: Token counts (prompt, completion, total)
        generation_time: Time taken to generate (seconds)
        has_citations: Whether answer includes file:line citations
    """
    query: str
    answer: str
    context_chunks: List[RetrievalResult]
    model: str
    tokens_used: Dict[str, int]
    generation_time: float
    has_citations: bool = field(default=False)

    @property
    def sources_cited(self) -> List[str]:
        """
        Extract file:line citations from the answer.

        Looks for patterns like:
        - file.py:10-20
        - path/to/file.py:45-67
        - (file.py:10-20)

        Returns:
            List of unique citations found in the answer

        Example:
            >>> result.answer = "See login_handler() (auth.py:45-67)"
            >>> result.sources_cited
            ['auth.py:45-67']
        """
        # Pattern: filename.ext:start-end or (filename.ext:start-end)
        pattern = r'(?:\()?([a-zA-Z0-9_/\-\.]+\.(?:py|js|ts|tsx|jsx)):\s*(\d+)\s*-\s*(\d+)(?:\))?'
        matches = re.findall(pattern, self.answer)

        # Format as file:start-end
        citations = [f"{file}:{start}-{end}" for file, start, end in matches]

        # Return unique citations (preserve order)
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return unique_citations

    @property
    def answer_length(self) -> int:
        """
        Word count of the answer.

        Returns:
            Number of words in the answer
        """
        return len(self.answer.split())

    @property
    def context_count(self) -> int:
        """Number of context chunks used."""
        return len(self.context_chunks)

    @property
    def prompt_tokens(self) -> int:
        """Tokens used in prompt (context + question)."""
        return self.tokens_used.get('prompt', 0)

    @property
    def completion_tokens(self) -> int:
        """Tokens used in completion (answer)."""
        return self.tokens_used.get('completion', 0)

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.tokens_used.get('total', 0)

    @property
    def tokens_per_second(self) -> float:
        """
        Generation speed in tokens per second.

        Useful for performance monitoring.
        """
        if self.generation_time > 0:
            return self.completion_tokens / self.generation_time
        return 0.0

    @property
    def citation_count(self) -> int:
        """Number of unique citations in the answer."""
        return len(self.sources_cited)

    @property
    def quality_score(self) -> float:
        """
        Simple quality score based on citations and length.

        Heuristic:
        - Has citations: +0.5
        - Answer length 50-500 words: +0.3
        - Multiple citations: +0.2

        Returns:
            Quality score (0.0 - 1.0)
        """
        score = 0.0

        # Has citations
        if self.has_citations:
            score += 0.5

        # Reasonable answer length
        if 50 <= self.answer_length <= 500:
            score += 0.3

        # Multiple citations
        if self.citation_count >= 2:
            score += 0.2

        return min(score, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Useful for logging, debugging, or API responses.
        """
        return {
            'query': self.query,
            'answer': self.answer,
            'context_count': self.context_count,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'generation_time': self.generation_time,
            'has_citations': self.has_citations,
            'sources_cited': self.sources_cited,
            'answer_length': self.answer_length,
            'citation_count': self.citation_count,
            'quality_score': self.quality_score
        }

    def format_for_display(self) -> str:
        """
        Format result for terminal display.

        Creates a human-readable summary of the generation result.
        """
        lines = [
            f"\n{'='*80}",
            f"QUERY: {self.query}",
            f"{'='*80}\n",
            f"ANSWER:\n{self.answer}\n",
            f"{'-'*80}",
            f"METADATA:",
            f"  Model: {self.model}",
            f"  Tokens: {self.total_tokens} (prompt: {self.prompt_tokens}, completion: {self.completion_tokens})",
            f"  Time: {self.generation_time:.2f}s ({self.tokens_per_second:.1f} tokens/s)",
            f"  Context chunks: {self.context_count}",
            f"  Citations: {self.citation_count}",
            f"  Answer length: {self.answer_length} words",
            f"  Quality score: {self.quality_score:.2f}",
        ]

        if self.sources_cited:
            lines.append(f"\nSOURCES CITED:")
            for i, citation in enumerate(self.sources_cited, 1):
                lines.append(f"  [{i}] {citation}")

        lines.append(f"{'='*80}\n")

        return "\n".join(lines)


def check_citations(answer: str) -> bool:
    """
    Check if an answer contains file:line citations.

    Quick check to determine if the answer includes citations
    without extracting them.

    Args:
        answer: Generated answer text

    Returns:
        True if citations found, False otherwise

    Example:
        >>> check_citations("See auth.py:45-67 for details")
        True
        >>> check_citations("The system uses authentication")
        False
    """
    # Pattern: filename.ext:start-end
    pattern = r'[a-zA-Z0-9_/\-\.]+\.(?:py|js|ts|tsx|jsx):\s*\d+\s*-\s*\d+'
    return bool(re.search(pattern, answer))


def extract_code_blocks(answer: str) -> List[str]:
    """
    Extract code blocks from the answer.

    Finds markdown code blocks (```language ... ```) in the answer.
    Useful for analyzing how much code Claude included.

    Args:
        answer: Generated answer text

    Returns:
        List of code block contents

    Example:
        >>> answer = "Here's the code:\n```python\ndef foo():\n    pass\n```"
        >>> extract_code_blocks(answer)
        ['def foo():\n    pass']
    """
    # Pattern: ```language ... ```
    pattern = r'```(?:\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, answer, re.DOTALL)
    return matches


def format_generation_stats(results: List[GenerationResult]) -> str:
    """
    Format statistics for multiple generation results.

    Useful for batch processing or evaluation.

    Args:
        results: List of GenerationResult objects

    Returns:
        Formatted statistics string
    """
    if not results:
        return "No results to display."

    total_tokens = sum(r.total_tokens for r in results)
    total_time = sum(r.generation_time for r in results)
    avg_tokens = total_tokens / len(results)
    avg_time = total_time / len(results)

    with_citations = sum(1 for r in results if r.has_citations)
    citation_rate = (with_citations / len(results)) * 100

    avg_quality = sum(r.quality_score for r in results) / len(results)

    lines = [
        f"\n{'='*80}",
        f"GENERATION STATISTICS ({len(results)} results)",
        f"{'='*80}",
        f"Total tokens: {total_tokens}",
        f"Average tokens per result: {avg_tokens:.1f}",
        f"Total time: {total_time:.2f}s",
        f"Average time per result: {avg_time:.2f}s",
        f"Results with citations: {with_citations}/{len(results)} ({citation_rate:.1f}%)",
        f"Average quality score: {avg_quality:.2f}",
        f"{'='*80}\n"
    ]

    return "\n".join(lines)
