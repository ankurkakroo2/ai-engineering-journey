"""
Generator Module - Component 6 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is Component 6 of your RAG-Code-QA pipeline (Week 1, Day 8-9).
After retrieving relevant code chunks, you now use Claude to generate natural
language answers grounded in actual code. This is where the RAG pattern comes
together to prevent hallucination.

PURPOSE:
Generates natural language answers using Claude API with retrieved code as context:
- Formats retrieved chunks into Claude-friendly context
- Builds prompts that emphasize citations and accuracy
- Calls Claude API with context + question
- Returns answers with file:line citations

KEY CONCEPTS:
- RAG prevents hallucination by providing actual code as context
- Retrieved chunks act as "working memory" for Claude
- Citation-based answers enable verification
- Token budget management balances context and answer length

ARCHITECTURE:
This module follows the facade pattern:
- generator.py: Main orchestrator (public API)
- context_builder.py: Format chunks for Claude
- prompt_builder.py: Build system and user prompts
- claude_client.py: Anthropic API integration
- models.py: GenerationResult dataclass

INTEGRATION POINTS:
- Input: List[RetrievalResult] from retriever module
- Uses: Anthropic Claude API for answer generation
- Output: GenerationResult with answer and metadata

THE RAG PATTERN:
Without RAG: LLM hallucinates answers about code it hasn't seen
With RAG: LLM answers based on actual retrieved code chunks
Result: Grounded, verifiable answers with citations

NEXT STEPS:
After generation, the CLI module will provide a command-line interface
for the complete index → query → answer workflow.
"""

from .models import GenerationResult
from .generator import generate_answer, generate_with_retrieval

# Public API
__all__ = [
    'GenerationResult',
    'generate_answer',
    'generate_with_retrieval'
]
