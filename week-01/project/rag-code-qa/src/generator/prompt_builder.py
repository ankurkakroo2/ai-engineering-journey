"""
Prompt Builder - Build Prompts for Claude

LEARNING CONTEXT:
This is Component 6 of your RAG-Code-QA pipeline (Week 1, Day 8-9).
The quality of Claude's answers depends heavily on prompt engineering.
This module builds prompts that emphasize citations and accuracy.

PURPOSE:
Creates system and user prompts for Claude:
- System prompt instructs Claude to cite sources and be accurate
- User prompt combines context + question in a clear format
- Handles empty context case gracefully
- Emphasizes answering only from provided code

KEY CONCEPTS:
- System prompt sets behavior (cite sources, no hallucination)
- User prompt provides context + question
- Clear formatting helps Claude understand structure
- Temperature=0.0 for deterministic, factual responses

PROMPT ENGINEERING PRINCIPLES:
1. Be explicit about rules (cite sources, don't hallucinate)
2. Provide clear structure (Context: ... Question: ...)
3. Use examples in system prompt if needed
4. Keep prompts concise but complete

INTEGRATION POINTS:
- Input: Query string + formatted context
- Output: Messages list for Claude API
"""

import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# System prompt that instructs Claude how to behave
SYSTEM_PROMPT = """You are a code assistant answering questions about a codebase.
You will be given relevant code snippets as context.

IMPORTANT RULES:
1. Answer based ONLY on the provided code snippets
2. If the answer isn't in the context, say so clearly
3. Always cite which file and line numbers your answer comes from
4. Use the format: function_name() (file.py:start-end)
5. Be specific and reference actual code when possible
6. Do not make assumptions or add information not in the code
7. If you're unsure, say so rather than guessing

CITATION FORMAT:
When referencing code, use: function_name() (file.py:line-line)
Example: "The login is handled by login_handler() (auth.py:45-67)"

Your goal is to provide accurate, verifiable answers that help developers understand the codebase."""


def build_system_prompt() -> str:
    """
    Return the system prompt for Claude.

    The system prompt sets Claude's behavior and establishes rules
    for how it should answer questions.

    Returns:
        System prompt string

    Example:
        >>> system_prompt = build_system_prompt()
        >>> print(system_prompt[:50])
        You are a code assistant answering questions...
    """
    return SYSTEM_PROMPT


def build_user_prompt(query: str, context: str) -> str:
    """
    Build user prompt with context and question.

    Combines the formatted context (from context_builder) with the
    user's question in a clear structure.

    Args:
        query: User's question
        context: Formatted context from context_builder

    Returns:
        User prompt string

    Format:
        Context:
        [1] file.py:10-20
        ```python
        code here
        ```

        [2] file.py:30-40
        ```python
        more code
        ```

        Question: {query}

    Example:
        >>> query = "How does authentication work?"
        >>> context = build_context(retrieval_results)
        >>> prompt = build_user_prompt(query, context)
    """
    if not context:
        logger.warning("Building user prompt with empty context")
        return f"Question: {query}\n\nNote: No relevant code found for this question."

    # Build structured prompt
    prompt_parts = [
        "Context:",
        context,
        "",  # Blank line for separation
        f"Question: {query}"
    ]

    return "\n".join(prompt_parts)


def build_messages(query: str, context: str) -> List[Dict[str, str]]:
    """
    Build complete message list for Claude API.

    Creates the messages array in the format Claude expects:
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    Args:
        query: User's question
        context: Formatted context from context_builder

    Returns:
        List of message dictionaries for Claude API

    Example:
        >>> messages = build_messages(query, context)
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     messages=messages
        ... )
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, context)

    # Note: Anthropic API uses system parameter separately,
    # not in messages array. But we return it here for flexibility.
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    logger.info(f"Built messages for query: '{query[:50]}...'")

    return messages


def build_no_context_prompt(query: str) -> str:
    """
    Build prompt when no relevant code found.

    Provides a helpful response when retrieval returns no results.

    Args:
        query: User's question

    Returns:
        Prompt explaining no code was found

    Example:
        >>> prompt = build_no_context_prompt("How does X work?")
        >>> print(prompt)
        I couldn't find any relevant code for your question...
    """
    return f"""I couldn't find any relevant code for your question: "{query}"

This could mean:
1. The code doesn't exist in the indexed codebase
2. The question uses different terminology than the code
3. The similarity threshold filtered out all results

Suggestions:
- Try rephrasing your question with different keywords
- Check if the code has been indexed
- Lower the similarity threshold (min_score) if using the API directly"""


def get_prompt_stats(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Get statistics about the prompt.

    Useful for debugging and monitoring.

    Args:
        messages: Messages list for Claude

    Returns:
        Dictionary with statistics:
        - message_count: Number of messages
        - total_chars: Total characters in all messages
        - user_message_length: Length of user message

    Example:
        >>> messages = build_messages(query, context)
        >>> stats = get_prompt_stats(messages)
        >>> print(f"Prompt has {stats['total_chars']} characters")
    """
    message_count = len(messages)
    total_chars = sum(len(msg['content']) for msg in messages)

    user_messages = [msg for msg in messages if msg['role'] == 'user']
    user_message_length = len(user_messages[0]['content']) if user_messages else 0

    return {
        'message_count': message_count,
        'total_chars': total_chars,
        'user_message_length': user_message_length
    }


def validate_prompt(query: str, context: str) -> tuple[bool, str]:
    """
    Validate prompt inputs before building.

    Checks for common issues that could cause poor results.

    Args:
        query: User's question
        context: Formatted context

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = validate_prompt(query, context)
        >>> if not is_valid:
        ...     print(f"Invalid prompt: {error}")
    """
    # Check query
    if not query or not query.strip():
        return False, "Query is empty"

    if len(query) < 5:
        return False, "Query is too short (< 5 characters)"

    if len(query) > 1000:
        return False, "Query is too long (> 1000 characters)"

    # Context can be empty (handled gracefully), but warn if very long
    if context and len(context) > 100000:
        logger.warning("Context is very long (> 100k characters)")

    return True, ""


def format_prompt_for_display(messages: List[Dict[str, str]], system_prompt: str = None) -> str:
    """
    Format prompt for terminal display.

    Creates a human-readable version of the prompt for debugging.

    Args:
        messages: Messages list for Claude
        system_prompt: Optional system prompt to include

    Returns:
        Formatted prompt string

    Example:
        >>> messages = build_messages(query, context)
        >>> print(format_prompt_for_display(messages, SYSTEM_PROMPT))
    """
    lines = ["\n" + "="*80, "PROMPT", "="*80]

    if system_prompt:
        lines.append("\nSYSTEM:")
        lines.append("-"*80)
        lines.append(system_prompt)

    for i, msg in enumerate(messages, 1):
        lines.append(f"\n{msg['role'].upper()} MESSAGE {i}:")
        lines.append("-"*80)
        # Truncate long messages for display
        content = msg['content']
        if len(content) > 1000:
            content = content[:1000] + "\n... (truncated)"
        lines.append(content)

    lines.append("\n" + "="*80 + "\n")

    return "\n".join(lines)
