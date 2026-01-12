"""
Claude Client - Anthropic API Integration

LEARNING CONTEXT:
This is Component 6 of your RAG-Code-QA pipeline (Week 1, Day 8-9).
This module handles all interactions with the Anthropic Claude API,
including client initialization, API calls, error handling, and retries.

PURPOSE:
Manages Claude API integration:
- Initialize Anthropic client from environment variable
- Call Claude with messages and parameters
- Handle rate limiting with exponential backoff
- Track token usage for cost monitoring
- Validate responses

KEY CONCEPTS:
- API key from environment variable (ANTHROPIC_API_KEY)
- Temperature=0.0 for deterministic responses
- Exponential backoff for rate limiting
- Token usage tracking
- Error handling and validation

INTEGRATION POINTS:
- Input: Messages list from prompt_builder
- Uses: Anthropic Python SDK
- Output: Response dict with answer and metadata
"""

import logging
import time
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from anthropic import Anthropic, APIError, RateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    APIError = Exception
    RateLimitError = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_claude_client() -> Anthropic:
    """
    Create Anthropic client from environment variable.

    Requires ANTHROPIC_API_KEY in .env file or environment.

    Returns:
        Anthropic client instance

    Raises:
        ValueError: If API key not found or anthropic not installed

    Example:
        >>> client = create_claude_client()
        >>> # Use client to call Claude API
    """
    if not ANTHROPIC_AVAILABLE:
        raise ValueError(
            "anthropic package not installed. "
            "Install with: pip install anthropic>=0.8.0"
        )

    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please set it in your .env file or export it in your shell.\n"
            "Get your API key from: https://console.anthropic.com/"
        )

    logger.info("Creating Anthropic client")
    client = Anthropic(api_key=api_key)

    return client


def call_claude_api(
    client: Anthropic,
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2048,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Call Claude API with messages.

    Args:
        client: Anthropic client instance
        messages: List of message dicts [{"role": "user", "content": "..."}]
        system_prompt: System prompt string
        model: Claude model to use (default: claude-sonnet-4-20250514)
        max_tokens: Maximum tokens for completion (default: 2048)
        temperature: Sampling temperature (default: 0.0 for deterministic)

    Returns:
        Dictionary with:
        - answer: Generated answer text
        - tokens: Dict with prompt, completion, total token counts
        - model: Model used
        - stop_reason: Why generation stopped

    Raises:
        APIError: If API call fails

    Example:
        >>> client = create_claude_client()
        >>> messages = build_messages(query, context)
        >>> response = call_claude_api(client, messages, SYSTEM_PROMPT)
        >>> print(response['answer'])
    """
    logger.info(f"Calling Claude API (model: {model}, max_tokens: {max_tokens})")

    start_time = time.time()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )

        elapsed = time.time() - start_time

        # Extract answer text
        answer = response.content[0].text

        # Extract token usage
        tokens = {
            'prompt': response.usage.input_tokens,
            'completion': response.usage.output_tokens,
            'total': response.usage.input_tokens + response.usage.output_tokens
        }

        logger.info(
            f"Claude API call successful in {elapsed:.2f}s "
            f"(tokens: {tokens['total']})"
        )

        return {
            'answer': answer,
            'tokens': tokens,
            'model': response.model,
            'stop_reason': response.stop_reason
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Claude API call failed after {elapsed:.2f}s: {e}")
        raise


def call_with_retry(
    client: Anthropic,
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Call Claude API with exponential backoff retry.

    Retries on rate limit errors with exponential backoff.
    Other errors are raised immediately.

    Args:
        client: Anthropic client
        messages: Messages list
        system_prompt: System prompt
        model: Claude model
        max_tokens: Max completion tokens
        temperature: Sampling temperature
        max_retries: Maximum retry attempts (default: 3)

    Returns:
        Response dictionary (same as call_claude_api)

    Raises:
        APIError: If all retries fail

    Example:
        >>> response = call_with_retry(client, messages, SYSTEM_PROMPT)
    """
    for attempt in range(max_retries):
        try:
            return call_claude_api(
                client, messages, system_prompt,
                model, max_tokens, temperature
            )

        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limit hit, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise

        except APIError as e:
            # Other API errors - don't retry
            logger.error(f"API error (not retrying): {e}")
            raise

    # Should never reach here, but just in case
    raise APIError("Max retries exceeded")


def validate_response(response: Dict[str, Any]) -> bool:
    """
    Validate Claude API response.

    Checks that response has required fields and valid values.

    Args:
        response: Response dict from call_claude_api

    Returns:
        True if valid, False otherwise

    Example:
        >>> response = call_claude_api(client, messages, SYSTEM_PROMPT)
        >>> if validate_response(response):
        ...     print("Valid response")
    """
    required_fields = ['answer', 'tokens', 'model', 'stop_reason']

    # Check all required fields present
    for field in required_fields:
        if field not in response:
            logger.error(f"Response missing required field: {field}")
            return False

    # Check answer is non-empty
    if not response['answer'] or not response['answer'].strip():
        logger.error("Response has empty answer")
        return False

    # Check tokens are positive
    if response['tokens']['total'] <= 0:
        logger.error("Response has invalid token count")
        return False

    return True


def estimate_cost(tokens: Dict[str, int], model: str = "claude-sonnet-4-20250514") -> float:
    """
    Estimate cost of API call based on token usage.

    Pricing (as of 2025):
    - Claude Sonnet 4: $3/MTok input, $15/MTok output

    Args:
        tokens: Token usage dict (prompt, completion, total)
        model: Model used

    Returns:
        Estimated cost in USD

    Example:
        >>> tokens = {'prompt': 1000, 'completion': 500, 'total': 1500}
        >>> cost = estimate_cost(tokens)
        >>> print(f"Cost: ${cost:.4f}")
    """
    # Pricing per million tokens
    if "sonnet-4" in model.lower():
        input_price = 3.0  # $3 per MTok
        output_price = 15.0  # $15 per MTok
    else:
        # Default pricing
        input_price = 3.0
        output_price = 15.0

    input_cost = (tokens['prompt'] / 1_000_000) * input_price
    output_cost = (tokens['completion'] / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return total_cost


def get_client_stats() -> Dict[str, Any]:
    """
    Get statistics about Claude API usage.

    Returns basic info about the client configuration.

    Returns:
        Dictionary with client statistics

    Example:
        >>> stats = get_client_stats()
        >>> print(f"API available: {stats['api_available']}")
    """
    return {
        'api_available': ANTHROPIC_AVAILABLE,
        'api_key_set': bool(os.getenv('ANTHROPIC_API_KEY')),
        'default_model': "claude-sonnet-4-20250514",
        'default_max_tokens': 2048,
        'default_temperature': 0.0
    }
