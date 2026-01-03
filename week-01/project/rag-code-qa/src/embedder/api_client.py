"""
OpenAI API Client - Embedding Generation via OpenAI API

LEARNING CONTEXT:
The OpenAI embedding model (text-embedding-3-small) is a pre-trained,
frozen transformer that maps text to 1536-dimensional vectors.
This module handles API communication with error handling and rate limiting.

PURPOSE:
Provides a clean interface to the OpenAI Embeddings API with:
- Single and batch embedding generation
- Exponential backoff for rate limiting
- Error handling for network and API errors
- Logging for debugging

KEY CONCEPTS:
- Model: text-embedding-3-small (1536 dimensions)
- Batch support: Up to 2048 inputs per request
- Rate limiting: Exponential backoff with max retries
- API key: Loaded from OPENAI_API_KEY environment variable

WHY BATCHING MATTERS:
Individual API calls have overhead (HTTP request, network latency).
Batching 100 chunks into 1 request is ~10x faster than 100 individual requests.

DEPENDENCIES:
- OpenAI Python SDK (openai>=1.0.0)
- Environment: OPENAI_API_KEY must be set
"""

import os
import time
import logging
from typing import List, Optional
from openai import OpenAI, OpenAIError, RateLimitError

logger = logging.getLogger(__name__)

MODEL_NAME = "text-embedding-3-small"
EXPECTED_DIMENSION = 1536
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0


def create_openai_client() -> OpenAI:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it to your OpenAI API key."
        )
    return OpenAI(api_key=api_key)


def call_embedding_api(
    client: OpenAI,
    text: str,
    retry_count: int = 0
) -> Optional[List[float]]:
    try:
        response = client.embeddings.create(
            input=text,
            model=MODEL_NAME
        )

        embedding = response.data[0].embedding

        if len(embedding) != EXPECTED_DIMENSION:
            logger.error(
                f"Unexpected embedding dimension: {len(embedding)} "
                f"(expected {EXPECTED_DIMENSION})"
            )
            return None

        return embedding

    except RateLimitError as e:
        if retry_count < MAX_RETRIES:
            delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
            logger.warning(
                f"Rate limit hit, retrying in {delay}s "
                f"(attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
            return call_embedding_api(client, text, retry_count + 1)
        else:
            logger.error(f"Max retries exceeded for rate limiting: {e}")
            return None

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error calling embedding API: {e}")
        return None


def call_embedding_api_batch(
    client: OpenAI,
    texts: List[str],
    retry_count: int = 0
) -> Optional[List[List[float]]]:
    try:
        if len(texts) > 2048:
            logger.warning(
                f"Batch size {len(texts)} exceeds API limit of 2048. "
                "Truncating batch."
            )
            texts = texts[:2048]

        response = client.embeddings.create(
            input=texts,
            model=MODEL_NAME
        )

        embeddings = [item.embedding for item in response.data]

        for i, embedding in enumerate(embeddings):
            if len(embedding) != EXPECTED_DIMENSION:
                logger.error(
                    f"Unexpected embedding dimension at index {i}: "
                    f"{len(embedding)} (expected {EXPECTED_DIMENSION})"
                )
                return None

        return embeddings

    except RateLimitError as e:
        if retry_count < MAX_RETRIES:
            delay = INITIAL_RETRY_DELAY * (2 ** retry_count)
            logger.warning(
                f"Rate limit hit, retrying in {delay}s "
                f"(attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(delay)
            return call_embedding_api_batch(client, texts, retry_count + 1)
        else:
            logger.error(f"Max retries exceeded for rate limiting: {e}")
            return None

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error calling embedding API (batch): {e}")
        return None
