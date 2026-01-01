"""
Experiment 1, Test 1: Pre-trained Models (Deterministic Lookup)

THEORY BEING VALIDATED:
    From day1-reading-notes.md, lines 20-33:
    - Embeddings are NOT computed at runtime
    - They are pre-trained and looked up
    - Same input ALWAYS returns same embedding

WHAT THIS TEST DOES:
    1. Call OpenAI embeddings API twice with identical text
    2. Compare the returned vectors element-by-element
    3. Verify exact match (bitwise identical)

WHY THIS MATTERS:
    - Confirms embeddings can be safely cached
    - No need to "refresh" embeddings for unchanged code
    - Foundation for incremental indexing in rag-code-qa

EXPECTED RESULT:
    embedding_1 == embedding_2 (exact match)
    No variation, no randomness

CONNECTION TO WEEK 1 PROJECT:
    - embedder.py can cache embeddings without worrying about staleness
    - Unchanged code files don't need re-embedding
    - Enables efficient incremental updates
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def test_pretraining():
    """Test that embeddings are deterministic (pre-computed lookups)"""

    print("\n" + "=" * 70)
    print("TEST 1: Pre-trained Models (Not Runtime Computation)")
    print("=" * 70)

    # Test input
    text = "The cat sat on the mat"

    print(f"\nInput text: '{text}'")
    print("Calling OpenAI API twice with same text...")

    # TODO: Get embeddings twice with same text
    embedding_1 = get_embedding(text)
    embedding_2 = get_embedding(text)

    # TODO: Print embedding dimensions
    print(f"\nEmbedding 1 length: {len(embedding_1)} dimensions")
    print(f"Embedding 2 length: {len(embedding_2)} dimensions")

    # TODO: Print first 5 values
    print(f"\nFirst 5 values of embedding_1: {embedding_1[:5]}")
    print(f"First 5 values of embedding_2: {embedding_2[:5]}")

    # TODO: Compare embeddings using np.allclose
    are_identical = np.allclose(embedding_1, embedding_2, rtol=0, atol=0)

    # TODO: Print result with ✅ PASS or ❌ FAIL
    print(
        f"\n{'✅ PASS' if are_identical else '❌ FAIL'}: Embeddings are {'identical' if are_identical else 'different'}"
    )

    # TODO: Print validation message
    if are_identical:
        print("\n✓ Validated: Embeddings are pre-computed lookups (Day 1, lines 20-33)")
        print("✓ Implication: Safe to cache embeddings in rag-code-qa")
    else:
        print("\n✗ Unexpected: Embeddings differ (theory may need revision)")

    # TODO: Return boolean result
    return are_identical


if __name__ == "__main__":
    test_pretraining()
