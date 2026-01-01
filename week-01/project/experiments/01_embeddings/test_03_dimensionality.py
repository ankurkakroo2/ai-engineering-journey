"""
Experiment 1, Test 3: Dimensionality (1536 vs 384)

THEORY BEING VALIDATED:
    From day1-reading-notes.md, lines 69-87:
    - Fewer dimensions (e.g., 384) = Less nuance, but faster computation
    - More dimensions (e.g., 1536) = More nuance, more accurate, but slower

WHAT THIS TEST DOES:
    1. Get 1536-dim embedding for complex code
    2. Truncate to 384 dims (simulate lower-dim model)
    3. Compare similarity scores with similar code using both

WHY THIS MATTERS:
    - Shows trade-off between accuracy and performance
    - Validates choice to use 1536-dim embeddings for code search
    - Larger dimension space captures finer semantic nuance

EXPECTED RESULT:
    similarity_full (1536 dims) >= similarity_truncated (384 dims)
    Full dimensionality preserves more semantic detail

CONNECTION TO WEEK 1 PROJECT:
    - Using text-embedding-3-small (1536 dims) gives best semantic search
    - ChromaDB will store all 1536 dimensions
    - Worth the storage cost for code search accuracy
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


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def test_dimensionality():
    """Test dimensionality trade-offs (1536 vs 384)"""

    print("\n" + "=" * 70)
    print("TEST 3: Dimensionality - 1536 vs 384 Dimensions")
    print("=" * 70)

    # Complex code examples
    code_1 = (
        "A complex recursive function implementing depth-first search with memoization"
    )
    code_2 = "A recursive algorithm for DFS with caching"

    print(f"\nCode 1: {code_1}")
    print(f"Code 2: {code_2}")

    print("\nGetting embeddings...")

    # TODO: Get embedding for code_1
    emb_1_full = get_embedding(code_1)

    # TODO: Verify it's 1536 dimensions
    print(f"\nFull embedding dimensions: {len(emb_1_full)}")
    assert len(emb_1_full) == 1536, "Expected 1536 dimensions"

    # TODO: Truncate to 384 dimensions
    emb_1_truncated = emb_1_full[:384]

    # TODO: Get embedding for code_2
    emb_2_full = get_embedding(code_2)
    emb_2_truncated = emb_2_full[:384]

    # TODO: Calculate similarities with both full and truncated
    similarity_full = cosine_similarity(emb_1_full, emb_2_full)
    similarity_truncated = cosine_similarity(emb_1_truncated, emb_2_truncated)

    # TODO: Print results
    print(f"\nSimilarity (1536 dims):  {similarity_full:.4f}")
    print(f"Similarity (384 dims):   {similarity_truncated:.4f}")
    print(f"Difference:              {similarity_full - similarity_truncated:.4f}")

    # TODO: Verify full dimensions preserve more nuance
    full_preserves_more = similarity_full >= similarity_truncated

    # TODO: Print result
    print(
        f"\n{'✅ PASS' if full_preserves_more else '❌ FAIL'}: Full dimensions capture more nuance: {full_preserves_more}"
    )

    # TODO: Print validation message
    if full_preserves_more:
        print(
            "\n✓ Validated: 1536 dims capture finer semantic detail (Day 1, lines 69-87)"
        )
        print(
            "✓ Implication: Worth using text-embedding-3-small (1536 dims) for code search"
        )
    else:
        print("\n✗ Unexpected: Truncation didn't reduce similarity as much as expected")

    # TODO: Return boolean result
    return full_preserves_more


if __name__ == "__main__":
    test_dimensionality()
