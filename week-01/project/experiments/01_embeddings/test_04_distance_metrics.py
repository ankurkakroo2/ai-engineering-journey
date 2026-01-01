"""
Experiment 1, Test 4: Distance Metrics (Cosine vs Euclidean)

THEORY BEING VALIDATED:
    From day1-reading-notes.md, lines 12-15:
    - Euclidean distance: The straight line between two points
    - Cosine distance: The circle and the angle between the two vectors
    - For high-dimensional embeddings, cosine is superior

WHAT THIS TEST DOES:
    1. Embed: short code, verbose equivalent, completely different code
    2. Calculate both Euclidean distance and cosine similarity
    3. Compare which metric better identifies semantic similarity

WHY THIS MATTERS:
    - Euclidean distance affected by vector magnitude/length
    - Cosine similarity measures angle (semantic direction)
    - For code search, semantic similarity > length

EXPECTED RESULT:
    Cosine correctly identifies short/verbose as similar
    Euclidean incorrectly penalizes length differences

CONNECTION TO WEEK 1 PROJECT:
    - ChromaDB uses cosine similarity by default
    - This validates the correct choice for code search
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

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def test_distance_metrics():
    """Test distance metrics (cosine vs euclidean)"""

    print("\n" + "="*70)
    print("TEST 4: Distance Metrics - Cosine vs Euclidean")
    print("="*70)

    # Code examples: short, verbose equivalent, completely different
    code_short = "def sum(arr): return sum(arr)"
    code_long = "def calculate_sum(array):\n    total = 0\n    for item in array:\n        total += item\n    return total"
    code_different = "def fetch_user(id): return db.query(User).get(id)"

    print(f"\nCode Short:      {code_short}")
    print(f"Code Long:       {code_long[:50]}...")
    print(f"Code Different:  {code_different}")

    print("\nGetting embeddings...")

    # TODO: Get embeddings for all three code snippets
    # emb_short = get_embedding(code_short)
    # emb_long = get_embedding(code_long)
    # emb_different = get_embedding(code_different)

    # TODO: Calculate cosine similarities
    # cosine_short_long = cosine_similarity(emb_short, emb_long)
    # cosine_short_diff = cosine_similarity(emb_short, emb_different)

    # TODO: Calculate euclidean distances
    # euclidean_short_long = euclidean_distance(emb_short, emb_long)
    # euclidean_short_diff = euclidean_distance(emb_short, emb_different)

    # TODO: Print cosine results
    # print(f"\nCOSINE SIMILARITY:")
    # print(f"  Short ↔ Long:      {cosine_short_long:.4f}")
    # print(f"  Short ↔ Different: {cosine_short_diff:.4f}")
    # cosine_distinguishes = cosine_short_long > cosine_short_diff

    # TODO: Print euclidean results
    # print(f"\nEUCLIDEAN DISTANCE:")
    # print(f"  Short ↔ Long:      {euclidean_short_long:.4f}")
    # print(f"  Short ↔ Different: {euclidean_short_diff:.4f}")
    # euclidean_distinguishes = euclidean_short_long < euclidean_short_diff

    # TODO: Print analysis
    # print(f"\nCosine correctly identifies short/long as similar: {cosine_distinguishes}")
    # print(f"Euclidean correctly identifies short/long as similar: {euclidean_distinguishes}")

    # TODO: Verify cosine is better
    # cosine_is_better = cosine_distinguishes and not euclidean_distinguishes

    # TODO: Print result
    # print(f"\n{'✅ PASS' if cosine_is_better else '❌ FAIL'}: Cosine is better for semantic similarity: {cosine_is_better}")

    # TODO: Print validation message
    # if cosine_is_better:
    #     print("\n✓ Validated: Cosine similarity > Euclidean for code (Day 1, lines 12-15)")
    #     print("✓ Implication: ChromaDB using cosine is the right choice")
    # else:
    #     print("\n✗ Unexpected: Euclidean performed unexpectedly")

    # TODO: Return boolean result
    # return cosine_is_better

if __name__ == "__main__":
    test_distance_metrics()
