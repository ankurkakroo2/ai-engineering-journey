"""
Experiment 1, Test 2: Semantic Clustering (Latent Dimensions)

THEORY BEING VALIDATED:
    From day1-reading-notes.md, lines 36-65:
    - Latent dimensions compress similar concepts together
    - "Catness" and "Dogness" emerge from training
    - Similar semantic meaning → similar embedding patterns

WHAT THIS TEST DOES:
    1. Embed Python add function, JavaScript add function, Python multiply
    2. Calculate pairwise cosine similarities
    3. Verify: add functions cluster together despite language difference

WHY THIS MATTERS:
    - Validates that semantic meaning > syntax
    - Shows embeddings learn "what code does" not just "how it looks"
    - Foundation for multi-language code search

EXPECTED RESULT:
    similarity(python_add, javascript_add) > similarity(python_add, python_multiply)
    Language syntax matters less than semantic function

CONNECTION TO WEEK 1 PROJECT:
    - Can index Python/JavaScript/TypeScript together
    - Queries will find semantically similar code across languages
    - config.yaml: languages: [python, javascript, typescript]
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


def test_semantic_clustering():
    """Test that similar code clusters together (latent dimensions)"""

    print("\n" + "=" * 70)
    print("TEST 2: Latent Dimensions Capture Semantic Meaning")
    print("=" * 70)

    # Code examples
    code_1 = "def add(a, b): return a + b"
    code_2 = "function add(a, b) { return a + b; }"
    code_3 = "def multiply(a, b): return a * b"

    print(f"\nCode 1 (Python add):      {code_1}")
    print(f"Code 2 (JavaScript add):  {code_2}")
    print(f"Code 3 (Python multiply): {code_3}")

    print("\nGetting embeddings...")

    # TODO: Get embeddings for all three code snippets
    emb_1 = get_embedding(code_1)
    emb_2 = get_embedding(code_2)
    emb_3 = get_embedding(code_3)

    # TODO: Calculate cosine similarities
    sim_py_js_add = cosine_similarity(emb_1, emb_2)
    sim_py_add_multiply = cosine_similarity(emb_1, emb_3)

    # TODO: Print similarities
    print(f"\nSimilarity (Python add ↔ JavaScript add):  {sim_py_js_add:.4f}")
    print(f"Similarity (Python add ↔ Python multiply): {sim_py_add_multiply:.4f}")

    # TODO: Verify clustering
    semantic_clustering_works = sim_py_js_add > sim_py_add_multiply

    # TODO: Print result
    print(
        f"\n{'✅ PASS' if semantic_clustering_works else '❌ FAIL'}: Add functions cluster together: {semantic_clustering_works}"
    )

    # TODO: Print validation message
    if semantic_clustering_works:
        print(
            "\n✓ Validated: Latent dimensions capture semantic meaning (Day 1, lines 36-65)"
        )
        print("✓ Implication: Multi-language code search will work in rag-code-qa")
    else:
        print("\n✗ Unexpected: Semantic clustering not as strong as predicted")

    # TODO: Return boolean result
    return semantic_clustering_works


if __name__ == "__main__":
    test_semantic_clustering()
