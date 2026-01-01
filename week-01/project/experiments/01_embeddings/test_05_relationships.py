"""
Experiment 1, Test 5: Semantic Relationships (Language Paradigms)

THEORY BEING VALIDATED:
    From day1-reading-notes.md, lines 227-234:
    - Embeddings capture semantic relationships as mathematical dimensions
    - Like king - man + woman ≈ queen
    - Network learned relationships as measurable dimensions

WHAT THIS TEST DOES:
    1. Embed Python concepts (list comp, dict comp, generator)
    2. Embed JavaScript concepts (map, filter, reduce)
    3. Calculate within-language vs cross-language similarity

WHY THIS MATTERS:
    - Shows embeddings learn domain-specific paradigms
    - Within-language concepts cluster stronger than cross-language
    - Validates multi-language code search capabilities

EXPECTED RESULT:
    avg_py_similarity > avg_cross_similarity
    avg_js_similarity > avg_cross_similarity
    Languages cluster by programming paradigm

CONNECTION TO WEEK 1 PROJECT:
    - Can search Python/JavaScript/TypeScript together
    - Multi-language semantic search will preserve language distinctions
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

def average_pairwise_cosine(embeddings):
    """Calculate average pairwise cosine similarity for a list of embeddings"""
    if len(embeddings) < 2:
        return 0
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
    return np.mean(similarities)

def average_cosine_between_groups(group1, group2):
    """Calculate average cosine similarity between two groups"""
    similarities = []
    for emb1 in group1:
        for emb2 in group2:
            similarities.append(cosine_similarity(emb1, emb2))
    return np.mean(similarities)

def test_semantic_relationships():
    """Test semantic relationships (language paradigms)"""

    print("\n" + "="*70)
    print("TEST 5: Semantic Relationships (Language Paradigms)")
    print("="*70)

    # Python concepts
    python_concepts = [
        "list comprehension [x for x in items]",
        "dictionary comprehension {k: v for k, v in items}",
        "generator expression (x for x in items)"
    ]

    # JavaScript concepts
    javascript_concepts = [
        "array map items.map(x => x)",
        "array filter items.filter(x => x > 0)",
        "array reduce items.reduce((a, b) => a + b)"
    ]

    print(f"\nPython concepts:       {python_concepts}")
    print(f"JavaScript concepts:   {javascript_concepts}")

    print("\nGetting embeddings...")

    # TODO: Get embeddings for all Python concepts
    # py_embs = [get_embedding(c) for c in python_concepts]

    # TODO: Get embeddings for all JavaScript concepts
    # js_embs = [get_embedding(c) for c in javascript_concepts]

    # TODO: Calculate within-language similarities
    # avg_py_similarity = average_pairwise_cosine(py_embs)
    # avg_js_similarity = average_pairwise_cosine(js_embs)

    # TODO: Calculate cross-language similarity
    # avg_cross_similarity = average_cosine_between_groups(py_embs, js_embs)

    # TODO: Print results
    # print(f"\nWithin-language similarities:")
    # print(f"  Python concepts:      {avg_py_similarity:.4f}")
    # print(f"  JavaScript concepts:  {avg_js_similarity:.4f}")
    # print(f"\nCross-language similarity:  {avg_cross_similarity:.4f}")

    # TODO: Verify clustering
    # py_clusters = avg_py_similarity > avg_cross_similarity
    # js_clusters = avg_js_similarity > avg_cross_similarity
    # both_cluster = py_clusters and js_clusters

    # TODO: Print result
    # print(f"\n{'✅ PASS' if both_cluster else '❌ FAIL'}: Languages cluster by paradigm: {both_cluster}")
    # print(f"  Python: {avg_py_similarity:.4f} > {avg_cross_similarity:.4f} = {py_clusters}")
    # print(f"  JavaScript: {avg_js_similarity:.4f} > {avg_cross_similarity:.4f} = {js_clusters}")

    # TODO: Print validation message
    # if both_cluster:
    #     print("\n✓ Validated: Embeddings capture semantic relationships (Day 1, lines 227-234)")
    #     print("✓ Implication: Multi-language code search preserves paradigm distinctions")
    # else:
    #     print("\n✗ Unexpected: Cross-language similarity too high")

    # TODO: Return boolean result
    # return both_cluster

if __name__ == "__main__":
    test_semantic_relationships()
