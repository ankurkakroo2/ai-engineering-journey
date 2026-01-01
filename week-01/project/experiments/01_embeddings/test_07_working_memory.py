"""
Experiment 1, Test 7: Working Memory (RAG Retrieval Quality)

THEORY BEING VALIDATED:
    From day2-3-deep-dive-notes.md, lines 691-704, 611-615:
    - Working Memory: What's in the current prompt (reliable, specific)
    - Knowledge: What's in parameters (vague, compressed)
    - Retrieved chunks = working memory for LLM
    - Quality retrieval prevents hallucinations

WHAT THIS TEST DOES:
    1. Create relevant chunks (calculate_tax, apply_discount)
    2. Create irrelevant chunks (connect_database, log_error)
    3. Query: "How do I calculate price after tax and discount?"
    4. Verify top-k retrieval returns relevant chunks

WHY THIS MATTERS:
    - Validates working memory pattern in RAG
    - Shows quality retrieval provides good context
    - Foundation for preventing LLM hallucinations

EXPECTED RESULT:
    max(relevant_scores) > max(irrelevant_scores)
    top-k results are the relevant chunks
    Quality working memory for LLM response

CONNECTION TO WEEK 1 PROJECT:
    - Core RAG pattern: retrieve → provide as context → LLM answers
    - This validates retriever.py output quality
    - Foundation for generator.py success (better context = better answers)
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

def test_working_memory():
    """Test working memory validation (RAG retrieval quality)"""

    print("\n" + "="*70)
    print("TEST 7: Working Memory Validation (RAG Retrieval Quality)")
    print("="*70)

    # Relevant chunks
    relevant_chunks = [
        "def calculate_tax(amount, rate): return amount * rate",
        "def apply_discount(price, discount_pct): return price * (1 - discount_pct)",
    ]

    # Irrelevant chunks
    irrelevant_chunks = [
        "def connect_database(host, port): return DatabaseConnection(host, port)",
        "def log_error(message): logger.error(message)",
    ]

    # Query
    query = "How do I calculate price after tax and discount?"

    print(f"\nQuery: '{query}'")
    print(f"Relevant chunks: {len(relevant_chunks)}")
    print(f"Irrelevant chunks: {len(irrelevant_chunks)}")

    print("\nGetting embeddings...")

    # TODO: Get query embedding
    # query_emb = get_embedding(query)

    # TODO: Get embeddings for relevant chunks
    # relevant_embs = [get_embedding(c) for c in relevant_chunks]

    # TODO: Get embeddings for irrelevant chunks
    # irrelevant_embs = [get_embedding(c) for c in irrelevant_chunks]

    # TODO: Calculate retrieval scores
    # relevant_scores = [cosine_similarity(query_emb, e) for e in relevant_embs]
    # irrelevant_scores = [cosine_similarity(query_emb, e) for e in irrelevant_embs]

    # TODO: Print scores
    # print(f"\nRelevant chunk scores:   {relevant_scores}")
    # print(f"Irrelevant chunk scores: {irrelevant_scores}")

    # TODO: Verify relevant > irrelevant
    # relevant_better = max(relevant_scores) > max(irrelevant_scores)

    # TODO: Test top-k retrieval
    # top_k = 2
    # all_chunks = relevant_chunks + irrelevant_chunks
    # all_scores = relevant_scores + irrelevant_scores

    # TODO: Sort by score and get top-k
    # sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    # top_k_chunks = [all_chunks[i] for i in sorted_indices[:top_k]]

    # TODO: Verify top-k are relevant
    # top_k_are_relevant = all(chunk in relevant_chunks for chunk in top_k_chunks)

    # TODO: Print top-k results
    # print(f"\nTop-k ({top_k}) retrieval results:")
    # for i, chunk in enumerate(top_k_chunks, 1):
    #     print(f"  {i}. {chunk[:40]}... (relevant: {chunk in relevant_chunks})")

    # TODO: Verify both criteria
    # working_memory_valid = relevant_better and top_k_are_relevant

    # TODO: Print result
    # print(f"\n{'✅ PASS' if working_memory_valid else '❌ FAIL'}: Working memory valid: {working_memory_valid}")

    # TODO: Print validation message
    # if working_memory_valid:
    #     print("\n✓ Validated: Quality retrieval = good working memory (Day 2-3, lines 691-704, 611-615)")
    #     print("✓ Implication: Foundation of RAG architecture; prevents LLM hallucinations")
    # else:
    #     print("\n✗ Unexpected: Retrieval didn't distinguish relevant from irrelevant well")

    # TODO: Return boolean result
    # return working_memory_valid

if __name__ == "__main__":
    test_working_memory()
