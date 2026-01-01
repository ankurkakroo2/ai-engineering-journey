"""
Experiment 1, Test 6: Chunking by Semantic Units (Functions)

THEORY BEING VALIDATED:
    From day2-3-deep-dive-notes.md, lines 666-668, 735-736:
    - Good chunking strategy = good tool output = better model response
    - Your prompt + retrieved code chunks = working memory
    - Semantic units (functions) > arbitrary splits

WHAT THIS TEST DOES:
    1. Create code file with 3 functions
    2. Chunk by: (1) function boundaries, (2) fixed 100-char splits
    3. Query: "authenticate user credentials"
    4. Compare retrieval quality

WHY THIS MATTERS:
    - Validates chunking strategy for rag-code-qa
    - Shows semantic units preserve meaning better
    - Foundation for effective working memory in RAG

EXPECTED RESULT:
    best_semantic_score > best_fixed_score
    Top result is authenticate_user function

CONNECTION TO WEEK 1 PROJECT:
    - Validates config.yaml: chunk_strategy: by_function
    - Informs parser.py and chunker.py design
    - Ensures good working memory for Claude responses
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

def test_chunking_strategies():
    """Test chunking strategies (function vs fixed-size)"""

    print("\n" + "="*70)
    print("TEST 6: Chunking by Semantic Units (Functions)")
    print("="*70)

    # Code file with 3 functions
    code_file = """
def authenticate_user(username, password):
    '''Validates user credentials against database'''
    user = db.get_user(username)
    return check_password(user, password)

def fetch_user_profile(user_id):
    '''Retrieves full user profile from database'''
    return db.query(UserProfile).filter_by(id=user_id).first()

def update_user_settings(user_id, settings):
    '''Updates user preferences in database'''
    user = db.get_user(user_id)
    user.settings = settings
    db.commit()
"""

    # Semantic chunks (by function)
    chunks_semantic = [
        "def authenticate_user(username, password):\n    '''Validates user credentials against database'''\n    user = db.get_user(username)\n    return check_password(user, password)",
        "def fetch_user_profile(user_id):\n    '''Retrieves full user profile from database'''\n    return db.query(UserProfile).filter_by(id=user_id).first()",
        "def update_user_settings(user_id, settings):\n    '''Updates user preferences in database'''\n    user = db.get_user(user_id)\n    user.settings = settings\n    db.commit()"
    ]

    # Fixed-size chunks (arbitrary splits)
    chunks_fixed = [
        code_file[0:150],
        code_file[150:300],
        code_file[300:450],
        code_file[450:]
    ]

    # Query
    query = "authenticate user credentials"

    print(f"\nQuery: '{query}'")
    print(f"Semantic chunks: {len(chunks_semantic)} (by function)")
    print(f"Fixed-size chunks: {len(chunks_fixed)} (100-150 chars each)")

    print("\nGetting embeddings...")

    # TODO: Get query embedding
    # query_emb = get_embedding(query)

    # TODO: Get embeddings for semantic chunks
    # semantic_embs = [get_embedding(c) for c in chunks_semantic]

    # TODO: Get embeddings for fixed chunks
    # fixed_embs = [get_embedding(c) for c in chunks_fixed]

    # TODO: Calculate similarities for semantic chunks
    # semantic_scores = [cosine_similarity(query_emb, e) for e in semantic_embs]

    # TODO: Calculate similarities for fixed chunks
    # fixed_scores = [cosine_similarity(query_emb, e) for e in fixed_embs]

    # TODO: Find best scores
    # best_semantic_score = max(semantic_scores)
    # best_semantic_idx = semantic_scores.index(best_semantic_score)
    # best_fixed_score = max(fixed_scores)

    # TODO: Print results
    # print(f"\nBest semantic chunk score: {best_semantic_score:.4f}")
    # print(f"  Content: {chunks_semantic[best_semantic_idx][:50]}...")
    # print(f"\nBest fixed chunk score:    {best_fixed_score:.4f}")

    # TODO: Verify semantic chunking is better
    # semantic_better = best_semantic_score > best_fixed_score
    # is_auth_function = chunks_semantic[best_semantic_idx].startswith("def authenticate_user")

    # TODO: Print result
    # print(f"\n{'✅ PASS' if semantic_better and is_auth_function else '❌ FAIL'}: Semantic chunking better: {semantic_better and is_auth_function}")

    # TODO: Print validation message
    # if semantic_better and is_auth_function:
    #     print("\n✓ Validated: Semantic units > fixed-size chunks (Day 2-3, lines 666-668, 735-736)")
    #     print("✓ Implication: Validates chunk_strategy: by_function in config.yaml")
    # else:
    #     print("\n✗ Unexpected: Fixed-size chunking competed too well")

    # TODO: Return boolean result
    # return semantic_better and is_auth_function

if __name__ == "__main__":
    test_chunking_strategies()
