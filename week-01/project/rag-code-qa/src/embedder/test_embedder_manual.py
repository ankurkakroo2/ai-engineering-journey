"""
Manual Test Suite for Embedder Module

Tests the embedding pipeline end-to-end:
1. Single chunk embedding (verify 1536 dimensions)
2. Batch embedding (10+ chunks)
3. Cache hit verification (embed twice, second should be instant)
4. Determinism (same content = identical vectors)
5. Integration with parser → chunker → embedder pipeline
"""

import sys
import os
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.parser import parse_file
from src.chunker import chunk_functions
from src.embedder import (
    embed_chunk,
    embed_chunks,
    get_embedding_stats,
    get_cache_stats,
    clear_cache
)


def test_single_chunk_embedding():
    print("\n" + "="*80)
    print("TEST 1: Single Chunk Embedding")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser", "models.py"
    )

    functions = parse_file(parser_models)
    chunks = chunk_functions(functions[:1])

    if not chunks:
        print("❌ No chunks created")
        return

    chunk = chunks[0]
    print(f"\n📦 Embedding chunk: {chunk.function_name}")
    print(f"   Content length: {len(chunk.content)} chars")
    print(f"   Token count: {chunk.token_count}")

    start_time = time.time()
    embedding = embed_chunk(chunk)
    elapsed = time.time() - start_time

    if embedding:
        print(f"\n✅ Embedding created successfully")
        print(f"   Dimension: {embedding.dimension}")
        print(f"   Cached: {embedding.cached}")
        print(f"   Valid: {embedding.is_valid}")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Vector preview: [{embedding.vector[0]:.4f}, {embedding.vector[1]:.4f}, ..., {embedding.vector[-1]:.4f}]")
    else:
        print("❌ Failed to create embedding")


def test_batch_embedding():
    print("\n" + "="*80)
    print("TEST 2: Batch Embedding (10 Chunks)")
    print("="*80)

    parser_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser"
    )

    functions = parse_file(os.path.join(parser_dir, "models.py"))
    chunks = chunk_functions(functions)

    print(f"\n📦 Embedding {len(chunks)} chunks in batch")

    start_time = time.time()
    embeddings = embed_chunks(chunks, batch_size=10)
    elapsed = time.time() - start_time

    print(f"\n✅ Batch embedding complete")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Avg time per chunk: {elapsed/len(embeddings):.3f}s")

    stats = get_embedding_stats(embeddings)
    print(f"\n📊 Embedding Statistics:")
    print(f"   Total: {stats['total_embeddings']}")
    print(f"   Cached: {stats['cached_count']}")
    print(f"   API calls: {stats['api_count']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']}%")
    print(f"   All valid: {stats['all_valid']}")


def test_cache_hit():
    print("\n" + "="*80)
    print("TEST 3: Cache Hit Verification")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chunker", "models.py"
    )

    functions = parse_file(parser_models)
    chunks = chunk_functions(functions[:1])

    if not chunks:
        print("❌ No chunks created")
        return

    chunk = chunks[0]

    print(f"\n📦 First embedding (should miss cache)")
    start_time = time.time()
    embedding1 = embed_chunk(chunk)
    time1 = time.time() - start_time

    print(f"   Cached: {embedding1.cached}")
    print(f"   Time: {time1:.3f}s")

    print(f"\n📦 Second embedding (should hit cache)")
    start_time = time.time()
    embedding2 = embed_chunk(chunk)
    time2 = time.time() - start_time

    print(f"   Cached: {embedding2.cached}")
    print(f"   Time: {time2:.3f}s")

    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n✅ Cache speedup: {speedup:.1f}x faster")

    if embedding2.cached:
        print("   ✅ Cache hit confirmed")
    else:
        print("   ⚠️  Expected cache hit but got cache miss")


def test_determinism():
    print("\n" + "="*80)
    print("TEST 4: Determinism Verification")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chunker", "models.py"
    )

    functions = parse_file(parser_models)
    chunks = chunk_functions(functions[:1])

    if not chunks:
        print("❌ No chunks created")
        return

    chunk = chunks[0]

    print(f"\n📦 Generating two embeddings for same chunk")
    embedding1 = embed_chunk(chunk)
    embedding2 = embed_chunk(chunk)

    vec1 = np.array(embedding1.vector)
    vec2 = np.array(embedding2.vector)

    are_identical = np.array_equal(vec1, vec2)
    max_diff = np.max(np.abs(vec1 - vec2))

    print(f"\n📊 Comparison:")
    print(f"   Vectors identical: {are_identical}")
    print(f"   Max difference: {max_diff}")

    if are_identical:
        print("   ✅ Determinism confirmed (identical vectors)")
    else:
        print(f"   ⚠️  Vectors differ by {max_diff}")


def test_dimensions():
    print("\n" + "="*80)
    print("TEST 5: Embedding Dimensions Validation")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser", "models.py"
    )

    functions = parse_file(parser_models)
    chunks = chunk_functions(functions[:3])

    embeddings = embed_chunks(chunks)

    print(f"\n📊 Checking dimensions for {len(embeddings)} embeddings:")

    all_correct = True
    for i, emb in enumerate(embeddings):
        if emb.dimension != 1536:
            print(f"   ❌ Embedding {i}: {emb.dimension} dimensions (expected 1536)")
            all_correct = False

    if all_correct:
        print(f"   ✅ All {len(embeddings)} embeddings have correct dimension (1536)")

    print(f"\n📊 Validation:")
    for i, emb in enumerate(embeddings):
        print(f"   Embedding {i}: valid={emb.is_valid}, dim={emb.dimension}")


def test_cache_stats():
    print("\n" + "="*80)
    print("TEST 6: Cache Statistics")
    print("="*80)

    stats = get_cache_stats()

    print(f"\n📊 Cache Statistics:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Total size: {stats['total_size_mb']} MB")
    print(f"   Cache directory: {stats['cache_dir']}")


def test_integration_pipeline():
    print("\n" + "="*80)
    print("TEST 7: Full Pipeline Integration")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chunker", "models.py"
    )

    print("\n📄 Step 1: Parse file")
    functions = parse_file(parser_models)
    print(f"   Parsed {len(functions)} functions")

    print("\n📦 Step 2: Chunk functions")
    chunks = chunk_functions(functions)
    print(f"   Created {len(chunks)} chunks")

    print("\n🔢 Step 3: Embed chunks")
    embeddings = embed_chunks(chunks[:5], batch_size=5)
    print(f"   Generated {len(embeddings)} embeddings")

    stats = get_embedding_stats(embeddings)

    print(f"\n✅ Pipeline complete!")
    print(f"   Functions → Chunks → Embeddings")
    print(f"   {len(functions)} → {len(chunks)} → {len(embeddings)}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']}%")


def main():
    print("\n" + "="*80)
    print("EMBEDDER MODULE - MANUAL TEST SUITE")
    print("="*80)
    print("Testing Component 3: Chunk → Embedding transformation")
    print("\n⚠️  Note: These tests require OPENAI_API_KEY environment variable")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set. Please set it to run tests.")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        test_single_chunk_embedding()
        test_batch_embedding()
        test_cache_hit()
        test_determinism()
        test_dimensions()
        test_cache_stats()
        test_integration_pipeline()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
