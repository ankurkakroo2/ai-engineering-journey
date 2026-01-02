"""
Manual Test Suite for Chunker Module

Tests the chunking pipeline end-to-end:
1. Parse functions from source files
2. Transform into embeddable chunks
3. Validate token limits and metadata preservation
4. Test edge cases (large functions, truncation, empty content)
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.parser import parse_file, parse_directory
from src.chunker import (
    chunk_function,
    chunk_functions,
    get_chunking_stats,
    count_tokens,
    get_token_stats
)


def test_single_function_chunking():
    print("\n" + "="*80)
    print("TEST 1: Single Function Chunking")
    print("="*80)

    parser_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser", "models.py"
    )

    functions = parse_file(parser_models)

    if not functions:
        print("❌ No functions parsed")
        return

    func = functions[0]
    print(f"\n📄 Parsing: {func.name} from {func.file_path}")
    print(f"   Lines: {func.start_line}-{func.end_line}")
    print(f"   Has docstring: {func.has_docstring}")

    chunk = chunk_function(func)

    if chunk:
        print(f"\n✅ Chunk created successfully")
        print(f"   ID: {chunk.id[:16]}...")
        print(f"   Tokens: {chunk.token_count}")
        print(f"   Truncated: {chunk.truncated}")
        print(f"   Valid: {chunk.is_valid}")
        print(f"\n📝 Content preview (first 300 chars):")
        print(chunk.content[:300])
        print("...")
    else:
        print("❌ Failed to create chunk")


def test_batch_chunking():
    print("\n" + "="*80)
    print("TEST 2: Batch Chunking (Parser Module)")
    print("="*80)

    parser_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser"
    )

    functions = parse_directory(parser_dir, languages=['python'])
    print(f"\n📦 Parsed {len(functions)} functions from parser module")

    chunks = chunk_functions(functions)
    print(f"✅ Created {len(chunks)} chunks")

    stats = get_chunking_stats(chunks)
    print(f"\n📊 Chunking Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Avg tokens: {stats['avg_tokens']}")
    print(f"   Max tokens: {stats['max_tokens']}")
    print(f"   Min tokens: {stats['min_tokens']}")
    print(f"   Truncated: {stats['truncated_count']}")
    print(f"   With docstring: {stats['with_docstring']}")


def test_token_counting():
    print("\n" + "="*80)
    print("TEST 3: Token Counting Accuracy")
    print("="*80)

    test_strings = [
        "def hello(): pass",
        "authenticate_user",
        "location",
        """
        def calculate_total(items):
            total = sum(item.price for item in items)
            return total
        """
    ]

    for text in test_strings:
        token_count = count_tokens(text)
        print(f"\n📝 Text: {text[:50]}...")
        print(f"   Characters: {len(text)}")
        print(f"   Tokens: {token_count}")
        print(f"   Chars/Token ratio: {len(text)/token_count:.2f}")


def test_chunk_metadata():
    print("\n" + "="*80)
    print("TEST 4: Metadata Preservation")
    print("="*80)

    chunker_models = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models.py"
    )

    functions = parse_file(chunker_models)

    if not functions:
        print("❌ No functions parsed")
        return

    chunks = chunk_functions(functions)

    print(f"\n📋 Inspecting {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n{i}. {chunk.function_name}")
        print(f"   Location: {chunk.location}")
        print(f"   Language: {chunk.metadata['language']}")
        print(f"   Has docstring: {chunk.has_docstring}")
        print(f"   Token count: {chunk.token_count}")


def test_large_function_truncation():
    print("\n" + "="*80)
    print("TEST 5: Large Function Truncation")
    print("="*80)

    parser_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "parser", "python_parser.py"
    )

    functions = parse_file(parser_python)

    large_functions = [f for f in functions if len(f.code) > 1000]

    if not large_functions:
        print("ℹ️  No large functions found (>1000 chars)")
        return

    print(f"\n📏 Found {len(large_functions)} large functions")

    for func in large_functions[:2]:
        chunk = chunk_function(func, max_tokens=500)

        print(f"\n📄 {func.name}")
        print(f"   Original code length: {len(func.code)} chars")
        print(f"   Chunk tokens: {chunk.token_count}")
        print(f"   Truncated: {chunk.truncated}")

        if chunk.truncated:
            print(f"   ✂️  Content was truncated to fit 500 token limit")


def test_chunk_serialization():
    print("\n" + "="*80)
    print("TEST 6: Chunk Serialization (to_dict/from_dict)")
    print("="*80)

    chunker_models = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models.py"
    )

    functions = parse_file(chunker_models)

    if not functions:
        print("❌ No functions parsed")
        return

    chunk = chunk_function(functions[0])

    chunk_dict = chunk.to_dict()
    print(f"\n📤 Serialized chunk to dict:")
    print(f"   Keys: {list(chunk_dict.keys())}")
    print(f"   ID: {chunk_dict['id'][:16]}...")
    print(f"   Token count: {chunk_dict['token_count']}")

    from src.chunker.models import Chunk
    restored_chunk = Chunk.from_dict(chunk_dict)

    print(f"\n📥 Restored chunk from dict:")
    print(f"   ID matches: {restored_chunk.id == chunk.id}")
    print(f"   Content matches: {restored_chunk.content == chunk.content}")
    print(f"   Metadata matches: {restored_chunk.metadata == chunk.metadata}")
    print(f"   ✅ Serialization round-trip successful")


def test_deterministic_ids():
    print("\n" + "="*80)
    print("TEST 7: Deterministic ID Generation")
    print("="*80)

    chunker_models = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models.py"
    )

    functions = parse_file(chunker_models)

    if not functions:
        print("❌ No functions parsed")
        return

    func = functions[0]

    chunk1 = chunk_function(func)
    chunk2 = chunk_function(func)

    print(f"\n🔑 Testing ID determinism for: {func.name}")
    print(f"   First ID:  {chunk1.id}")
    print(f"   Second ID: {chunk2.id}")
    print(f"   IDs match: {chunk1.id == chunk2.id}")

    if chunk1.id == chunk2.id:
        print(f"   ✅ IDs are deterministic (same content → same ID)")
    else:
        print(f"   ❌ IDs differ (non-deterministic)")


def main():
    print("\n" + "="*80)
    print("CHUNKER MODULE - MANUAL TEST SUITE")
    print("="*80)
    print("Testing Component 2: ParsedFunction → Chunk transformation")

    try:
        test_single_function_chunking()
        test_batch_chunking()
        test_token_counting()
        test_chunk_metadata()
        test_large_function_truncation()
        test_chunk_serialization()
        test_deterministic_ids()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
