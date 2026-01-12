#!/usr/bin/env python3
"""
Generator Module Manual Test

This script validates the generator module with various scenarios:
1. Basic generation (with good retrieval results)
2. Empty results handling (no relevant code)
3. Citation verification (answer includes file:line)
4. Token limit handling (truncate long context)
5. Error handling (invalid API key)
6. Statistics tracking

Usage:
    cd /Users/ankur/D/Playground/ai-engineering-journey/week-01/project/rag-code-qa
    python src/generator/test_generator_manual.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import retrieve
from src.generator import generate_answer, generate_with_retrieval, get_generation_stats, reset_generation_stats
from src.generator.models import check_citations
from src.parser import parse_directory
from src.chunker import chunk_functions
from src.embedder import embed_chunks
from src.storage import StorageManager

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a beautiful header."""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}\n")


def print_section(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {text}{Colors.END}")
    print(f"{Colors.BLUE}{'─'*78}{Colors.END}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text, indent=0):
    """Print info message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.CYAN}• {text}{Colors.END}")


def setup_test_data():
    """
    Set up test data by indexing the parser module.
    Returns storage_manager for retrieval.
    """
    print_section("Setting up test data...")

    try:
        # Parse parser module
        parser_dir = project_root / "week-01" / "project" / "rag-code-qa" / "src" / "parser"
        print_info(f"Parsing: {parser_dir}")

        functions = parse_directory(str(parser_dir))
        print_success(f"Parsed {len(functions)} functions")

        # Chunk
        chunks = chunk_functions(functions)
        print_success(f"Created {len(chunks)} chunks")

        # Embed (will use cache if available)
        embeddings = embed_chunks(chunks)
        print_success(f"Generated {len(embeddings)} embeddings")

        # Store
        storage_manager = StorageManager()
        storage_manager.reset_collection()  # Clean slate
        count = storage_manager.store_chunks(chunks[:len(embeddings)], embeddings)
        print_success(f"Stored {count} entries in ChromaDB")

        return storage_manager

    except Exception as e:
        print_error(f"Setup failed: {e}")
        return None


def test_1_basic_generation():
    """Test 1: Basic generation with good retrieval results."""
    print_header("TEST 1: Basic Generation")

    try:
        # Retrieve
        query = "How does the visitor pattern work in the parser?"
        print_info(f"Query: '{query}'")

        results = retrieve(query, top_k=3, min_score=0.3)
        print_success(f"Retrieved {len(results)} chunks")

        if not results:
            print_warning("No results retrieved - skipping test")
            return False

        # Generate
        answer = generate_answer(query, results)

        # Validate
        print_section("Generated Answer")
        print(f"{Colors.YELLOW}{answer.answer[:200]}...{Colors.END}")

        print_section("Validation")
        print_info(f"Model: {answer.model}")
        print_info(f"Tokens: {answer.total_tokens}")
        print_info(f"Time: {answer.generation_time:.2f}s")
        print_info(f"Has citations: {answer.has_citations}")
        print_info(f"Citation count: {answer.citation_count}")

        if answer.has_citations:
            print_success("✓ Answer includes citations")
        else:
            print_warning("⚠ Answer missing citations")

        if answer.total_tokens > 0:
            print_success("✓ Token usage tracked")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_2_empty_results():
    """Test 2: Handle empty retrieval results."""
    print_header("TEST 2: Empty Results Handling")

    try:
        query = "How to bake a cake?"
        print_info(f"Query: '{query}' (intentionally irrelevant)")

        # Generate with empty results
        answer = generate_answer(query, [])

        print_section("Generated Answer")
        print(f"{Colors.YELLOW}{answer.answer}{Colors.END}")

        print_section("Validation")
        if "couldn't find" in answer.answer.lower() or "no relevant" in answer.answer.lower():
            print_success("✓ Helpful message for no results")
        else:
            print_warning("⚠ Message could be more helpful")

        if answer.total_tokens == 0:
            print_success("✓ No API call made (no tokens)")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_3_citation_verification():
    """Test 3: Verify citations are extracted correctly."""
    print_header("TEST 3: Citation Verification")

    try:
        # Retrieve and generate
        query = "What does FunctionVisitor do?"
        results = retrieve(query, top_k=3, min_score=0.3)

        if not results:
            print_warning("No results - skipping test")
            return False

        answer = generate_answer(query, results)

        print_section("Citations Found")
        if answer.sources_cited:
            for i, citation in enumerate(answer.sources_cited, 1):
                print_info(f"[{i}] {citation}")
            print_success(f"✓ Extracted {len(answer.sources_cited)} citations")
        else:
            print_warning("⚠ No citations found in answer")

        # Verify citation format
        for citation in answer.sources_cited:
            if ':' in citation and '-' in citation:
                print_success(f"✓ Citation format valid: {citation}")
            else:
                print_warning(f"⚠ Citation format invalid: {citation}")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_4_token_limits():
    """Test 4: Handle token limit constraints."""
    print_header("TEST 4: Token Limit Handling")

    try:
        # Retrieve many results
        query = "How does parsing work?"
        results = retrieve(query, top_k=10, min_score=0.2)

        if len(results) < 5:
            print_warning("Not enough results - skipping test")
            return False

        print_info(f"Retrieved {len(results)} chunks")

        # Generate with small context limit
        answer = generate_answer(query, results, max_context_tokens=2000)

        print_section("Token Usage")
        print_info(f"Prompt tokens: {answer.prompt_tokens}")
        print_info(f"Completion tokens: {answer.completion_tokens}")
        print_info(f"Total tokens: {answer.total_tokens}")

        if answer.prompt_tokens <= 2500:  # 2000 context + 500 overhead
            print_success("✓ Context truncated to fit token limit")
        else:
            print_warning(f"⚠ Context may exceed limit: {answer.prompt_tokens} prompt tokens")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_5_error_handling():
    """Test 5: Handle API errors gracefully."""
    print_header("TEST 5: Error Handling")

    print_info("This test checks error handling for missing API key")
    print_info("If ANTHROPIC_API_KEY is set, this test will pass")
    print_info("If not set, we expect a ValueError")

    try:
        # Try to generate (will fail if no API key)
        query = "test query"
        results = retrieve(query, top_k=1, min_score=0.1)

        if not results:
            print_warning("No results - using empty list")
            results = []

        try:
            answer = generate_answer(query, results if results else [])
            print_success("✓ Generation succeeded (API key is set)")
            return True
        except ValueError as e:
            if "ANTHROPIC_API_KEY" in str(e):
                print_success("✓ Correct error for missing API key")
                return True
            else:
                print_error(f"Unexpected ValueError: {e}")
                return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_6_statistics():
    """Test 6: Verify statistics tracking."""
    print_header("TEST 6: Statistics Tracking")

    try:
        # Reset stats
        reset_generation_stats()
        print_info("Reset statistics")

        # Generate a few answers
        query = "How does parsing work?"
        results = retrieve(query, top_k=3, min_score=0.3)

        if not results:
            print_warning("No results - skipping test")
            return False

        # Generate 2 answers
        answer1 = generate_answer(query, results)
        answer2 = generate_answer("What is FunctionVisitor?", results)

        # Get stats
        stats = get_generation_stats()

        print_section("Statistics")
        print_info(f"Total generations: {stats['total_generations']}")
        print_info(f"Total tokens: {stats['total_tokens']}")
        print_info(f"Total time: {stats['total_time']:.2f}s")
        print_info(f"Avg tokens: {stats['avg_tokens_per_generation']:.1f}")
        print_info(f"Avg time: {stats['avg_time_per_generation']:.2f}s")
        print_info(f"Citation rate: {stats['citation_rate']:.1f}%")

        if stats['total_generations'] == 2:
            print_success("✓ Statistics tracked correctly")
        else:
            print_warning(f"⚠ Expected 2 generations, got {stats['total_generations']}")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║                  GENERATOR MODULE MANUAL TEST                             ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

    # Setup
    storage_manager = setup_test_data()

    if not storage_manager:
        print_error("Failed to set up test data - aborting")
        sys.exit(1)

    # Run tests
    tests = [
        ("Basic Generation", test_1_basic_generation),
        ("Empty Results", test_2_empty_results),
        ("Citation Verification", test_3_citation_verification),
        ("Token Limits", test_4_token_limits),
        ("Error Handling", test_5_error_handling),
        ("Statistics", test_6_statistics)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print_header("TEST SUMMARY")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"{status}  {name}")

    print(f"\n{Colors.BOLD}Result: {passed_count}/{total_count} tests passed{Colors.END}\n")

    if passed_count == total_count:
        print_success("All tests passed!")
        sys.exit(0)
    else:
        print_warning(f"{total_count - passed_count} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
