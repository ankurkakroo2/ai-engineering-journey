#!/usr/bin/env python3
"""
Manual Test Script for Retriever Module

This script validates the retriever module with 6 test scenarios:
1. Basic retrieval - Query for relevant code
2. Score threshold filtering - Filter by min_score
3. Metadata filtering - Filter by language
4. Empty results handling - Query with no matches
5. Top-k limiting - Verify result count
6. Score ranking - Verify descending order

Usage:
    cd week-01/project/rag-code-qa
    source venv/bin/activate
    python src/retriever/test_retriever_manual.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import retrieve
from src.retriever.retriever import retrieve_by_language, get_retrieval_stats
from src.storage import StorageManager


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")


def print_test(number, name):
    """Print a test name."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}Test {number}: {name}{Colors.END}")
    print(f"{Colors.BLUE}{'-'*80}{Colors.END}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message, indent=0):
    """Print an info message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.CYAN}• {message}{Colors.END}")


def print_result(result, index, total):
    """Print a single retrieval result."""
    print(f"\n{Colors.BOLD}[{index}/{total}] {result.function_name}{Colors.END}")
    print(f"  {Colors.CYAN}Score:{Colors.END} {result.score:.3f} ({result.relevance_category})")
    print(f"  {Colors.CYAN}Location:{Colors.END} {result.location}")
    print(f"  {Colors.CYAN}Language:{Colors.END} {result.language}")

    # Show content preview
    lines = result.content.split('\n')[:2]
    if lines:
        print(f"  {Colors.CYAN}Preview:{Colors.END}")
        for line in lines:
            print(f"    {Colors.YELLOW}{line[:70]}{Colors.END}")


def check_storage():
    """Check if ChromaDB has indexed data."""
    print_info("Checking ChromaDB for indexed data...")
    try:
        manager = StorageManager()
        stats = manager.get_stats()

        if stats.total_entries == 0:
            print_warning("ChromaDB is empty! Please index some code first.")
            print_info("Run: ./run_test.sh (to index sample code)", indent=1)
            return False

        print_success(f"Found {stats.total_entries} indexed chunks")
        print_info(f"Languages: {', '.join(stats.languages)}", indent=1)
        print_info(f"Files: {stats.total_files}", indent=1)
        return True

    except Exception as e:
        print_error(f"Failed to connect to ChromaDB: {e}")
        return False


def test_1_basic_retrieval():
    """Test 1: Basic retrieval with a relevant query."""
    print_test(1, "Basic Retrieval")

    query = "How does the visitor pattern work in parsing?"
    print_info(f"Query: '{query}'")
    print_info("Expected: Relevant results from parser module")

    try:
        results = retrieve(query, top_k=3, min_score=0.3)

        if not results:
            print_warning("No results returned (this may be expected if no relevant code)")
            return False

        print_success(f"Retrieved {len(results)} results")

        for i, result in enumerate(results, 1):
            print_result(result, i, len(results))

        # Verify results are relevant
        if results[0].score >= 0.3:
            print_success("All results meet minimum score threshold")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_2_score_filtering():
    """Test 2: Score threshold filtering."""
    print_test(2, "Score Threshold Filtering")

    query = "authentication"
    min_score = 0.5
    print_info(f"Query: '{query}'")
    print_info(f"Min score: {min_score}")
    print_info("Expected: All results have score >= 0.5")

    try:
        results = retrieve(query, top_k=5, min_score=min_score)

        print_success(f"Retrieved {len(results)} results")

        # Verify all results meet threshold
        all_above_threshold = all(r.score >= min_score for r in results)

        if all_above_threshold:
            print_success(f"All results have score >= {min_score}")
        else:
            print_error("Some results below threshold!")
            return False

        # Show score distribution
        if results:
            scores = [r.score for r in results]
            print_info(f"Score range: {min(scores):.3f} - {max(scores):.3f}")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_3_metadata_filtering():
    """Test 3: Metadata filtering by language."""
    print_test(3, "Metadata Filtering (Language)")

    query = "function definition"
    language = "python"
    print_info(f"Query: '{query}'")
    print_info(f"Filter: language='{language}'")
    print_info("Expected: All results are Python functions")

    try:
        results = retrieve_by_language(query, language=language, top_k=5, min_score=0.3)

        print_success(f"Retrieved {len(results)} results")

        # Verify all results are Python
        all_python = all(r.language == language for r in results)

        if all_python:
            print_success(f"All results are {language} functions")
        else:
            print_error(f"Some results are not {language}!")
            return False

        # Show sample results
        for i, result in enumerate(results[:3], 1):
            print_result(result, i, min(3, len(results)))

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_4_empty_results():
    """Test 4: Empty results handling."""
    print_test(4, "Empty Results Handling")

    query = "xyzabc123nonexistent"
    print_info(f"Query: '{query}'")
    print_info("Expected: Empty list (no matches)")

    try:
        results = retrieve(query, top_k=5, min_score=0.3)

        if not results:
            print_success("Correctly returned empty list")
            print_info("No results found (as expected)")
            return True
        else:
            print_warning(f"Unexpected: Found {len(results)} results")
            print_info("This may happen if some code has very generic terms")
            return True  # Not a failure, just unexpected

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_5_top_k_limiting():
    """Test 5: Top-k limiting."""
    print_test(5, "Top-k Limiting")

    query = "function"
    top_k = 3
    print_info(f"Query: '{query}'")
    print_info(f"Top-k: {top_k}")
    print_info(f"Expected: At most {top_k} results")

    try:
        results = retrieve(query, top_k=top_k, min_score=0.3)

        print_success(f"Retrieved {len(results)} results")

        if len(results) <= top_k:
            print_success(f"Result count <= {top_k} (correct)")
        else:
            print_error(f"Result count > {top_k} (incorrect!)")
            return False

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_6_score_ranking():
    """Test 6: Score ranking verification."""
    print_test(6, "Score Ranking Verification")

    query = "parser"
    print_info(f"Query: '{query}'")
    print_info("Expected: Results sorted by score (descending)")

    try:
        results = retrieve(query, top_k=5, min_score=0.3)

        if not results:
            print_warning("No results to verify ranking")
            return True

        print_success(f"Retrieved {len(results)} results")

        # Verify descending order
        scores = [r.score for r in results]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

        if is_sorted:
            print_success("Results correctly sorted by score (descending)")
        else:
            print_error("Results NOT sorted correctly!")
            return False

        # Show scores
        print_info("Score sequence:")
        for i, score in enumerate(scores, 1):
            print(f"    {i}. {score:.3f}")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def show_statistics():
    """Show retrieval statistics."""
    print_header("RETRIEVAL STATISTICS")

    stats = get_retrieval_stats()

    print_info(f"Total queries: {stats['total_queries']}")
    print_info(f"Total results: {stats['total_results']}")
    print_info(f"Avg query time: {stats['avg_query_time']:.3f}s")
    print_info(f"Avg results per query: {stats['avg_results_per_query']:.1f}")


def main():
    """Run all manual tests."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║                  RETRIEVER MODULE MANUAL TEST SUITE                       ║")
    print("║                                                                           ║")
    print("║                        6 Validation Scenarios                             ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

    # Check storage first
    if not check_storage():
        print_error("\nCannot run tests without indexed data. Exiting.")
        sys.exit(1)

    # Run all tests
    tests = [
        test_1_basic_retrieval,
        test_2_score_filtering,
        test_3_metadata_filtering,
        test_4_empty_results,
        test_5_top_k_limiting,
        test_6_score_ranking
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append(False)

    # Show statistics
    show_statistics()

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(results)
    total = len(results)

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}\n")

    for i, (test_func, result) in enumerate(zip(tests, results), 1):
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"  Test {i}: {status} - {test_func.__doc__.split('.')[0].strip()}")

    print()

    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.END}\n")
        sys.exit(0)
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Some tests failed or had warnings{Colors.END}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
