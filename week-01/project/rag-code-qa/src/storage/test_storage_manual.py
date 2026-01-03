#!/usr/bin/env python3
"""
Storage Module Manual Test Script

This script validates the storage module functionality through 9 comprehensive tests.
Run this to verify that the storage module is working correctly before integrating
with the full pipeline.

Tests:
1. Client initialization and persistence
2. Collection creation with cosine similarity
3. Store chunks (upsert operation)
4. Upsert behavior (update vs insert)
5. Query similar chunks
6. Metadata filtering
7. Collection statistics
8. List all entries with pagination
9. Delete collection

Usage:
    cd /path/to/rag-code-qa
    source venv/bin/activate
    python3 src/storage/test_storage_manual.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(src_dir))

from storage import StorageManager, StorageEntry, StorageStats
from chunker.models import Chunk
from embedder.models import Embedding

# Test configuration
TEST_COLLECTION = "test-storage-manual"
TEST_PERSIST_DIR = "./data/test-storage"


def print_header(title: str):
    """Print test section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_test(test_num: int, description: str):
    """Print test number and description."""
    print(f"\n[Test {test_num}] {description}")
    print("-" * 80)


def print_success(message: str):
    """Print success message."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"✗ {message}")


def create_sample_chunks(count: int = 3) -> list:
    """Create sample chunks for testing."""
    chunks = []
    for i in range(count):
        chunk = Chunk(
            id=f"test-chunk-{i}",
            content=f"# test/file{i}.py:10\n\ndef test_function_{i}():\n    return {i}",
            metadata={
                "file_path": f"test/file{i}.py",
                "name": f"test_function_{i}",
                "start_line": 10,
                "end_line": 12,
                "language": "python",
                "docstring": f"Test function {i}",
                "token_count": 20
            },
            token_count=20,
            truncated=False
        )
        chunks.append(chunk)
    return chunks


def create_sample_embeddings(count: int = 3) -> list:
    """Create sample embeddings for testing."""
    embeddings = []
    for i in range(count):
        # Create a simple vector (1536 dimensions)
        # Use different patterns to test similarity
        vector = [0.0] * 1536
        vector[i] = 1.0  # Make each embedding slightly different

        embedding = Embedding(
            chunk_id=f"test-chunk-{i}",
            vector=vector,
            cached=False,
            metadata={"test": True}
        )
        embeddings.append(embedding)
    return embeddings


def test_1_client_initialization():
    """Test 1: Client initialization and persistence."""
    print_test(1, "Client Initialization and Persistence")

    try:
        # Create manager with test configuration
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        print_success("StorageManager initialized")
        print(f"  Persist directory: {TEST_PERSIST_DIR}")
        print(f"  Collection name: {TEST_COLLECTION}")

        # Verify persistence directory was created
        persist_path = Path(TEST_PERSIST_DIR)
        if persist_path.exists():
            print_success(f"Persistence directory created: {persist_path.absolute()}")
        else:
            print_error("Persistence directory not created")
            return False

        return True

    except Exception as e:
        print_error(f"Client initialization failed: {e}")
        return False


def test_2_collection_creation():
    """Test 2: Collection creation with cosine similarity."""
    print_test(2, "Collection Creation with Cosine Similarity")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION,
            distance_metric="cosine"
        )

        # Get initial stats (should be empty)
        stats = manager.get_stats()

        print_success("Collection created successfully")
        print(f"  Collection name: {stats.collection_name}")
        print(f"  Distance metric: {stats.distance_metric}")
        print(f"  Dimension: {stats.dimension}")
        print(f"  Total entries: {stats.total_entries}")

        if stats.distance_metric != "cosine":
            print_error(f"Expected cosine metric, got: {stats.distance_metric}")
            return False

        if stats.dimension != 1536:
            print_error(f"Expected 1536 dimensions, got: {stats.dimension}")
            return False

        return True

    except Exception as e:
        print_error(f"Collection creation failed: {e}")
        return False


def test_3_store_chunks():
    """Test 3: Store chunks (upsert operation)."""
    print_test(3, "Store Chunks (Upsert Operation)")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # Create sample data
        chunks = create_sample_chunks(3)
        embeddings = create_sample_embeddings(3)

        # Store chunks
        count = manager.store_chunks(chunks, embeddings)

        print_success(f"Stored {count} chunks")

        # Verify count
        total = manager.count()
        print_success(f"Total entries in collection: {total}")

        if total != 3:
            print_error(f"Expected 3 entries, got: {total}")
            return False

        return True

    except Exception as e:
        print_error(f"Store chunks failed: {e}")
        return False


def test_4_upsert_behavior():
    """Test 4: Upsert behavior (update vs insert)."""
    print_test(4, "Upsert Behavior (Update vs Insert)")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # Get initial count
        initial_count = manager.count()
        print(f"Initial count: {initial_count}")

        # Modify first chunk (same ID, different content)
        chunks = create_sample_chunks(1)
        chunks[0].content = "# test/file0.py:10\n\ndef test_function_0_modified():\n    return 'modified'"
        chunks[0].metadata["name"] = "test_function_0_modified"

        embeddings = create_sample_embeddings(1)

        # Upsert (should update, not insert)
        count = manager.store_chunks(chunks, embeddings)
        print_success(f"Upserted {count} chunk")

        # Verify count didn't increase
        final_count = manager.count()
        print(f"Final count: {final_count}")

        if final_count != initial_count:
            print_error(f"Upsert created duplicate! Expected {initial_count}, got {final_count}")
            return False

        print_success("Upsert correctly updated existing entry (no duplicate)")

        # Verify content was updated
        entry = manager.get_by_id("test-chunk-0")
        if entry and "modified" in entry.content:
            print_success("Content was updated successfully")
        else:
            print_error("Content was not updated")
            return False

        return True

    except Exception as e:
        print_error(f"Upsert behavior test failed: {e}")
        return False


def test_5_query_similar():
    """Test 5: Query similar chunks."""
    print_test(5, "Query Similar Chunks")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # Create query embedding (similar to chunk 1)
        query_vector = [0.0] * 1536
        query_vector[1] = 1.0  # Should match test-chunk-1

        # Search for similar chunks
        results = manager.search_similar(query_vector, top_k=3)

        print_success(f"Found {len(results)} results")

        # Display results
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.function_name} (score: {result.score:.3f})")
            print(f"     Location: {result.location}")
            print(f"     Language: {result.language}")

        if not results:
            print_error("No results returned")
            return False

        # Verify results have scores
        if not all(r.has_score for r in results):
            print_error("Some results missing similarity scores")
            return False

        # Verify results are sorted by score (descending)
        scores = [r.score for r in results]
        if scores != sorted(scores, reverse=True):
            print_error("Results not sorted by similarity score")
            return False

        print_success("Results correctly sorted by similarity")

        return True

    except Exception as e:
        print_error(f"Query similar failed: {e}")
        return False


def test_6_metadata_filtering():
    """Test 6: Metadata filtering."""
    print_test(6, "Metadata Filtering")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # Add a JavaScript chunk for filtering test
        js_chunk = Chunk(
            id="test-chunk-js",
            content="// test.js:5\nfunction testJS() { return 'js'; }",
            metadata={
                "file_path": "test.js",
                "name": "testJS",
                "start_line": 5,
                "end_line": 7,
                "language": "javascript",
                "token_count": 15
            },
            token_count=15,
            truncated=False
        )

        js_embedding = Embedding(
            chunk_id="test-chunk-js",
            vector=[0.5] * 1536,
            cached=False
        )

        manager.store_chunks([js_chunk], [js_embedding])
        print_success("Added JavaScript chunk for filtering test")

        # Query with language filter (Python only)
        query_vector = [0.0] * 1536
        query_vector[0] = 1.0

        results = manager.search_similar(
            query_vector,
            top_k=10,
            filter_by={"language": "python"}
        )

        print_success(f"Found {len(results)} Python chunks")

        # Verify all results are Python
        languages = [r.language for r in results]
        if all(lang == "python" for lang in languages):
            print_success("All results match filter (language: python)")
        else:
            print_error(f"Filter failed, found languages: {set(languages)}")
            return False

        return True

    except Exception as e:
        print_error(f"Metadata filtering failed: {e}")
        return False


def test_7_collection_statistics():
    """Test 7: Collection statistics."""
    print_test(7, "Collection Statistics")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        stats = manager.get_stats()

        print_success("Retrieved collection statistics:")
        print(f"  Total entries: {stats.total_entries}")
        print(f"  Total files: {stats.total_files}")
        print(f"  Languages: {', '.join(stats.languages)}")
        print(f"  Collection: {stats.collection_name}")
        print(f"  Dimension: {stats.dimension}")
        print(f"  Distance metric: {stats.distance_metric}")
        print(f"  Avg per file: {stats.avg_entries_per_file:.1f}")

        if stats.total_entries == 0:
            print_error("No entries in collection")
            return False

        if not stats.languages:
            print_error("No languages detected")
            return False

        print_success("Statistics look correct")

        return True

    except Exception as e:
        print_error(f"Collection statistics failed: {e}")
        return False


def test_8_list_all_entries():
    """Test 8: List all entries with pagination."""
    print_test(8, "List All Entries with Pagination")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # List first 2 entries
        entries = manager.list_entries(limit=2, offset=0)
        print_success(f"Listed {len(entries)} entries (limit=2, offset=0)")

        for i, entry in enumerate(entries, 1):
            print(f"  {i}. {entry.function_name}")
            print(f"     ID: {entry.id[:16]}...")
            print(f"     Location: {entry.location}")

        if len(entries) > 2:
            print_error(f"Limit not respected: got {len(entries)} entries")
            return False

        # List next 2 entries
        entries_page2 = manager.list_entries(limit=2, offset=2)
        print_success(f"Listed {len(entries_page2)} entries (limit=2, offset=2)")

        # Verify no overlap
        ids_page1 = {e.id for e in entries}
        ids_page2 = {e.id for e in entries_page2}

        if ids_page1 & ids_page2:
            print_error("Pagination overlap detected")
            return False

        print_success("Pagination working correctly (no overlap)")

        return True

    except Exception as e:
        print_error(f"List all entries failed: {e}")
        return False


def test_9_delete_collection():
    """Test 9: Delete collection."""
    print_test(9, "Delete Collection")

    try:
        manager = StorageManager(
            persist_directory=TEST_PERSIST_DIR,
            collection_name=TEST_COLLECTION
        )

        # Get count before deletion
        count_before = manager.count()
        print(f"Entries before deletion: {count_before}")

        # Delete collection
        deleted = manager.clear_collection()

        if deleted:
            print_success("Collection deleted successfully")
        else:
            print_error("Collection deletion returned False")
            return False

        # Verify collection is empty
        count_after = manager.count()
        print(f"Entries after deletion: {count_after}")

        if count_after != 0:
            print_error(f"Collection not empty after deletion: {count_after} entries")
            return False

        print_success("Collection is empty after deletion")

        return True

    except Exception as e:
        print_error(f"Delete collection failed: {e}")
        return False


def cleanup():
    """Clean up test data."""
    print_header("Cleanup")

    try:
        import shutil
        test_path = Path(TEST_PERSIST_DIR)

        if test_path.exists():
            shutil.rmtree(test_path)
            print_success(f"Removed test directory: {test_path}")

    except Exception as e:
        print_error(f"Cleanup failed: {e}")


def main():
    """Run all tests."""
    print_header("Storage Module Manual Test Suite")
    print("This script validates the storage module functionality.")
    print(f"Test collection: {TEST_COLLECTION}")
    print(f"Test directory: {TEST_PERSIST_DIR}")

    # Run tests
    tests = [
        test_1_client_initialization,
        test_2_collection_creation,
        test_3_store_chunks,
        test_4_upsert_behavior,
        test_5_query_similar,
        test_6_metadata_filtering,
        test_7_collection_statistics,
        test_8_list_all_entries,
        test_9_delete_collection
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append(False)

    # Summary
    print_header("Test Summary")
    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Tests failed: {total - passed}/{total}")

    if passed == total:
        print_success("All tests passed! ✓")
    else:
        print_error(f"{total - passed} test(s) failed")

    # Cleanup
    cleanup()

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
