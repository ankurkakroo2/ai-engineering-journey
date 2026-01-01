"""
Experiment 1: Run All Tests

Orchestrates all 7 tests in sequence, maintaining the learning progression.

SEQUENCE:
    1. Pre-trained models (deterministic)
    2. Semantic clustering (latent dimensions)
    3. Dimensionality (1536 vs 384)
    4. Distance metrics (cosine vs Euclidean)
    5. Semantic relationships (language paradigms)
    6. Chunking strategies (function vs fixed-size)
    7. Working memory (RAG retrieval quality)

OUTPUT:
    - Prints results to stdout
    - Optionally appends to results.md

USAGE:
    python run_all.py
"""

from test_01_pretraining import test_pretraining
from test_02_semantic_clustering import test_semantic_clustering
from test_03_dimensionality import test_dimensionality
from test_04_distance_metrics import test_distance_metrics
from test_05_relationships import test_semantic_relationships
from test_06_chunking import test_chunking_strategies
from test_07_working_memory import test_working_memory

def main():
    print("\n" + "="*70)
    print("EXPERIMENT 1: EMBEDDINGS VALIDATION")
    print("Validating 7 concepts from Day 1-3 learning")
    print("="*70)

    results = {}

    # Run tests in sequence
    print("\nRunning tests in sequence...")

    try:
        results['test_01_pretraining'] = test_pretraining()
    except Exception as e:
        print(f"❌ Test 1 failed with error: {e}")
        results['test_01_pretraining'] = False

    try:
        results['test_02_semantic_clustering'] = test_semantic_clustering()
    except Exception as e:
        print(f"❌ Test 2 failed with error: {e}")
        results['test_02_semantic_clustering'] = False

    try:
        results['test_03_dimensionality'] = test_dimensionality()
    except Exception as e:
        print(f"❌ Test 3 failed with error: {e}")
        results['test_03_dimensionality'] = False

    try:
        results['test_04_distance_metrics'] = test_distance_metrics()
    except Exception as e:
        print(f"❌ Test 4 failed with error: {e}")
        results['test_04_distance_metrics'] = False

    try:
        results['test_05_relationships'] = test_semantic_relationships()
    except Exception as e:
        print(f"❌ Test 5 failed with error: {e}")
        results['test_05_relationships'] = False

    try:
        results['test_06_chunking'] = test_chunking_strategies()
    except Exception as e:
        print(f"❌ Test 6 failed with error: {e}")
        results['test_06_chunking'] = False

    try:
        results['test_07_working_memory'] = test_working_memory()
    except Exception as e:
        print(f"❌ Test 7 failed with error: {e}")
        results['test_07_working_memory'] = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    # Detailed results
    print("\nDetailed Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n✅ All tests passed! Ready to build rag-code-qa with confidence.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review findings before proceeding.")

if __name__ == "__main__":
    main()
