# Retriever Module - Semantic Code Search

## Learning Context

This is **Component 5** of your RAG-Code-QA pipeline (Week 1, Day 7-8).

After indexing code through the Parser → Chunker → Embedder → Storage pipeline, you now build the **query side** of RAG. The retriever enables semantic search: finding code by meaning, not just keywords.

**Previous Components:**
1. ✅ Parser - Extract functions from code
2. ✅ Chunker - Create embeddable units
3. ✅ Embedder - Generate vector embeddings
4. ✅ Storage - Persist to ChromaDB

**This Component:**
5. **Retriever** - Semantic search over indexed code

**Next Components:**
6. Generator - Use Claude to generate answers
7. CLI - Command-line interface

---

## Mental Model: How Semantic Search Works

### The Core Concept

Traditional search uses **keyword matching** (grep, regex). Semantic search uses **meaning**:

```
Query: "How does authentication work?"
   ↓ (embed with same model)
Query Vector: [0.023, -0.145, 0.089, ..., 0.234]  (1536 dims)
   ↓ (compute cosine similarity)
Compare against ALL stored vectors in ChromaDB
   ↓ (rank by similarity score)
Top 5 Results:
  1. login_handler() - score: 0.87
  2. verify_token() - score: 0.82
  3. authenticate_user() - score: 0.79
  4. check_permissions() - score: 0.65
  5. hash_password() - score: 0.58
```

**Key Insight:** The query "authentication" didn't appear in the code, but the embeddings captured the semantic relationship between the query and functions like `login_handler()` and `verify_token()`.

### Why Same Model Matters

Embeddings create a **semantic space** where similar concepts are close together. If you use different models for indexing vs querying, you're comparing vectors from **different spaces**.

**Analogy:** It's like comparing GPS coordinates from Earth to Mars coordinates - they're in different coordinate systems!

**Example:**
```
text-embedding-3-small: "authentication" → [0.12, 0.45, ...]
text-embedding-ada-002:  "authentication" → [0.89, -0.23, ...]
                                             ↑ Different space!
```

**Critical Rule:** Query embeddings MUST use the same model as indexing (`text-embedding-3-small`).

### Distance vs Similarity

ChromaDB returns **distance** (how far apart), but we think in **similarity** (how close):

```
Cosine Distance: 0.0 = identical, 2.0 = opposite
Cosine Similarity: 1.0 = identical, 0.0 = unrelated

Conversion: similarity = 1 - distance

Example Results:
  distance=0.13 → similarity=0.87 (very relevant)
  distance=0.35 → similarity=0.65 (somewhat relevant)
  distance=0.70 → similarity=0.30 (barely relevant)
```

### Score Thresholds

Not all results are useful. A **minimum score threshold** filters noise:

```
Query: "authentication"
Results without threshold:
  1. login_handler() - 0.87 ✓ relevant
  2. verify_token() - 0.82 ✓ relevant
  3. format_date() - 0.25 ✗ noise (unrelated)
  4. log_error() - 0.18 ✗ noise

Results with min_score=0.3:
  1. login_handler() - 0.87 ✓
  2. verify_token() - 0.82 ✓
  (noise filtered out)
```

**Typical thresholds:**
- **0.7+** : Highly relevant (strong semantic match)
- **0.5-0.7** : Moderately relevant (good match)
- **0.3-0.5** : Possibly relevant (weak match)
- **<0.3** : Likely noise (unrelated)

**Default:** `min_score=0.3` (based on Experiment 1, Test 7 validation)

### Metadata Filtering

Combine semantic search with **structured filters**:

```
Query: "authentication" + filter: {"language": "python"}
   ↓
Only search Python functions, ignore JavaScript/TypeScript
   ↓
More precise results for specific contexts
```

---

## Module Architecture

### High-Level Flow

```
User Query
    ↓
Query Embedder → OpenAI API → Query Vector (1536d)
    ↓
Searcher → ChromaDB → Raw Results
    ↓
Result Processor → Filter by Score → Rank → Convert
    ↓
RetrievalResult List (for Generator)
```

### Module Structure

```
src/retriever/
├── __init__.py              # Public API (retrieve function)
├── models.py                # RetrievalResult dataclass
├── query_embedder.py        # Embed queries (reuses embedder)
├── searcher.py              # ChromaDB search operations
├── result_processor.py      # Score filtering, ranking
├── retriever.py             # Main orchestrator (facade)
├── README.md                # This file
├── LLD.md                   # Low-level design diagrams
└── test_retriever_manual.py # Manual validation script
```

### File Responsibilities

| File | Purpose | Key Functions |
|------|---------|---------------|
| `models.py` | Data structures | `RetrievalResult` dataclass |
| `query_embedder.py` | Query embedding | `embed_query()` |
| `searcher.py` | ChromaDB search | `search_similar()` |
| `result_processor.py` | Result processing | `filter_by_score()`, `rank_by_score()` |
| `retriever.py` | Main orchestrator | `retrieve()` (public API) |

---

## Key Learning Points

### 1. Same Model Requirement

**Why:** Query and index embeddings must live in the same semantic space.

**Implementation:**
```python
# query_embedder.py reuses embedder.api_client
from ..embedder.api_client import create_openai_client, call_embedding_api

def embed_query(query: str) -> List[float]:
    client = create_openai_client()
    vector = call_embedding_api(client, query)  # text-embedding-3-small
    return vector
```

### 2. Over-fetching for Filtering

**Why:** Score filtering may remove many results. Request more than needed.

**Implementation:**
```python
# retriever.py
raw_results = search_similar(
    query_vector=query_vector,
    top_k=top_k * 2,  # Request 2x, filter to top_k
    filter_metadata=filter_metadata
)
```

### 3. Result Processing Pipeline

**Why:** Separate concerns (search vs filter vs convert) for testability.

**Implementation:**
```python
# result_processor.py
def process_results(storage_entries, min_score, top_k):
    filtered = filter_by_score(storage_entries, min_score)
    ranked = rank_by_score(filtered)
    results = convert_to_retrieval_results(ranked)
    deduplicated = deduplicate_results(results)
    final = limit_results(deduplicated, top_k)
    return final
```

### 4. Facade Pattern

**Why:** Simple public API hides complexity.

**Implementation:**
```python
# retriever.py (public API)
def retrieve(query, top_k=5, min_score=0.3, filter_metadata=None):
    query_vector = embed_query(query)
    raw_results = search_similar(query_vector, top_k * 2, filter_metadata)
    results = process_results(raw_results, min_score, top_k)
    return results
```

---

## Usage Examples

### Basic Retrieval

```python
from src.retriever import retrieve

# Simple query
results = retrieve("How does authentication work?", top_k=5)

for result in results:
    print(f"{result.function_name}: {result.score:.3f}")
    print(f"  Location: {result.location}")
    print(f"  Relevance: {result.relevance_category}")
    print()
```

**Output:**
```
login_handler: 0.872
  Location: src/auth/handlers.py:45-67
  Relevance: highly_relevant

verify_token: 0.819
  Location: src/auth/tokens.py:23-41
  Relevance: highly_relevant

authenticate_user: 0.785
  Location: src/auth/auth.py:12-34
  Relevance: highly_relevant
```

### Filtered Retrieval (by Language)

```python
# Search only Python functions
results = retrieve(
    "error handling",
    top_k=5,
    min_score=0.5,
    filter_metadata={"language": "python"}
)

# Or use convenience function
from src.retriever.retriever import retrieve_by_language

results = retrieve_by_language(
    "error handling",
    language="python",
    top_k=5,
    min_score=0.5
)
```

### Score Analysis

```python
from src.retriever.result_processor import get_result_statistics

results = retrieve("authentication", top_k=10, min_score=0.3)
stats = get_result_statistics(results)

print(f"Total results: {stats['total_results']}")
print(f"Average score: {stats['avg_score']:.3f}")
print(f"Highly relevant: {stats['highly_relevant_count']}")
print(f"Score distribution: {stats['score_distribution']}")
```

**Output:**
```
Total results: 10
Average score: 0.642
Highly relevant: 3
Score distribution: {
    'highly_relevant': 3,
    'moderately_relevant': 4,
    'possibly_relevant': 2,
    'low_relevance': 1
}
```

### Handling Empty Results

```python
results = retrieve("xyzabc123", top_k=5, min_score=0.3)

if not results:
    print("No relevant results found.")
    print("Try:")
    print("  - Lowering min_score (e.g., 0.2)")
    print("  - Rephrasing the query")
    print("  - Checking if code is indexed")
```

---

## Integration Points

### With Embedder Module

**Reuses:** `api_client.create_openai_client()`, `call_embedding_api()`

**Why:** Ensures same model (`text-embedding-3-small`) for consistency

**Import:**
```python
from ..embedder.api_client import create_openai_client, call_embedding_api
```

### With Storage Module

**Reuses:** `StorageManager.search_similar()` (wraps `operations.query_similar()`)

**Why:** Storage already implements ChromaDB query logic

**Import:**
```python
from ..storage import StorageManager
from ..storage.models import StorageEntry
```

### With Generator Module (Next)

**Provides:** `List[RetrievalResult]` with code chunks and scores

**Generator will:** Use retrieved chunks as context for Claude

**Data Flow:**
```
Retriever → RetrievalResult list → Generator → Answer with citations
```

---

## Testing

### Manual Test Script

Run the manual test script to validate all scenarios:

```bash
cd week-01/project/rag-code-qa
source venv/bin/activate
python src/retriever/test_retriever_manual.py
```

**Test Cases:**
1. Basic retrieval (query: "visitor pattern")
2. Score threshold filtering (min_score=0.5)
3. Metadata filtering (language="python")
4. Empty results handling (query: "xyzabc123")
5. Top-k limiting (request 3, get max 3)
6. Score ranking verification (descending order)

### Pipeline Integration

The retriever is integrated into the main pipeline test (`src/test.py`) as Stage 5:

```bash
./run_test.sh
```

This shows the complete flow: Parser → Chunker → Embedder → Storage → **Retriever**

---

## Performance Considerations

### Query Time Breakdown

Typical query takes ~0.5-1.0 seconds:
- **Embedding:** ~0.2-0.4s (OpenAI API call)
- **Search:** ~0.05-0.1s (ChromaDB HNSW index)
- **Processing:** ~0.01-0.05s (filtering, ranking)

### Optimization Tips

1. **Batch queries** when possible (use `embed_queries_batch()`)
2. **Adjust top_k** based on needs (lower = faster)
3. **Use metadata filters** to reduce search space
4. **Monitor cache hit rate** for embedder

### Scaling Considerations

For production:
- **ChromaDB server mode** (instead of embedded)
- **Query caching** (cache query embeddings for common queries)
- **Result caching** (cache results for identical queries)
- **Async API calls** (non-blocking embedding)

---

## Common Issues & Solutions

### Issue: No results returned

**Causes:**
- min_score too high
- Query too specific
- Code not indexed

**Solutions:**
```python
# Lower threshold
results = retrieve(query, min_score=0.2)

# Check if code is indexed
from src.storage import StorageManager
manager = StorageManager()
stats = manager.get_stats()
print(f"Indexed entries: {stats.total_entries}")
```

### Issue: Low relevance scores

**Causes:**
- Query doesn't match code semantics
- Different terminology used

**Solutions:**
- Rephrase query using code-specific terms
- Try multiple query variations
- Check query embedding dimension (should be 1536)

### Issue: Slow queries

**Causes:**
- Large collection
- Network latency (OpenAI API)

**Solutions:**
- Use metadata filters to reduce search space
- Consider query caching
- Monitor with `get_retrieval_stats()`

---

## Next Steps

After completing the retriever:

1. **Test thoroughly** - Run manual tests and pipeline test
2. **Build Generator** - Use Claude to generate answers with retrieved context
3. **Build CLI** - Command-line interface for `index` and `query` commands
4. **End-to-End Testing** - Full pipeline from code to answer

The retriever is the bridge between indexed code and generated answers. It's where semantic search comes alive! 🎯
