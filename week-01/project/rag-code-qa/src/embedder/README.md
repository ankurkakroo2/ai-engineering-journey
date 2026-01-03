# Embedder Module - Component 3 of RAG-Code-QA Pipeline

**Learning Context**: Week 1, Day 5-6 of 8-Week AI Engineering Journey
**Status**: ✅ Complete and Tested
**Purpose**: Generate vector embeddings from chunks using OpenAI API with caching

---

## Table of Contents

1. [Overview](#overview)
2. [Mental Model](#mental-model)
3. [Learning Context](#learning-context)
4. [Architecture](#architecture)
5. [Module Structure](#module-structure)
6. [Key Learning Points](#key-learning-points)
7. [Usage Examples](#usage-examples)
8. [Testing](#testing)

---

## Overview

### What This Module Does

The embedder module is **Component 3** of the RAG-Code-QA pipeline. It transforms formatted chunk text into high-dimensional vectors for semantic search:

- **Calls** OpenAI Embedding API (text-embedding-3-small)
- **Generates** 1536-dimensional vectors
- **Caches** embeddings to avoid redundant API calls
- **Batches** multiple chunks for efficiency
- **Handles** rate limiting with exponential backoff

### Why It Exists

**Problem**: Chunks are text strings, but semantic search requires vectors:
- Text can't be compared for "similarity" directly
- Need mathematical representation in high-dimensional space
- API calls are slow (~50-200ms) and costly

**Solution**: The embedder bridges the gap by:
1. Calling pre-trained embedding model (frozen, deterministic)
2. Caching results (same input = same output, always)
3. Batching requests (reduce API overhead)

**Without the embedder**, we can't perform semantic search on code.

---

## Mental Model

### What is an Embedding?

An embedding is a **learned representation** of text in high-dimensional space. Think of it as coordinates in a 1536-dimensional space where:
- Each dimension captures a learned semantic feature
- Similar meanings → nearby points
- Dissimilar meanings → distant points

**The OpenAI Model:**
- Pre-trained on massive text corpora
- Frozen (no training during inference)
- Deterministic (same input = same output)
- 1536 dimensions (text-embedding-3-small)

**Geometric Interpretation:**

```
2D visualization (actual: 1536D):

    "def authenticate_user()"
            ↓
        [0.23, 0.45]
            ↓
    Point in space
            ↓
    Near: "def login()", "def verify_credentials()"
    Far: "def calculate_total()", "def render_html()"
```

### Why Determinism Enables Caching

From your Test 1 validation: Embeddings are **deterministic**.

```python
chunk_content = "def hello(): pass"

embedding_1 = embed(chunk_content)  # API call
embedding_2 = embed(chunk_content)  # Should use cache!

embedding_1 == embedding_2  # Always True (exactly identical)
```

**Why this works:**
1. The model is pre-trained and frozen (no randomness)
2. Same input → same tokenization → same forward pass → same output
3. Not "similar" vectors - **identical** vectors (every single number matches)

**Efficiency gain:**
- API call: 50-200ms + $0.00002 per 1K tokens
- Cache hit: <1ms + $0 (free)
- For 1000 functions: Save ~100 seconds and ~$0.20 on re-indexing

### The Three Layers of Embedding

**Layer 1: Tokenization**

Text is split into subword units (using tiktoken):

```
"def authenticate_user()"
    ↓
["def", " authenticate", "_user", "(", ")"]
    ↓
[1234, 5678, 9012, 3456, 7890]  # Token IDs
```

**Layer 2: Transformer Processing**

Token IDs pass through neural network layers:
- Embedding layer (token ID → initial vector)
- Attention layers (capture context and relationships)
- Feed-forward layers (non-linear transformations)

**Layer 3: Pooling**

Multiple token vectors → single chunk vector:
- Average pooling or CLS token
- Result: 1536-dimensional vector
- This vector represents the entire chunk

### Content-Based Caching Strategy

**Cache Key Generation:**

```
chunk.content → SHA-256 hash → 64-character hex string
```

**Why SHA-256:**
1. **Deterministic**: Same content → same hash
2. **Avalanche effect**: 1 char change → completely different hash
3. **Collision-resistant**: Probability ≈ 0

**Cache Workflow:**

```
1. Compute cache key: SHA-256(chunk.content)
2. Check if cache file exists: ./data/cache/{key}.json
3. If yes: Load vector from file (fast path)
4. If no: Call API + save to cache (slow path)
```

**No Manual Invalidation:**
- Changed code → different chunk.content → different cache key
- Old cache entries naturally become unused
- Can garbage collect periodically if needed

### Batch Processing Strategy

The OpenAI API supports batching up to 2048 inputs per request.

**Individual calls (slow):**
```python
for chunk in chunks:  # 100 chunks
    embedding = api.embed(chunk.content)  # 100 API calls
# Total time: ~10 seconds
```

**Batch processing (fast):**
```python
batch = [chunk.content for chunk in chunks[:100]]
embeddings = api.embed(batch)  # 1 API call
# Total time: ~1 second (10x faster)
```

**Trade-offs:**
- **Pros**: Reduced API overhead, fewer HTTP requests, faster
- **Cons**: More memory usage, all-or-nothing failure mode
- **Optimal**: 100-500 chunks per batch (balance latency/memory)

### Cache-First Orchestration

The embedder always checks cache before calling API:

```
embed_chunks(chunks):
    ↓
For each chunk:
    ↓
Check cache → Hit? → Return cached vector
    ↓ (miss)
Add to uncached_batch
    ↓
For each batch of uncached:
    ↓
Call API (batch of 100)
    ↓
Save all to cache
    ↓
Return all embeddings (cached + new)
```

**Statistics Tracking:**
- Cache hits: How many from cache
- API calls: How many batches sent
- Cache hit rate: % from cache
- Time: Total elapsed time

---

## Learning Context

### Your Journey So Far

**Days 1-3**: Deep learning foundation
- Embeddings as pre-trained semantic lookups
- High-dimensional vector spaces
- Cosine similarity for semantic search

**Day 4**: Planning & experiments
- Test 1: Validated embedding determinism
- Test 2: Semantic clustering across languages
- Test 3: Dimensionality impact on accuracy

**Day 5-6**: Implementation
- Built parser (Component 1)
- Built chunker (Component 2)
- Built embedder (Component 3) ← **You are here**

### What You're Learning

1. **Deterministic Embeddings**: Same input always produces same output (enables caching)
2. **Content-Based Caching**: SHA-256 keys eliminate manual cache invalidation
3. **Batch Processing**: Reduce API overhead by grouping requests
4. **Rate Limit Handling**: Exponential backoff for graceful degradation
5. **High-Dimensional Vectors**: 1536 dimensions capture semantic relationships

### Connection to RAG Theory

From your Day 2-3 learning:

> "Embeddings map text to vectors where semantic similarity = geometric proximity."

The embedder's job is to transform code chunks into this geometric representation. When a user queries "How does authentication work?", we:
1. Embed the query → vector in 1536D space
2. Find nearest chunk vectors → cosine similarity
3. Retrieve those chunks → pass to LLM as context

This is the core of semantic search: **meaning-based retrieval**, not keyword matching.

---

## Architecture

### Pipeline Flow

```
Chunk Objects (from chunker)
    ↓
Embedder Module
    ↓
[Cache Check] → Cache Hit? → Return Cached Embedding
    ↓ (miss)
[Batch Grouping] → Group into batches of ~100
    ↓
[API Call] → OpenAI text-embedding-3-small
    ↓
[Cache Save] → Store for future use
    ↓
Vector Embeddings (1536-dim arrays)
    ↓
Storage Module (ChromaDB)
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Embedding Model** | text-embedding-3-small | 1536 dims, validated in Test 1 & 3 |
| **Cache Strategy** | Content-based SHA-256 keys | Deterministic, no manual invalidation |
| **Cache Storage** | Local disk (JSON files) | **Learning/dev choice** - simple, debuggable |
| **Batch Size** | 100 chunks (configurable) | Balance API efficiency & memory |
| **Rate Limiting** | Exponential backoff | Graceful handling of API limits |
| **Error Handling** | Retry 3x, then fail gracefully | Don't crash entire batch on one error |

**⚠️ Cache Storage Note:**

The current implementation uses **local disk caching** (`./data/cache/`) - this is a **conscious design choice for learning and development**:

**Why Local Disk for Learning:**
- ✅ Simple to understand and debug
- ✅ No external dependencies (Redis, databases)
- ✅ Easy to inspect cache files manually
- ✅ Perfect for single-machine development

**Production Considerations:**

For production deployment (Kubernetes, multiple pods, high availability), you would need:

1. **Centralized Cache (Redis/Memcached)** - Shared across pods, fast (~1ms)
2. **Database Cache (PostgreSQL)** - Persistent, queryable, ACID guarantees
3. **Vector DB as Cache (ChromaDB)** - Single source of truth (recommended for RAG)
4. **Object Storage (S3/GCS)** - Durable, cheap, infinite storage

**Recommended Production Architecture:**
```
Request → Redis (L1: hot cache) → ChromaDB (L2: persistent) → OpenAI API
```

This hybrid approach gives you:
- Fast cache hits (Redis: ~1ms)
- Persistent storage (ChromaDB: already in your stack)
- No duplicate storage (ChromaDB serves both search and cache)

**When to Upgrade:**
- Deploying to containers/Kubernetes
- Need horizontal scaling (multiple pods)
- High availability requirements
- Cache durability is critical

---

## Module Structure

### File Organization

```
src/embedder/
├── __init__.py              # Public API + learning context
├── models.py                # Embedding dataclass
├── cache.py                 # Cache management
├── api_client.py            # OpenAI API integration
├── embedder.py              # Main orchestrator
├── test_embedder_manual.py  # Manual test suite
├── README.md                # This file
└── LLD.md                   # Low-level design diagram
```

### File Summaries

#### `models.py`
Defines the `Embedding` dataclass - the vector representation of a chunk. Each embedding has a chunk_id, 1536-dimensional vector, cached flag, and metadata. Includes properties for validation and numpy conversion.

#### `cache.py`
Implements content-based caching using SHA-256 hashes as keys. Provides functions to load from cache, save to cache, get cache statistics, and clear cache. Cache files stored as JSON in `./data/cache/`.

#### `api_client.py`
Handles OpenAI API communication with error handling and rate limiting. Implements single and batch embedding API calls with exponential backoff retry logic. Validates embedding dimensions (1536).

#### `embedder.py`
Main orchestrator that coordinates the cache-first embedding workflow. Implements `embed_chunk()` for single embeddings and `embed_chunks()` for batch processing with cache checks. Tracks statistics (cache hits, API calls, timing).

#### `test_embedder_manual.py`
Comprehensive manual test suite covering single/batch embedding, cache verification, determinism, dimensions validation, cache statistics, and full pipeline integration.

---

## Key Learning Points

### 1. Deterministic Embeddings Enable Caching

**The Insight:**
Embeddings are generated by a pre-trained, frozen model. There's no randomness in the process. Same input → same tokenization → same forward pass → same output.

**Validation:**
```python
embedding_1 = embed("def hello(): pass")
embedding_2 = embed("def hello(): pass")

np.array_equal(embedding_1.vector, embedding_2.vector)  # True
```

Not just "similar" - **exactly identical** (every single float matches).

### 2. Content-Based Caching Eliminates Manual Invalidation

**Traditional Caching:**
```
cache_key = f"{file_path}:{function_name}"
# Problem: If function changes, need to invalidate cache manually
```

**Content-Based Caching:**
```
cache_key = SHA-256(chunk.content)
# If function changes, content changes, key changes automatically
```

**Benefits:**
- No manual invalidation logic
- No stale cache entries
- Automatic change detection

### 3. Batch Processing Reduces API Overhead

**Overhead Breakdown:**
- HTTP request setup: ~10ms
- Network latency: ~20-50ms
- API processing: ~50-100ms
- HTTP response: ~10ms

**Individual calls:**
```
100 chunks × 90ms overhead = 9 seconds overhead
```

**Batch call:**
```
1 batch × 90ms overhead = 90ms overhead
```

**Savings:** ~8.9 seconds (99% reduction in overhead)

### 4. Exponential Backoff for Rate Limits

**Strategy:**
```
Attempt 1: Call API → Rate limit error
Wait: 1 second
Attempt 2: Call API → Rate limit error
Wait: 2 seconds
Attempt 3: Call API → Success
```

**Why Exponential:**
- Gives API time to recover
- Avoids thundering herd problem
- Graceful degradation

### 5. High-Dimensional Vectors Capture Semantics

**1536 dimensions = 1536 learned features**

Each dimension might capture concepts like:
- Dimension 42: "Is this authentication-related?"
- Dimension 137: "Does this involve data validation?"
- Dimension 891: "Is this a property accessor?"

**From your Test 3:**
- 1536 dims: 7% better accuracy than 384 dims
- More dimensions = more nuanced semantic understanding

---

## Usage Examples

### Basic Usage

```python
from parser import parse_file
from chunker import chunk_functions
from embedder import embed_chunk, embed_chunks

# Parse and chunk
functions = parse_file("models.py")
chunks = chunk_functions(functions)

# Single chunk
embedding = embed_chunk(chunks[0])
print(f"Dimension: {embedding.dimension}")  # 1536
print(f"Cached: {embedding.cached}")  # False (first time)
print(f"Valid: {embedding.is_valid}")  # True

# Batch processing
embeddings = embed_chunks(chunks, batch_size=100)
print(f"Embedded {len(embeddings)} chunks")
```

### Cache Statistics

```python
from embedder import get_cache_stats, get_embedding_stats

# Cache stats
cache_stats = get_cache_stats()
print(f"Cache entries: {cache_stats['total_entries']}")
print(f"Cache size: {cache_stats['total_size_mb']} MB")

# Embedding stats
embedding_stats = get_embedding_stats(embeddings)
print(f"Cache hit rate: {embedding_stats['cache_hit_rate']}%")
print(f"API calls: {embedding_stats['api_count']}")
```

### Verifying Determinism

```python
# Embed same chunk twice
embedding1 = embed_chunk(chunk)
embedding2 = embed_chunk(chunk)

# Check if vectors are identical
import numpy as np
vec1 = np.array(embedding1.vector)
vec2 = np.array(embedding2.vector)

print(f"Identical: {np.array_equal(vec1, vec2)}")  # True
print(f"Max diff: {np.max(np.abs(vec1 - vec2))}")  # 0.0
```

---

## Testing

### Running Manual Tests

```bash
cd /Users/ankur/D/Playground/ai-engineering-journey/week-01/project/rag-code-qa

# Set API key
export OPENAI_API_KEY='your-api-key-here'

# Activate venv
source venv/bin/activate

# Run tests
python3 src/embedder/test_embedder_manual.py
```

### Test Coverage

The manual test suite covers:

1. **Single Chunk Embedding**: Basic API call, verify 1536 dimensions
2. **Batch Embedding**: Process 10+ chunks, verify all succeed
3. **Cache Hit**: Embed twice, second should be instant
4. **Determinism**: Same input → identical vectors
5. **Dimensions**: All embeddings have correct dimension
6. **Cache Statistics**: Track cache size and entries
7. **Integration**: Full parser → chunker → embedder pipeline

### Expected Output

```
✅ TEST 1: Single Chunk Embedding
   Dimension: 1536
   Cached: False
   Time: 0.156s

✅ TEST 2: Batch Embedding (10 Chunks)
   Total embeddings: 10
   Time: 0.892s
   Cache hit rate: 0%

✅ TEST 3: Cache Hit Verification
   First: Cached=False, Time=0.145s
   Second: Cached=True, Time=0.001s
   Cache speedup: 145x faster

✅ TEST 4: Determinism Verification
   Vectors identical: True
   Max difference: 0.0

✅ ALL TESTS COMPLETED
```

---

## Next Steps

After embedding, you'll move to **Component 4: Storage**, which:
1. Takes `Embedding` objects
2. Stores vectors in ChromaDB
3. Enables similarity search via cosine distance
4. Supports incremental updates (upsert)

The embedder's output is perfectly formatted for this next step:
- `vector` is ready for ChromaDB insertion
- `chunk_id` links back to original chunk
- `metadata` enables filtering and citations
