# Experiment 1: Embeddings Validation - Specification

## Purpose

Validate fundamental embeddings concepts from Day 1-3 learning through hands-on Python tests before building the main RAG-Powered Code Q&A project.

## Background

After 3 days of deep learning on:
- Embeddings fundamentals (Day 1)
- Transformer architecture (Day 2-3)
- Working memory vs knowledge (Day 2-3)
- RAG patterns (Day 2-3)

This experiment bridges theory to practice by testing each concept with actual code examples.

## Test Specifications

### Test 1: Pre-trained Models (test_01_pretraining.py)

**Theory**: Embeddings are pre-computed during model training, not calculated at runtime.

**Hypothesis**: Same input text will always return identical embedding vectors.

**Method**:
- Call OpenAI embeddings API twice with same text
- Compare vectors element-by-element
- Verify exact match

**Success Criteria**: embedding_1 == embedding_2 (bitwise identical)

**Implication for Project**: Safe to cache embeddings; no need to refresh for unchanged code

---

### Test 2: Semantic Clustering (test_02_semantic_clustering.py)

**Theory**: Latent dimensions compress similar concepts together in embedding space.

**Hypothesis**: Code doing the same thing (e.g., add functions) will cluster together regardless of language.

**Method**:
- Embed: Python add function, JavaScript add function, Python multiply function
- Calculate pairwise cosine similarities
- Compare: similarity(py_add, js_add) vs similarity(py_add, py_multiply)

**Success Criteria**: similarity(py_add, js_add) > similarity(py_add, py_multiply)

**Implication for Project**: Multi-language code search will work; semantic meaning > syntax

---

### Test 3: Dimensionality (test_03_dimensionality.py)

**Theory**: More dimensions = more nuance but slower; fewer dimensions = less nuance but faster.

**Hypothesis**: Full 1536-dim embeddings capture finer semantic detail than truncated 384-dim.

**Method**:
- Get 1536-dim embedding for complex code
- Truncate to 384 dims
- Compare similarity scores for related code with both full and truncated

**Success Criteria**: similarity_full >= similarity_truncated

**Implication for Project**: Using text-embedding-3-small (1536 dims) is worth the storage cost

---

### Test 4: Distance Metrics (test_04_distance_metrics.py)

**Theory**: Cosine similarity (angle-based) is better than Euclidean distance (magnitude-based) for high-dimensional embeddings.

**Hypothesis**: Cosine similarity handles length variations better for semantically similar code.

**Method**:
- Embed: short code snippet, verbose equivalent, completely different code
- Calculate both Euclidean distance and cosine similarity
- Compare metrics' ability to distinguish semantic similarity

**Success Criteria**: Cosine correctly identifies short/verbose as similar; Euclidean affected by length

**Implication for Project**: Use cosine similarity in ChromaDB (default)

---

### Test 5: Semantic Relationships (test_05_relationships.py)

**Theory**: Embeddings capture domain-specific relationships (like king-man+woman=queen).

**Hypothesis**: Within-language concepts cluster more strongly than cross-language.

**Method**:
- Embed Python concepts (list comp, dict comp, generator)
- Embed JavaScript concepts (map, filter, reduce)
- Calculate average within-language vs cross-language similarity

**Success Criteria**: avg_py_similarity > avg_cross_similarity AND avg_js_similarity > avg_cross_similarity

**Implication for Project**: Semantic search can distinguish language paradigms

---

### Test 6: Chunking Strategies (test_06_chunking.py)

**Theory**: Semantic units (functions) are better chunks than arbitrary fixed-size splits.

**Hypothesis**: Function-level chunks retrieve more relevant code than fixed-size chunks.

**Method**:
- Create code file with 3 functions
- Chunk by: (1) function boundaries, (2) fixed 100-char splits
- Query: "authenticate user credentials"
- Compare retrieval quality

**Success Criteria**: best_semantic_score > best_fixed_score AND top result is authenticate_user function

**Implication for Project**: Validates config.yaml setting: chunk_strategy: by_function

---

### Test 7: Working Memory (test_07_working_memory.py)

**Theory**: Retrieved chunks = working memory for LLM; quality retrieval prevents hallucinations.

**Hypothesis**: Relevant chunks score higher than irrelevant; top-k retrieval provides good working memory.

**Method**:
- Create relevant chunks (calculate_tax, apply_discount)
- Create irrelevant chunks (connect_database, log_error)
- Query: "How do I calculate price after tax and discount?"
- Retrieve top-k

**Success Criteria**: max(relevant_scores) > max(irrelevant_scores) AND top-k = relevant chunks

**Implication for Project**: Foundation of RAG architecture; validates retrieval → working memory pattern

---

## Deliverables

1. **7 Python test files** - One per concept, with clear comments
2. **run_all.py** - Orchestrator to run all tests in sequence
3. **results.md** - Documented findings with theory-practice mapping
4. **Actionable decisions** - Extracted from results for rag-code-qa/

## Timeline

- Setup: 5 min
- Implement tests: 25 min (3-4 min each)
- Run and observe: 10 min
- Document: 5 min
- **Total: 45 minutes**
