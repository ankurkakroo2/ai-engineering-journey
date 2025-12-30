# Day 4: Hands-On Experimentation - Running Notes

**Date**: Tuesday, December 31, 2025
**Focus**: Validating Day 1-3 learning through practical experiments with embeddings, similarity metrics, storage, and chunking strategies
**Strategy**: Detailed running notes during experiments, will synthesize findings and make architecture decisions

⚠️ **What This Is**: Hands-on experimentation notes—capturing observations, "aha moments," code behavior, and connections as they happen. Not just templates, but filled with actual findings.

---

## Foundation Coming Into Day 4

### From Day 1-3: Complete Learning Foundation

**Deep Dive Video Content** (See Date2-3-Deep-Dive-Notes.md):
- Checkpoint 1: Pretraining flow (text → tokens → 12 layers → predictions)
- Checkpoint 2: Post-training revolution (base model → helpful assistant)
- Checkpoint 3: Hallucinations, Tool Use, Knowledge vs. Working Memory, Models Need Tokens to Think
- Checkpoint 4: (Remaining sections from video)

**Key Insights to Apply**:
- **Understanding why CoT works** (multiple passes through layers) informs chunking decisions
- **Understanding hallucinations** (no "I don't know" state) explains why external tools and current context matter
- **Understanding knowledge vs. working memory** (parameters vs. current prompt) guides how to retrieve and present code chunks
- **Tokens ARE thinking**: Intermediate steps matter for accuracy

### Why Day 4 Matters: Theory Meets Practice

**The Connection You're Building:**
- Day 1 taught: How transformers work (mechanisms)
- Day 2-3 taught: How LLMs are trained (pretraining → post-training → alignment)
- Day 4 proves: All of this actually works in real code

**Practical Questions You'll Answer:**
- Do similar semantic meanings really have similar embeddings? (Test with real API)
- Does cosine similarity actually capture intuition better than Euclidean? (Compare metrics)
- How much context does a vector database actually retrieve correctly? (Test ChromaDB)
- Which chunking strategy actually works best? (Compare strategies)

**Critical Realization:**
- Understanding why CoT works (multiple passes through layers) informs chunking decisions
- Understanding "jagged intelligence" (models better at some domains than others) explains why certain code chunks retrieve better than others
- Understanding hallucinations (no "I don't know") explains why you need external tools and current context (working memory)

---

## EXPERIMENT 1: Embeddings Generation - Validating Semantic Meaning

**Goal**: Generate real embeddings with OpenAI API and validate that similar semantic concepts actually have similar vectors.

**Key Question**: Do embeddings from theory actually capture semantic meaning in practice?

### Setup & Configuration
- [ ] Create `project/` directory structure
- [ ] Set up Python environment and virtual environment
- [ ] Install dependencies: openai, numpy, chromadb
- [ ] Configure OpenAI API key securely
- [ ] Verify API access works

### Test 1A: Simple Words - Validating Semantic Relationships

**Theory to Test**:
From Day 1: "cat" and "dog" should have similar embeddings (both animals). "cat" and "car" should be different (different semantic domains).

**Methodology**:
```
Words to embed: cat, dog, car, tree, happy, sad, run, walk
Expected:
- cat ↔ dog = high similarity
- cat ↔ car = low similarity
- happy ↔ sad = medium-high (opposites, both emotions)
- run ↔ walk = high similarity (both movement verbs)
```

**What to Measure**:
- Embedding dimensions (verify 1536 for text-embedding-3-small)
- Similarity matrix between all word pairs
- Do similarities match intuition?

**Observations to Capture**:
- Which similarities surprised you?
- Any unexpected high/low similarities?
- Do distance metrics confirm intuition?

### Test 1B: Phrases & Context - Semantic Relationships Hold at Scale

**Theory to Test**:
Semantically equivalent phrases should have similar embeddings even if word-for-word different.

**Test Phrases**:
```
Pair 1: "the big dog ran fast" vs "a large canine sprinted"
Pair 2: "software engineer writes code" vs "developer programs applications"
Pair 3: "machine learning algorithm" vs "AI model training process"
```

**Observations to Capture**:
- Do semantically equivalent phrases cluster together?
- How similar is "very similar meaning" vs "identical meaning"?
- Confidence in similarity scores?

### Test 1C: Code Snippets - Can Embeddings Capture Code Semantics?

**Theory to Test**:
From Day 4 goal: Can embeddings capture semantic meaning of code across languages?

**Code Examples**:
```python
# Python: Sum an array
def sum_array(arr):
    return sum(arr)

# JavaScript: Sum an array
function sumArray(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

# SQL: Sum a column
SELECT SUM(amount) FROM transactions;
```

**Observations to Capture**:
- Do all three "sum" examples cluster together semantically?
- How similar are they despite different syntaxes?
- Implications for semantic code search?

### Findings & Synthesis
- [ ] Embedding dimensions confirmed: ___
- [ ] Similarity patterns observed (write down 3-5 key observations):
- [ ] Unexpected behaviors/anomalies:
- [ ] Confidence in embeddings capturing semantics: High / Medium / Low
- [ ] Implications for core implementation:

---

## EXPERIMENT 2: Vector Similarity Metrics - Cosine vs. Euclidean

**Goal**: Understand why cosine similarity is preferred for embeddings, validate through practical comparison.

**Key Question**: Does cosine distance actually match semantic intuition better than Euclidean?

### Theory Recap (from Day 1)
- **Cosine**: Angle between vectors, ignores magnitude
- **Euclidean**: Straight-line distance, includes magnitude
- **Hypothesis**: Cosine better for high-dimensional embeddings

### Implementation: Building from Scratch

**Step 1: Cosine Similarity Implementation**
```
Manual implementation to understand (don't just use numpy):
1. Compute dot product of two vectors
2. Compute magnitude of each vector
3. Cosine = dot_product / (magnitude1 * magnitude2)
```

**Step 2: Euclidean Distance Implementation**
```
1. Element-wise difference of vectors
2. Square each difference
3. Sum all squared differences
4. Take square root
```

**Step 3: Compare Against Library**
- Verify your implementation matches numpy/scipy
- Build confidence in understanding

### Test Comparisons

**Test Pair 1: Semantically Very Similar**
- Words: "cat" and "feline"
- Expected: Very high cosine, moderate-high Euclidean
- Actual - Cosine: ___ | Euclidean: ___
- Observations:

**Test Pair 2: Semantically Different**
- Words: "cat" and "car"
- Expected: Low cosine, large Euclidean
- Actual - Cosine: ___ | Euclidean: ___
- Observations:

**Test Pair 3: Opposite Meaning**
- Words: "happy" and "sad"
- Expected: Negative cosine (opposite directions), still large Euclidean
- Actual - Cosine: ___ | Euclidean: ___
- Observations:

### Key Questions to Answer
- [ ] Does cosine better capture semantic intuition? How?
- [ ] When might Euclidean be better? When worse?
- [ ] Why do you think cosine is standard for embeddings?
- [ ] How would magnitude affect search results?

### Findings & Synthesis
- [ ] Why cosine wins for embeddings (explain in your own words):
- [ ] Trade-offs between metrics:
- [ ] Decision for core implementation: Use ___ because ___

---

## EXPERIMENT 3: ChromaDB Storage & Semantic Retrieval

**Goal**: See vector database retrieval work end-to-end, validate that semantic search actually returns relevant results.

**Key Question**: Does storing embeddings + searching by similarity actually retrieve semantically relevant code?

### Setup: Creating Vector Database

**Step 1: Prepare Code Snippets**
- [ ] Source 10-15 diverse code snippets (Python, JavaScript, SQL mix)
- [ ] Document what each does
- [ ] Manually assign semantic categories

**Step 2: Generate & Store Embeddings**
- [ ] Generate embeddings for each snippet
- [ ] Store in ChromaDB collection
- [ ] Verify storage structure

### Retrieval Tests: Does Semantic Search Actually Work?

**Query 1: "How do I fetch data from a database?"**
- Code snippets you expect: SQL SELECT, Python database calls, JavaScript fetch API
- Retrieved snippets (ranked):
  1. ___ (relevance 1-5: ___)
  2. ___ (relevance: ___)
  3. ___ (relevance: ___)
- Observations: How well did semantic search understand "fetch data from database"?

**Query 2: "What does this function do to filter an array?"**
- Code snippets you expect: Array filter operations, WHERE clauses, conditional logic
- Retrieved snippets (ranked):
  1. ___ (relevance: ___)
  2. ___ (relevance: ___)
  3. ___ (relevance: ___)
- Observations: How well did it understand "filter"?

**Query 3: "Show me code that loops through items"**
- Code snippets you expect: for loops, while loops, map/reduce, SQL iteration
- Retrieved snippets (ranked):
  1. ___ (relevance: ___)
  2. ___ (relevance: ___)
  3. ___ (relevance: ___)
- Observations: Did it find all loop variations?

### Variations: How Many Results Is Optimal?

**Test with top_k values**:
- top_k=1: Returns ___ snippets
  - Precision (relevant/returned): __%
  - Would user find what they need? Yes/No

- top_k=3: Returns ___ snippets
  - Precision: __%
  - Better coverage? Yes/No

- top_k=5: Returns ___ snippets
  - Precision: __%
  - Too much noise? Yes/No

### Key Questions
- [ ] What's the optimal top_k for your use case?
- [ ] How often did search miss relevant code?
- [ ] When did it return irrelevant results?
- [ ] How does query phrasing affect results?

### Findings & Synthesis
- [ ] ChromaDB effectiveness: Works well / Works okay / Needs improvement
- [ ] Semantic search quality: High precision / Some false positives / Many false negatives
- [ ] Optimal number of results: top_k = ___
- [ ] Query phrasing impact: Strong / Moderate / Weak
- [ ] Relevance vs. Performance tradeoff: Decided to optimize for ___

---

## EXPERIMENT 4: Chunking Strategies - Which Works Best?

**Goal**: Empirically determine which chunking strategy returns most relevant code snippets.

**Key Question**: Does chunking strategy really affect semantic search quality significantly?

### Theory to Validate

From Day 2-3: "Chunking strategy is critical—impacts both storage and search quality"

**Three Strategies to Compare**:

1. **Strategy A: Chunk by Function**
   - Treat entire function as one chunk
   - Chunk size: variable (1 function = 1 chunk)
   - Semantically complete units

2. **Strategy B: Chunk by Semantic Blocks**
   - Split function into logical 5-10 line blocks
   - Chunk size: ~5-10 lines
   - Preserves related operations together

3. **Strategy C: Chunk by Fixed Line Count**
   - Split every 20 lines regardless of logic
   - Chunk size: exactly 20 lines
   - May split semantic units

### Setup: Prepare Test Functions

**Function 1: Data Validation**
```
Lines: ~30
Semantic blocks: 3-4 blocks
Chunks by line: 2 chunks
```

**Function 2: Complex Algorithm**
```
Lines: ~50
Semantic blocks: 5+ blocks
Chunks by line: 3 chunks
```

### Strategy Comparison

**STRATEGY A: By Function**
- Number of chunks created: ___
- Average chunk size: ___ lines
- Embeddings generated: ___

Query test results:
- Query: "How does this validate user input?"
  - Chunks returned: ___ (all relevant? Yes/No)
  - Precision: ___%

- Query: "Show me error handling"
  - Chunks returned: ___
  - Precision: ___%

Pros:
- Semantically complete units
- Easy to understand full context

Cons:
- Large chunks might contain irrelevant code
- All-or-nothing retrieval

**STRATEGY B: By Semantic Blocks**
- Number of chunks created: ___
- Average chunk size: ___ lines
- Embeddings generated: ___

Query test results:
- Query: "How does this validate user input?"
  - Chunks returned: ___
  - Precision: ___%
  - More/fewer results than Strategy A? Why?

- Query: "Show me error handling"
  - Chunks returned: ___
  - Precision: ___%

Pros:
- Focused chunks (less noise)
- Better precision

Cons:
- Manual effort to define blocks
- More embeddings to store

**STRATEGY C: Fixed Line Count**
- Number of chunks created: ___
- Average chunk size: exactly 20 lines
- Embeddings generated: ___

Query test results:
- Query: "How does this validate user input?"
  - Chunks returned: ___
  - Precision: ___%
  - Issues with split functions? Examples:

- Query: "Show me error handling"
  - Chunks returned: ___
  - Precision: ___%

Pros:
- Consistent sizes
- Easy to implement

Cons:
- May split important operations
- Chunks might be incoherent
- Examples of broken semantics:

### Comparative Analysis

**Precision Comparison** (relevant chunks / total chunks):
- Strategy A: ___%
- Strategy B: ___%
- Strategy C: ___%

**Coverage Comparison** (retrieved vs. ideal):
- Strategy A: Comprehensive / Moderate / Sparse
- Strategy B: Comprehensive / Moderate / Sparse
- Strategy C: Comprehensive / Moderate / Sparse

**Cost Comparison** (number of embeddings, storage):
- Strategy A: ___ embeddings
- Strategy B: ___ embeddings (__ % more/less than A)
- Strategy C: ___ embeddings

### Key Questions
- [ ] Which strategy won? Why?
- [ ] How important is chunking strategy? (Critical / Important / Somewhat important)
- [ ] What edge cases broke each strategy?
- [ ] Would hybrid approach work better?

### Findings & Synthesis
- [ ] Best chunking strategy: **Strategy _** because ___
- [ ] Reasoning: Balances precision (_%), completeness (_%), and cost (___ embeddings)
- [ ] Trade-off you optimized for: Relevance / Speed / Cost
- [ ] Will implement using: ___

---

## SYNTHESIS & DECISION MAKING

### What Theory Predicted vs. What You Observed

**Embeddings & Semantics**:
- Theory said: Similar meanings → similar vectors
- You observed: ___
- Confidence: High / Medium / Low

**Similarity Metrics**:
- Theory said: Cosine > Euclidean for high-dim embeddings
- You observed: ___
- Confidence: High / Medium / Low

**Semantic Search**:
- Theory said: Vector DB retrieval = semantic relevance
- You observed: ___
- Confidence: High / Medium / Low

**Chunking Impact**:
- Theory said: Chunking strategy significantly affects results
- You observed: ___
- Confidence: High / Medium / Low

### Critical Insights for Core Implementation

1. **Embedding Dimension Impact**: ___
2. **Similarity Metric Choice**: Will use ___ because ___
3. **Chunking Strategy**: Will use ___ because ___
4. **Context Window**: Retrieved ___ chunks per query, optimal for ___
5. **Hallucination Risk**: Mitigated by ___

### Architecture Decisions Made

- [ ] Embedding model: text-embedding-3-small (1536-dim) or other?
- [ ] Similarity metric: Cosine
- [ ] Vector database: ChromaDB (local) or other?
- [ ] Chunking strategy: By ___ because ___
- [ ] Top-K results: Retrieve ___ chunks per query
- [ ] Fallback strategy: If search returns nothing, ___

### Edge Cases & Gotchas Discovered

1. Edge case: ___
   - Impact: ___
   - Mitigation: ___

2. Edge case: ___
   - Impact: ___
   - Mitigation: ___

### Time & Performance Notes

- Experiment 1 (Embeddings): ___ minutes
- Experiment 2 (Similarity): ___ minutes
- Experiment 3 (ChromaDB): ___ minutes
- Experiment 4 (Chunking): ___ minutes
- Analysis & notes: ___ minutes
- **Total: ___ minutes**

### Confidence & Readiness for Implementation

- Confidence in understanding: High / Medium / Low
- Ready to build? Yes / Need more investigation
- Key uncertainties remaining: ___

---

*These notes document the hands-on validation of Day 1-3 theory. You're not just learning how embeddings work—you're seeing them work in real code. This bridges the gap between understanding and building.*
