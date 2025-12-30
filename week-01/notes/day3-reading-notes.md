# Day 3: Hands-On Experimentation - Running Notes

**Date**: Monday, December 30, 2025
**Focus**: Validating Day 1-2 learning through practical experiments
**Strategy**: Running notes during experiments, will synthesize in `experiments.md` and daily-log at end of day

---

## Foundation Coming Into Day 3

### From Day 1 (RNN/LSTM/Transformer Deep Dives)
- How embeddings represent semantic meaning
- Why attention mechanism solves long-distance dependency
- How transformers process sequences in parallel
- Concept of learned weights through backpropagation

### From Day 2 (Deep Dive Video - Pretraining & Post-Training)
- Complete LLM pipeline: Text → Tokens → Embeddings → 12 Transformer Layers → Output
- Tokenization via BPE (Byte Pair Encoding)
- Query/Key/Value attention mechanism with softmax
- Post-training teaches helpful assistant behavior
- Base model = internet simulator → Post-training = helpful labeler simulator

### Why Day 3 Matters
- Convert theory to practical understanding
- See embeddings, similarity metrics, storage, and chunking work in real code
- Build intuition for decisions in Day 4-5 implementation

---

## EXPERIMENT 1: Embeddings Generation

**Goal**: Generate embeddings with OpenAI API and observe semantic properties

### Setup
- [ ] Create `project/` directory structure
- [ ] Set up Python environment
- [ ] Install dependencies: openai, chromadb, numpy, pandas
- [ ] Configure OpenAI API key

### Testing

**Test 1a: Simple Words**
- Generate embeddings for: cat, dog, car, tree, happy, sad
- Check dimensions (expect 1536 for text-embedding-3-small)
- Observation notes:

**Test 1b: Semantic Relationships**
- Generate: "the big dog ran", "a large canine sprinted"
- Observation: Do synonymous phrases have similar embeddings?
- Notes:

**Test 1c: Code Snippets**
- Generate embeddings for:
  - "function calculateSum(arr) { return arr.reduce((a,b) => a+b, 0); }"
  - "def sum_array(arr): return sum(arr)"
  - "SELECT SUM(column) FROM table"
- Observation: How similar are semantically related code in different languages?
- Notes:

### Findings
- Embedding dimensions verified: ___
- Similarity patterns observed:
- Unexpected behavior:

---

## EXPERIMENT 2: Vector Similarity & Distance Metrics

**Goal**: Understand how similarity metrics work and validate mathematical intuition

### Implementation Notes

**Part 1: Cosine Similarity from Scratch**
```python
# Implementation log:
- Define cosine_similarity(v1, v2)
- Test with known vectors
- Verify against numpy/scipy
```

**Part 2: Distance Metric Comparison**
- Test pairs:
  - (cat embedding, dog embedding)
  - (cat embedding, car embedding)
  - (function, method) - from code snippets

- Compare:
  - Cosine distance: ___
  - Euclidean distance: ___
  - Observations: ___

### Key Questions to Answer
- Does cosine distance align with semantic intuition?
- Why is cosine often better than Euclidean for embeddings?
- Which distance metric should we use for semantic search?

### Findings
- Cosine vs Euclidean comparison results:
- Why cosine wins for high-dimensional embeddings:
- Practical implications for Day 4-5 implementation:

---

## EXPERIMENT 3: ChromaDB Storage & Retrieval

**Goal**: See vector database retrieval work end-to-end

### Setup
- [ ] Create ChromaDB collection locally
- [ ] Prepare 10 code snippets from a real codebase (or use examples)
- [ ] Generate embeddings for each
- [ ] Store in ChromaDB

### Code Snippets (populate these)
```
Snippet 1: [description]
Snippet 2: [description]
...
```

### Retrieval Tests
**Query 1**: "How do I match job candidates to positions?"
- Retrieved snippets (ranked):
- Relevance assessment (1-5):
- Notes:

**Query 2**: "What algorithm sorts this list?"
- Retrieved snippets:
- Relevance assessment:
- Notes:

**Query 3**: "Show me code that fetches data from API"
- Retrieved snippets:
- Relevance assessment:
- Notes:

### Variations
- [ ] Test with top_k=1, 3, 5, 10
- [ ] Observations about result quality vs. quantity:

### Findings
- ChromaDB effectiveness:
- Quality of semantic search results:
- Optimal number of results to return:
- Relevance vs. performance tradeoff:

---

## EXPERIMENT 4: Chunking Strategies

**Goal**: Understand how different chunking approaches affect search quality

### Setup
- [ ] Select 3 real functions from codebase
- [ ] Implement 3 chunking strategies:
  1. **By Function**: Treat entire function as one chunk
  2. **By Semantic Blocks**: Split into 5-10 line semantic units
  3. **Fixed Line Count**: Split every 20 lines

### Test Functions
```
Function 1: [description and code]
Function 2: [description and code]
Function 3: [description and code]
```

### Strategy Comparison

**Strategy 1: By Function**
- Number of chunks: ___
- Chunk sizes: ___
- Query test results:
  - Query 1: ___ relevant chunks returned
  - Query 2: ___ relevant chunks returned
- Pros/Cons:

**Strategy 2: By Semantic Blocks**
- Number of chunks: ___
- Chunk sizes: ___
- Query test results:
  - Query 1: ___ relevant chunks returned
  - Query 2: ___ relevant chunks returned
- Pros/Cons:

**Strategy 3: Fixed Line Count (20 lines)**
- Number of chunks: ___
- Chunk sizes: ___
- Query test results:
  - Query 1: ___ relevant chunks returned
  - Query 2: ___ relevant chunks returned
- Pros/Cons:

### Key Questions
- Which strategy returns most relevant results?
- Which strategy has best precision/recall?
- What happens when a semantic unit spans chunk boundary?
- Cost implications (number of embeddings needed)?

### Findings
- Best chunking strategy: ___ because ___
- Surprising discovery: ___
- Tradeoff to optimize for Week 1 project: ___

---

## Synthesis & Architecture Decisions

### What I Discovered About Theory in Practice
- Day 2 embedding concept (768 dimensions) becomes: ___
- Attention mechanism understanding manifests as: ___
- Post-training's helpfulness concept seen in: ___

### Edge Cases Identified
- Edge case 1: ___
- Edge case 2: ___
- How to handle in Day 4 implementation:

### Decisions for Day 4 Implementation
1. Chunking strategy: ___
2. Similarity metric: ___
3. Storage approach: ___
4. Result ranking: ___

### Questions for Deeper Research
- Question 1: ___
- Question 2: ___
- Question 3: ___

---

## Code Quality Notes
- Setup quality: ___
- Code clarity: ___
- Testing approach: ___
- Documentation needs: ___

---

## Time Log
- Setup: ___ minutes
- Experiment 1 (Embeddings): ___ minutes
- Experiment 2 (Similarity): ___ minutes
- Experiment 3 (ChromaDB): ___ minutes
- Experiment 4 (Chunking): ___ minutes
- Synthesis & Notes: ___ minutes
- **Total**: ___ minutes

---

## Next Steps for Day 4
Based on today's findings:
1. Focus implementation on: ___
2. Skip/defer: ___
3. Architecture approach: ___
4. Potential bottlenecks: ___
