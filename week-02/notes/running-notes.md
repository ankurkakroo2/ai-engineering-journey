# Week 2: Vector Databases - Running Notes

**Author**: Ankur  
**Week**: January 13-19, 2026  
**Goal**: Deep understanding of vector database internals

---

## Learning Philosophy

This document captures detailed learnings as I progress through Week 2. Following the same approach as Week 1, I'll document:
- **Precise terminology**: Exact definitions, no vague language
- **Timing clarity**: WHEN things happen (index-time vs query-time)
- **Concrete examples**: End-to-end walkthroughs with actual numbers
- **Connections**: How concepts build on Week 1 learnings
- **Questions answered**: Each section ends with "Can I explain X?"

---

## Day 1: Vector Database Fundamentals

**Date**: January 13, 2026  
**Reading**: Pinecone "What is a Vector Database?" + HNSW Deep Dive  
**Time**: 3-4 hours

---

### Session 1: The Problem Vector Databases Solve

**Resource**: [What is a Vector Database? - Pinecone](https://www.pinecone.io/learn/vector-database/)

#### Why do vector databases exist?

**Context from Week 1**:
- Week 1 taught me: Embeddings are pre-computed 1536-dimensional vectors
- I learned: OpenAI API generates embeddings deterministically
- I built: ChromaDB for local storage and basic retrieval

**NEW PROBLEM Week 2 addresses**:

[Document your insights here as you read]

**Key insight 1**: 

**Key insight 2**:

**Key insight 3**:

---

#### Vector Database vs Vector Index - The 7 Differences

[As you read the article, document each difference with your understanding]

**1. Data Management**:
- Vector Index: 
- Vector Database:
- **Why this matters**: 

**2. Metadata Storage and Filtering**:
- Vector Index:
- Vector Database:
- **Why this matters**:

**3. Scalability**:
- Vector Index:
- Vector Database:
- **Why this matters**:

**4. Real-time Updates**:
- Vector Index:
- Vector Database:
- **Timing precision**: 

**5. Backups and Collections**:
- Vector Index:
- Vector Database:
- **Why this matters**:

**6. Ecosystem Integration**:
- Vector Index:
- Vector Database:
- **Examples**:

**7. Security and Access Control**:
- Vector Index:
- Vector Database:
- **Why this matters**:

---

#### The Pipeline: How Vector Databases Work

[Document the 3-stage pipeline with timing annotations]

**Stage 1: Indexing**
- **What happens**: 
- **WHEN this happens**: 
- **Algorithms used**: 
- **Cost**: 

**Stage 2: Querying**
- **What happens**:
- **WHEN this happens**:
- **How similarity is computed**:
- **Cost**:

**Stage 3: Post-Processing**
- **What happens**:
- **WHEN this happens**:
- **Why rescoring is needed**:
- **Cost**:

---

#### Serverless Vector Databases

[Document the 3 key problems serverless solves]

**Problem 1: Separation of Storage from Compute**
- **Challenge**: 
- **Solution**: 
- **Cost impact**: 

**Problem 2: Multitenancy**
- **Challenge**:
- **Solution**:
- **Example scenario**:

**Problem 3: Freshness**
- **Challenge**:
- **Solution (Freshness Layer)**:
- **How it works**:

**Connection to Week 1**:
- Week 1 caching insight: Embeddings are deterministic → safe to cache
- Week 2 serverless insight: 

---

### Checkpoint Questions (Session 1)

**Q1: When is a vector index built? When is it traversed?**
Answer: 

**Q2: What's stored WHERE? (RAM vs SSD vs both)**
Answer:

**Q3: Why can't traditional databases handle vector search efficiently?**
Answer:

**Q4: How does this connect to Week 1's learnings?**
Answer:

---

### Session 2: HNSW Algorithm Deep Dive

**Resource**: [HNSW Explained - Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/)

#### Foundations: Probability Skip List

[Document the multi-layer concept]

**Core idea**:

**Analogy to understand**:

**Key properties**:
1.
2.
3.

**Search process**:
- Start at: 
- Move through: 
- Stop when: 

**Drawing** (describe your paper sketch):

---

#### Navigable Small World (NSW) Graphs

[Document how NSW works before HNSW's improvements]

**Graph structure**:
- Vertices represent: 
- Edges connect: 
- Friend list: 

**Search process - Greedy Routing**:
- Step 1 (Entry): 
- Step 2 (Traverse): 
- Step 3 (Stop): 

**Two phases**:
1. **Zoom-out phase**: 
   - What: 
   - Why: 
   
2. **Zoom-in phase**:
   - What: 
   - Why: 

**Problem with NSW**:

---

#### HNSW: The Multi-Layer Innovation

[Document how HNSW improves upon NSW]

**Key innovation**:

**Layer structure**:
- **Layer 0 (bottom)**: 
- **Layer 1-N (middle)**: 
- **Layer N (top)**: 

**Why this works**:
1.
2.
3.

**Layer assignment**:
- Probability function: 
- Level multiplier (m_L): 
- Result: 

---

#### HNSW Graph Construction

[Document the build process with precise timing]

**Step 1: Insert a new vector**
- WHEN: 
- Probability calculation: 
- Assigned to layer: 

**Step 2: Find neighbors**
- Algorithm: 
- Parameter: M (number of neighbors)
- Parameter: ef_construction (candidates explored)
- **Timing**: This happens at [index-time / query-time]

**Step 3: Create edges**
- Connect to: 
- Bidirectional: 
- Constraints: M_max, M_max0

**Step 4: Repeat**
- For each new vector
- Graph evolves over time

---

#### HNSW Graph Traversal (Search)

[Document the query process with concrete example]

**Concrete example: Search through 1M vectors**

**Initial state**:
- Entry point: Vertex V_118295 (pre-determined)
- Query vector: Q
- Goal: Find 10 nearest neighbors

**Traversal steps**:

**Layer 3 (entry)**:
- At vertex: 
- Check neighbors: [count]
- Best match: 
- Distance calculations: [count]
- Next action: 

**Layer 2**:
- At vertex: 
- Check neighbors: [count]
- Distance calculations: [total so far]
- Next action: 

**Layer 1**:
- At vertex:
- Check neighbors: [count]
- Distance calculations: [total so far]
- Next action:

**Layer 0 (final)**:
- At vertex:
- Check neighbors: [count]
- Distance calculations: [total so far]
- Return: Top 10 results

**Total cost**:
- Distance calculations: ~100 (vs 1M brute-force!)
- Time: ~5ms (vs ~500ms brute-force)
- Speedup: 100x

---

#### HNSW Parameters

[Document each parameter with precise meaning]

**Parameter: M (edges per vertex)**
- **What it controls**: 
- **Effect on recall**: 
- **Effect on memory**: 
- **Effect on speed**: 
- **Typical values**: 8, 16, 32, 64
- **Optimal choice**: 

**Parameter: ef_construction**
- **What it controls**: 
- **WHEN it's used**: 
- **Effect on recall**: 
- **Effect on build time**: 
- **Typical values**: 100, 200, 400

**Parameter: ef_search**
- **What it controls**: 
- **WHEN it's used**: 
- **Effect on recall**: 
- **Effect on query time**: 
- **Typical values**: 50, 100, 200

**Parameter: M_max and M_max0**
- **What they control**: 
- **How they're set**: 
- **Why M_max0 is different**: 

---

### Checkpoint Questions (Session 2)

**Q1: Why does HNSW use multiple layers?**
Answer (in my own words):

**Q2: What happens at each layer during traversal?**
Answer:

**Q3: How does M parameter affect recall vs memory?**
Answer:

**Q4: When is ef_construction used vs ef_search?**
Answer:

**Q5: Draw the HNSW structure on paper - can you trace a query?**
Answer: [Yes/No - describe your drawing]

---

## Day 2: Quantization & Storage Trade-offs

**Date**: January 14, 2026  
**Reading**: Product Quantization + Timescale DiskANN  
**Time**: 3-4 hours

---

### Session 1: Quantization Fundamentals

**Resource**: [Product Quantization - Pinecone](https://www.pinecone.io/learn/series/faiss/product-quantization/)

#### Why Quantize? The Memory Problem

[Document the scale problem]

**Memory calculation**:
- 1 vector: 1536 dimensions × 4 bytes (float32) = _____ bytes = _____ KB
- 1K vectors: _____ KB
- 10K vectors: _____ KB = _____ MB
- 100K vectors: _____ MB
- 1M vectors: _____ MB = _____ GB
- 10M vectors: _____ GB

**At scale**:
- 10M vectors = _____ GB RAM needed
- Cost: AWS RAM = $8/GB-month
- Monthly cost: $_____ 

**The question**: Can we compress vectors without losing too much accuracy?

---

#### Product Quantization (PQ) - Deep Dive

[Document each step with timing precision]

**Step 1: Splitting**
- Original vector: 1536 dimensions
- Split into: _____ subvectors
- Each subvector: _____ dimensions
- **WHEN this happens**: 

**Step 2: Training (Codebook Creation)**
- For each subvector position:
  - Run K-means clustering
  - K = _____ (typical: 256)
  - Result: _____ centroids
- Total codebooks: _____ (one per subvector position)
- **WHEN this happens**: 
- **How long it takes**: 

**Step 3: Encoding**
- For each vector:
  - For each subvector:
    - Find nearest centroid
    - Replace subvector with centroid ID (8-bit integer)
- Original: _____ bytes
- Encoded: _____ bytes (_____ subvectors × 1 byte)
- Compression ratio: _____x
- **WHEN this happens**: 

**Step 4: Querying (Distance Calculation)**
- Query vector: Also split and encoded
- Distance calculation: Use pre-computed distances between centroids
- Faster because: 
- **WHEN this happens**: 

**PQ Compression Analysis**:
- Original size: 6 KB per vector
- PQ size: 96 bytes per vector
- Compression: 64x
- Recall loss: 5-10%
- Speed gain: 2x (fewer bytes to process)

---

#### Binary Quantization (BQ) - Simple but Lossy

[Document the simple thresholding approach]

**The Rule**:
- For each dimension:
  - If value > 0.0 → bit = 1
  - If value ≤ 0.0 → bit = 0

**Example**:
```
Original vector: [0.5, -0.3, 0.8, -0.1, 0.2]
Binary vector:   [1,    0,    1,    0,    1  ]
```

**Distance function**:
- XOR (exclusive OR): Counts bit differences
- Why XOR works: 

**Compression**:
- Original: 1536 dims × 4 bytes = 6144 bytes
- Binary: 1536 bits ÷ 8 = 192 bytes
- Compression: 32x
- Recall loss: 10-15%
- Speed gain: 3x (XOR is very fast)

**Geometric interpretation**:
- Binary vector divides space into quadrants
- Each dimension = one plane
- XOR distance = number of planes crossed
- [Sketch this on paper]

---

#### Scalar Quantization - Best Balance

[Document the middle-ground approach]

**The Rule**:
- Map float32 range to int8 range [0, 255]
- Find min and max in dimension
- Normalize: (value - min) / (max - min) × 255

**Example**:
```
Original dimension values: [-0.5, 0.0, 0.5, 1.0]
Min = -0.5, Max = 1.0, Range = 1.5

Quantized:
-0.5 → 0
 0.0 → 85
 0.5 → 170
 1.0 → 255
```

**Compression**:
- Original: 4 bytes per dimension
- Quantized: 1 byte per dimension
- Compression: 4x
- Recall loss: 2-5%
- Speed gain: 1.5x

---

#### Comparison Table

[Create comparison after reading all three]

| Method | Compression | Recall Loss | Speed | Memory (1M vectors) | Best For |
|--------|-------------|-------------|-------|---------------------|----------|
| None (baseline) | 1x | 0% | 1x | 6 GB | < 100K vectors |
| Scalar | 4x | 2-5% | 1.5x | 1.5 GB | 100K-1M vectors |
| Binary (BQ) | 32x | 10-15% | 3x | 192 MB | 1M-10M vectors |
| Product (PQ) | 64x | 5-10% | 2x | 96 MB | 10M+ vectors |

---

### Session 2: DiskANN & Storage Architecture

**Resource**: [Timescale: DiskANN & SBQ](https://www.tigerdata.com/blog/how-we-made-postgresql-as-fast-as-pinecone-for-vector-data/)

#### The RAM vs SSD Problem

[Document why HNSW needs RAM]

**HNSW Architecture**:
- Multi-layer graph
- Random access pattern: 
- Number of seeks per query: 
- Why this requires RAM: 

**Cost Analysis**:
- RAM: $8/GB-month (AWS)
- SSD: $0.10/GB-month (AWS)
- For 100GB of vectors:
  - RAM cost: $_____/month
  - SSD cost: $_____/month
  - Savings: _____x

**At what scale does this matter?**

---

#### DiskANN: Making SSD Viable

[Document the architectural innovation]

**Key Innovation**:
- Single-layer graph (not multi-layer)
- Long-range edges added during construction
- Fewer random seeks per query

**How it works**:
1. Graph Construction:
   - 
2. Long-range edges:
   - 
3. Search process:
   - 

**Why this enables SSD**:
- HNSW seeks: _____ per query
- DiskANN seeks: _____ per query
- SSD latency: ~100 microseconds per seek
- DiskANN total latency: 

---

#### Statistical Binary Quantization (SBQ)

[Document Timescale's innovation]

**Standard BQ Limitation**:
- Uses 0.0 as cutoff for each dimension
- Problem: Real data may not be centered at 0
- Result: Uneven space partitioning

**SBQ Improvement #1: Mean Cutoff**:
- Learn the mean of each dimension during training
- Use mean instead of 0.0 as cutoff
- Result: 
- **WHEN learning happens**: 

**SBQ Improvement #2: 2-bit Encoding**:
- For lower dimensions (< 900), use 2 bits per dimension
- Three regions based on z-score:
  - Region 1: z < -σ → encode as 00
  - Region 2: -σ ≤ z ≤ σ → encode as 01
  - Region 3: z > σ → encode as 11
- More quadrants = better differentiation

**Results**:
- Standard BQ (768 dims): 96.5% recall
- SBQ (768 dims): 98.6% recall
- Improvement: 2.1% (significant at high recall!)

---

#### Streaming Filtering

[Document the filtering problem and solution]

**Problem: Two-Stage Post-Filtering**:

**Example scenario**:
```
Total vectors: 10,000
Query: "machine learning"
Filter: category="AI" (only 1,000 vectors match = 10%)
ef_search = 100 (retrieve top 100 first)
Requested results: 10

Issue:
Step 1: Retrieve top 100 similar vectors
Step 2: Filter for category="AI"
Result: Only _____ match (if unlucky)
Missing: True matches beyond position 100
```

**Why this fails**:

**Solution: Streaming Filtering**:
- Don't have fixed ef_search cutoff
- Continuously retrieve "next nearest" vector
- Check filter on each
- Stop when k valid matches found
- **Guarantee**: 100% recall

**Implementation in Qdrant**:

---

### Checkpoint Questions (Day 2)

**Q1: When does each quantization method become necessary?**
Answer:

**Q2: What's the trade-off between compression and recall?**
Answer:

**Q3: Why does DiskANN work on SSD but HNSW doesn't?**
Answer:

**Q4: How does SBQ improve upon standard BQ?**
Answer:

**Q5: What's the difference between streaming and two-stage filtering?**
Answer:

---

## Day 3-4: Hands-On Experiments

**Dates**: January 15-16, 2026  
**Goal**: Validate theory through concrete testing with Qdrant

[Document experiment setup, results, and insights]

### Experiment 1: HNSW Parameter Tuning

**Date**: 
**Time**: 3 hours

#### Setup
[Document your setup process]

#### Results

**M=8 Results**:
- Build time: 
- Memory usage: 
- Query latency: 
- Recall@10: 

**M=16 Results**:
- Build time:
- Memory usage:
- Query latency:
- Recall@10:

**M=32 Results**:
- Build time:
- Memory usage:
- Query latency:
- Recall@10:

**M=64 Results**:
- Build time:
- Memory usage:
- Query latency:
- Recall@10:

#### Analysis

**Question 1: At what M does recall plateau?**
Answer:

**Question 2: Is 2% recall gain worth 2x memory?**
Answer:

**Question 3: What's the optimal M for production?**
Answer:

**Decision for my knowledge base**:
- I will use M = _____ because:

---

### Experiment 2: Quantization Comparison

**Date**:
**Time**: 2.5 hours

#### Results

**Full Precision (Baseline)**:
- Memory: 
- Query latency: 
- Recall@10: 100%

**Scalar Quantization**:
- Memory: 
- Compression ratio: 
- Query latency: 
- Recall@10: 
- Recall loss: 

#### Analysis

**Question: Is the recall loss acceptable?**
Answer:

**Decision for my knowledge base**:
- I will [enable/disable] quantization because:

---

### Experiment 3: Filtering Performance

**Date**:
**Time**: 2.5 hours

#### Results

**Test with 10% selectivity filter**:
- Total vectors: 
- Matching filter: 
- Results returned: 
- All match filter: [Yes/No]
- Recall: 

#### Analysis

**Question: Does streaming maintain 100% recall?**
Answer:

**Insight**:

---

## Day 5-7: Building Knowledge Base

**Dates**: January 17-19, 2026  
**Goal**: Apply learnings to production-ready system

[Document implementation decisions and lessons learned]

### Architecture Decisions

[Document WHY each decision was made, referencing experiments]

**HNSW Configuration**:
- M = _____ (because: experiment 1 showed ___)
- ef_construct = _____ (because: ___)
- ef_search = _____ (because: ___)

**Quantization**:
- Enabled: [Yes/No]
- Type: 
- Reason: 

**Filtering Strategy**:
- Approach: Streaming
- Reason: Experiment 3 proved 100% recall

---

### Implementation Insights

[Document lessons learned while building]

**Challenge 1**:
- Problem: 
- Solution: 
- Learning: 

**Challenge 2**:
- Problem:
- Solution:
- Learning:

---

### Performance Observations

[Document actual performance metrics]

**Indexing Performance**:
- Documents indexed: 
- Time taken: 
- Vectors per second: 

**Query Performance**:
- Average latency: 
- p95 latency: 
- p99 latency: 

**Quality**:
- Subjective relevance: [Good/Excellent/Poor]
- Example good result: 
- Example poor result: 

---

## Weekly Synthesis

**Date**: January 19, 2026

### Core Understanding Achieved

**Vector databases solve**:

**HNSW works by**:

**Quantization enables**:

**Key parameters**:
- M: 
- ef_construction: 
- ef_search: 

---

### Connections to Week 1

**Week 1 taught**: Embeddings are pre-computed lookups  
**Week 2 taught**: Indexes are pre-computed graphs  
**Common insight**: Pre-computation at [time] enables fast [operation] at [time]

---

### Production Insights

**When to use Qdrant**:
1.
2.
3.

**When NOT to use Qdrant**:
1.
2.

**Optimal configuration for < 1M vectors**:
- M = 
- ef_construct = 
- Quantization: 
- Reason: 

---

### Questions for Week 3

1.
2.
3.

---

## Total Word Count

[Update at end of week]

- Day 1: _____ words
- Day 2: _____ words  
- Day 3-4: _____ words
- Day 5-7: _____ words
- **Total: _____ words**

---

**Note**: This document follows my learning style:
- Precise terminology (no vague words)
- Explicit timing (when things happen)
- Concrete examples (with actual numbers)
- Connections to previous knowledge
- Questions as checkpoints
