# Week 2: Vector Databases Deep Dive

**Dates**: January 13-19, 2026 (7 days)  
**Time Commitment**: 24-26 hours  
**Goal**: Understand how vector databases work internally and build a production-ready knowledge base

**Tech Stack**: Qdrant (vector DB), Python, OpenAI embeddings, Claude API

---

## Core Question

**"How do vector databases enable fast similarity search at scale, and what architectural decisions determine production performance?"**

---

## Learning Philosophy

This week focuses on **universal concepts** that apply to ALL vector databases:
- Approximate Nearest Neighbor (ANN) search algorithms
- Index structures (HNSW, DiskANN)
- Quantization techniques (PQ, BQ, SBQ)
- Metadata filtering (pre/post/streaming)
- Production operations (sharding, replication, monitoring)

**Qdrant is the vehicle**, but concepts are **DB-agnostic**.

---

## Week Structure

### Day 1-2: Foundation (6-8 hours)
**Goal**: Build precise mental model of vector database architecture

### Day 3-4: Experiments (8 hours)
**Goal**: Validate theory through hands-on testing with Qdrant

### Day 5-7: Build (10 hours)
**Goal**: Apply learnings to production-ready knowledge base system

---

## Day 1: Vector Database Fundamentals (3-4 hours)

### Session 1: The Problem & Solution (1.5 hours)

**Start here**: [What is a Vector Database? - Pinecone](https://www.pinecone.io/learn/vector-database/)

**Reading approach**:
1. Read "What is a Vector Database?" section (15 min)
   - **Question**: What problem do vector DBs solve that Week 1's ChromaDB didn't address?
   - **Answer expected**: Scale (10M+ vectors), production features (sharding, replication, monitoring), speed (ANN vs brute-force)

2. Read "Vector Database vs Vector Index" section (15 min)
   - **Take notes**: List all 7 differences in your running notes
   - **Critical insight**: CRUD operations, metadata filtering, real-time updates, backups, ecosystem integration, security, multi-tenancy

3. Read "How does a vector database work?" section (30 min)
   - **Draw the pipeline**: Indexing → Querying → Post-processing
   - **Timing precision**: WHEN does each step happen?
     - Indexing: Index-time (once per vector insert)
     - Querying: Query-time (every search)
     - Post-processing: Query-time (after initial retrieval)

4. Read "Serverless Vector Databases" section (30 min)
   - **Focus**: Separation of storage/compute, multitenancy, freshness layer
   - **Connection**: How does this relate to Week 1's caching insights?

**Checkpoint questions** (answer in running notes):
- Q1: When is a vector index built? When is it traversed?
- Q2: What's stored WHERE? (RAM vs SSD vs both)
- Q3: Why can't traditional databases handle vector search efficiently?

**Deliverable**: `week-02/notes/day1-foundations.md` (800+ words)

---

### Session 2: HNSW Algorithm Deep Dive (2 hours)

**Continue to**: [HNSW Explained - Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/)

**Reading sequence**:
1. **Probability Skip List** (30 min)
   - Understand: Multi-layer structure enables fast search
   - **Key insight**: Top layer = long skips, bottom layer = short skips
   - Draw on paper: 10 nodes across 3 layers

2. **Navigable Small World (NSW)** (30 min)
   - Understand: Greedy routing, entry points, friend lists
   - **Critical**: Zoom-out phase (low-degree vertices) vs zoom-in phase (high-degree vertices)
   - **Question**: Why does starting at high-degree vertices improve performance?

3. **Creating HNSW** (45 min)
   - Understand: Layer assignment via probability function
   - **Parameters**: M (edges per vertex), ef_construction (build-time), ef_search (query-time)
   - **Timing precision**: 
     - Layer assignment: At insert-time, once per vector
     - Graph construction: At insert-time, O(M × log N) per insert
     - Graph traversal: At query-time, O(log N) comparisons

4. **Graph Construction** (15 min)
   - Understand: How M, M_max, M_max0 control graph structure
   - **Critical insight**: Higher M = better recall but more memory

**Work through concrete example**:
```
Given: 1M vectors (1536 dims)
Query: Find 10 nearest neighbors

HNSW traversal:
Layer 3 (entry): 1 vertex, check 4 neighbors → 4 distances
Layer 2: check 8 neighbors → 12 total distances  
Layer 1: check 16 neighbors → 28 total distances
Layer 0: check 64 neighbors → 92 total distances

Total: ~100 distance calculations vs 1M brute-force!
Time: ~5ms vs ~500ms
```

**Checkpoint questions**:
- Q1: Why does HNSW use multiple layers? (Answer in your own words)
- Q2: What happens at each layer during traversal?
- Q3: How does M parameter affect recall vs memory?
- Q4: When is ef_construction used vs ef_search?

**Deliverable**: Update `week-02/notes/day1-foundations.md` with HNSW section (500+ words)

**End-of-Day 1 Total**: 1300+ words in running notes

---

## Day 2: Quantization & Storage Trade-offs (3-4 hours)

### Session 1: Quantization Fundamentals (2 hours)

**Read**: [Product Quantization - Pinecone](https://www.pinecone.io/learn/series/faiss/product-quantization/)

**Reading approach**:
1. **Why Quantize?** (15 min)
   - Memory problem: 1M vectors × 1536 dims × 4 bytes = 6GB
   - At 10M vectors: 60GB RAM = expensive!
   - **Question**: At what scale does quantization become necessary?

2. **Product Quantization (PQ)** (45 min)
   - **4 steps**: Splitting → Training → Encoding → Querying
   - **Concrete example**:
     ```
     Original: 1536 dims × 4 bytes = 6 KB per vector
     PQ: Split into 96 subvectors of 16 dims
         K-means clustering: 256 centroids per subvector
         Encode: 96 × 1 byte (centroid ID) = 96 bytes
     Compression: 6 KB → 96 bytes = 64x!
     Accuracy loss: 5-10% recall drop
     ```
   - **Timing**: 
     - Training: Index-time (build codebook once)
     - Encoding: Index-time (encode each vector once)
     - Distance calculation: Query-time (use codebook lookups)

3. **Binary Quantization (BQ)** (30 min)
   - **Simple rule**: dim > 0 → 1, dim ≤ 0 → 0
   - **Distance**: XOR function (counts bit differences)
   - **Compression**: 1536 dims × 1 bit = 192 bytes = 32x
   - **Accuracy loss**: 10-15% recall drop

4. **Scalar Quantization** (30 min)
   - **Rule**: float32 → int8 (map to [0, 255])
   - **Compression**: 6 KB → 1.5 KB = 4x
   - **Accuracy loss**: 2-5% recall drop (best ratio!)

**Comparison table** (create in running notes):
```
| Method | Compression | Recall Loss | Speed Gain | Use Case |
|--------|-------------|-------------|------------|----------|
| None   | 1x          | 0%          | 1x         | < 100K vectors |
| Scalar | 4x          | 2-5%        | 1.5x       | 100K-1M vectors |
| BQ     | 32x         | 10-15%      | 3x         | 1M-10M vectors |
| PQ     | 64x         | 5-10%       | 2x         | 10M+ vectors |
```

**Checkpoint questions**:
- Q1: When does each quantization method become necessary?
- Q2: What's the trade-off between compression and recall?
- Q3: Why is PQ better than BQ for very large scales?

---

### Session 2: DiskANN & Storage Architecture (1.5 hours)

**Read**: [Timescale: DiskANN & SBQ](https://www.tigerdata.com/blog/how-we-made-postgresql-as-fast-as-pinecone-for-vector-data)

**Focus sections**:
1. **DiskANN Algorithm** (45 min)
   - **Problem**: HNSW requires RAM (multi-layer = more random access)
   - **Solution**: Single-layer graph with long-range edges
   - **Key insight**: Fewer random seeks = SSD viable
   - **Cost difference**:
     ```
     RAM: $8/GB-month (AWS)
     SSD: $0.10/GB-month (AWS)
     At 100GB: $800/month vs $10/month = 80x savings!
     ```

2. **Statistical Binary Quantization (SBQ)** (30 min)
   - **BQ limitation**: Uses 0.0 as cutoff (arbitrary!)
   - **SBQ improvement**: Use mean of each dimension as cutoff
   - **2-bit encoding**: More quadrants for 768-dim vectors
   - **Result**: 2% recall improvement over standard BQ

3. **Streaming Filtering** (15 min)
   - **Problem**: Two-stage post-filtering misses results
   - **Solution**: Continuous retrieval until k matches found
   - **Guarantee**: 100% recall regardless of filter selectivity

**Checkpoint questions**:
- Q1: Why does DiskANN work on SSD but HNSW doesn't?
- Q2: How does SBQ improve upon standard BQ?
- Q3: What's the difference between streaming and two-stage filtering?

**Deliverable**: Update `week-02/notes/day2-quantization-storage.md` (1000+ words)

**End-of-Day 2 Total**: 2300+ words in running notes

---

## Day 3-4: Hands-On Experiments (8 hours)

### Setup (1 hour)

**Install Qdrant**:
```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/week-02/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Terminal 2: Install Python client
cd week-02/project/experiments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install qdrant-client numpy openai python-dotenv
```

**Verify setup**:
```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
print(client.get_collections())  # Should return empty list
print("✓ Qdrant ready!")
```

**Create experiment structure**:
```
week-02/project/experiments/
├── 01_hnsw_tuning/
│   ├── test_m_parameter.py
│   ├── test_ef_construction.py
│   ├── test_ef_search.py
│   └── results.md
├── 02_quantization/
│   ├── test_scalar_quant.py
│   ├── test_binary_quant.py
│   └── results.md
├── 03_filtering/
│   ├── test_streaming_filter.py
│   ├── test_post_filter.py
│   └── results.md
└── shared/
    ├── data_loader.py    # Reuse Week 1 embeddings
    ├── metrics.py        # Recall, latency calculations
    └── config.yaml       # Experiment parameters
```

---

### Experiment 1: HNSW Parameter Tuning (3 hours)

**Hypothesis**: Increasing M improves recall but increases memory and build time

**Test matrix**:
```python
M_values = [8, 16, 32, 64]
ef_construct_values = [100, 200, 400]
ef_search_values = [50, 100, 200]

# Dataset: 10K vectors from Week 1 (1536 dims)
# Queries: 100 test queries
# Measure: Build time, memory, query latency, recall@10
```

**Implementation** (`01_hnsw_tuning/test_m_parameter.py`):
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
import time
import numpy as np

def test_m_parameter(client, vectors, queries, ground_truth, M):
    """Test HNSW with different M values"""
    
    collection_name = f"hnsw_m_{M}"
    
    # Create collection
    start_time = time.time()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=M,
            ef_construct=200,
        ),
    )
    
    # Insert vectors
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        ids=list(range(len(vectors))),
    )
    build_time = time.time() - start_time
    
    # Measure memory (approximate via collection info)
    info = client.get_collection(collection_name)
    memory_mb = info.vectors_count * 1536 * 4 / 1024 / 1024  # Rough estimate
    
    # Query with ef_search=100
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(ef_construct=200),
    )
    
    # Test queries
    query_times = []
    recalls = []
    
    for i, query in enumerate(queries):
        start = time.time()
        results = client.search(
            collection_name=collection_name,
            query_vector=query,
            limit=10,
        )
        query_times.append(time.time() - start)
        
        # Calculate recall@10
        retrieved_ids = [r.id for r in results]
        true_ids = ground_truth[i][:10]
        recall = len(set(retrieved_ids) & set(true_ids)) / 10
        recalls.append(recall)
    
    return {
        "M": M,
        "build_time_sec": build_time,
        "memory_mb": memory_mb,
        "avg_query_ms": np.mean(query_times) * 1000,
        "avg_recall": np.mean(recalls),
    }
```

**Expected results**:
```
M=8:  Build=15s, Memory=80MB,  Query=2ms, Recall=85%
M=16: Build=22s, Memory=140MB, Query=3ms, Recall=92%
M=32: Build=35s, Memory=250MB, Query=5ms, Recall=97%
M=64: Build=55s, Memory=450MB, Query=8ms, Recall=98.5%
```

**Analysis questions** (document in `results.md`):
1. At what M does recall plateau? (Diminishing returns point)
2. Is 2% recall gain (95% → 97%) worth 2x memory?
3. What's the optimal M for production? (Justify with data)

**Deliverable**: `week-02/project/experiments/01_hnsw_tuning/results.md`

---

### Experiment 2: Quantization Comparison (2.5 hours)

**Hypothesis**: Scalar quantization provides best recall/compression ratio

**Test**: Compare full precision vs scalar quantization (Qdrant supports this natively)

**Implementation** (`02_quantization/test_scalar_quant.py`):
```python
from qdrant_client.models import ScalarQuantization, ScalarType

def test_quantization(client, vectors, queries, ground_truth):
    """Compare full vs scalar quantization"""
    
    # Test 1: Full precision (baseline)
    results_full = create_and_test_collection(
        client,
        "quant_full",
        vectors,
        queries,
        ground_truth,
        quantization=None,
    )
    
    # Test 2: Scalar quantization (int8)
    results_scalar = create_and_test_collection(
        client,
        "quant_scalar",
        vectors,
        queries,
        ground_truth,
        quantization=ScalarQuantization(
            scalar=ScalarType.INT8,
            always_ram=True,  # Keep in RAM for speed
        ),
    )
    
    return {
        "full": results_full,
        "scalar": results_scalar,
    }
```

**Expected results**:
```
Full precision:    Memory=300MB, Query=15ms, Recall=100%
Scalar (int8):     Memory=75MB,  Query=8ms,  Recall=96.5%

Compression: 4x
Speed gain: 1.9x
Recall loss: 3.5%
```

**Analysis questions**:
1. Is 3.5% recall loss acceptable for 4x memory savings?
2. At what scale does quantization become necessary?
3. Would you use this in production? Why or why not?

**Deliverable**: `week-02/project/experiments/02_quantization/results.md`

---

### Experiment 3: Metadata Filtering (2.5 hours)

**Hypothesis**: Qdrant's streaming filtering maintains 100% recall

**Test scenario**:
```python
# Create 10K documents with categories:
# - 70% category="general"
# - 20% category="technical"  
# - 10% category="research"

# Query: Find similar documents WHERE category="research"
# Challenge: Only 10% of data matches filter
```

**Implementation** (`03_filtering/test_streaming_filter.py`):
```python
def test_filtering(client, vectors, queries, ground_truth):
    """Test filtering with different selectivity"""
    
    # Create collection with metadata
    client.create_collection(
        collection_name="filtering_test",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    
    # Insert vectors with category payload
    points = []
    for i, vector in enumerate(vectors):
        category = (
            "research" if i < 1000 else
            "technical" if i < 3000 else
            "general"
        )
        points.append({
            "id": i,
            "vector": vector,
            "payload": {"category": category}
        })
    
    client.upsert(collection_name="filtering_test", points=points)
    
    # Test query WITH filter
    results = client.search(
        collection_name="filtering_test",
        query_vector=queries[0],
        query_filter={"must": [{"key": "category", "match": {"value": "research"}}]},
        limit=10,
    )
    
    # Verify all results match filter
    all_match = all(r.payload["category"] == "research" for r in results)
    
    return {
        "total_vectors": len(vectors),
        "matching_filter": 1000,  # 10% of data
        "results_returned": len(results),
        "all_match_filter": all_match,
        "recall": calculate_recall(results, ground_truth),
    }
```

**Expected findings**:
- Qdrant returns correct top-10 even when filter matches only 10% of data
- Recall: 100% (no degradation from filtering)
- Latency: Slightly higher (needs to check more candidates)

**Deliverable**: `week-02/project/experiments/03_filtering/results.md`

---

### Day 3-4 Synthesis (30 min)

**Create**: `week-02/notes/experiment-insights.md`

**Document**:
1. **Optimal HNSW parameters** (from Experiment 1):
   - M=16 or M=32? (Justify)
   - ef_construct=200 (why?)
   - ef_search=100 (balance recall/speed)

2. **Quantization decision** (from Experiment 2):
   - Use scalar quantization? At what scale?
   - Expected recall loss: X%
   - Memory savings: Yx

3. **Filtering strategy** (from Experiment 3):
   - Streaming filtering works (100% recall verified)
   - Use cases: When filters are selective

**End-of-Day 4 Total**: 3300+ words in running notes + experiment results

---

## Day 5-7: Build Knowledge Base System (10 hours)

### Project: Personal Knowledge Base with Semantic Memory

**Architecture**:
```
Documents (txt, md, pdf, URLs)
    ↓
Ingestion Pipeline
    ├─ Document Loader
    ├─ Chunker (reuse Week 1 strategy)
    ├─ Embedder (OpenAI, with caching)
    └─ Qdrant Indexer (optimal params from experiments)
    ↓
Qdrant Collection (with metadata)
    ↓
Search Interface
    ├─ Semantic Search (pure vector)
    ├─ Filtered Search (with metadata)
    └─ Hybrid Search (vector + keyword via payload)
    ↓
CLI (Click interface)
```

**Project structure**:
```
week-02/project/knowledge-base/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # Load files/URLs
│   │   ├── chunker.py             # Reuse Week 1 semantic chunking
│   │   ├── embedder.py            # OpenAI with caching
│   │   └── indexer.py             # Qdrant bulk upload
│   ├── search/
│   │   ├── __init__.py
│   │   ├── semantic_search.py     # Pure vector similarity
│   │   ├── filtered_search.py     # With metadata filters
│   │   └── hybrid_search.py       # Vector + keyword
│   ├── storage/
│   │   ├── __init__.py
│   │   └── qdrant_manager.py      # Collection management
│   └── cli.py                      # Click interface
├── config/
│   └── config.yaml                 # Parameters from experiments
├── tests/
│   ├── test_ingestion.py
│   ├── test_search.py
│   └── test_integration.py
├── data/
│   ├── documents/                  # Sample docs for testing
│   └── cache/                      # Embedding cache
├── README.md
├── requirements.txt
└── setup.py
```

---

### Day 5: Ingestion Pipeline (4 hours)

**Module 1: Document Loader** (1 hour)

**Implementation** (`src/ingestion/document_loader.py`):
```python
from pathlib import Path
from typing import List, Dict
import requests
from dataclasses import dataclass

@dataclass
class Document:
    """Represents a loaded document"""
    content: str
    metadata: Dict[str, str]
    source: str

class DocumentLoader:
    """Load documents from various sources"""
    
    def load_file(self, file_path: str) -> Document:
        """Load from local file (.txt, .md, .pdf)"""
        path = Path(file_path)
        
        if path.suffix == ".txt" or path.suffix == ".md":
            content = path.read_text(encoding="utf-8")
        elif path.suffix == ".pdf":
            content = self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return Document(
            content=content,
            metadata={
                "filename": path.name,
                "type": path.suffix[1:],
                "size_bytes": path.stat().st_size,
            },
            source=str(path.absolute()),
        )
    
    def load_url(self, url: str) -> Document:
        """Load from web URL"""
        response = requests.get(url)
        response.raise_for_status()
        
        return Document(
            content=response.text,
            metadata={
                "url": url,
                "type": "web",
            },
            source=url,
        )
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF using PyPDF2"""
        import PyPDF2
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages)
```

---

**Module 2: Chunker** (1 hour)

**Reuse Week 1 strategy, adapted for documents**:
```python
# src/ingestion/chunker.py
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a document chunk"""
    content: str
    metadata: Dict[str, any]
    chunk_id: str

class DocumentChunker:
    """Chunk documents into semantic units"""
    
    def __init__(self, target_size: int = 512, overlap: int = 50):
        """
        Args:
            target_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
        """
        self.target_size = target_size
        self.overlap = overlap
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document using semantic boundaries
        
        Strategy:
        1. Split on paragraph boundaries (\n\n)
        2. If paragraph > target_size, split on sentences
        3. Add overlap for context preservation
        """
        paragraphs = document.content.split("\n\n")
        chunks = []
        
        for i, para in enumerate(paragraphs):
            # Simple strategy: each paragraph = chunk (if not too large)
            tokens = self._count_tokens(para)
            
            if tokens <= self.target_size:
                chunks.append(Chunk(
                    content=para,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "chunk_type": "paragraph",
                    },
                    chunk_id=self._generate_id(document.source, i),
                ))
            else:
                # Split large paragraph into smaller chunks
                sub_chunks = self._split_large_paragraph(para, i, document)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (reuse Week 1's tiktoken)"""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
```

---

**Module 3: Embedder** (1 hour)

**Reuse Week 1's caching strategy**:
```python
# src/ingestion/embedder.py
from typing import List
import hashlib
import json
from pathlib import Path
import openai

class CachedEmbedder:
    """Generate embeddings with caching (from Week 1)"""
    
    def __init__(self, model: str = "text-embedding-3-small", cache_dir: str = "data/cache"):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching"""
        embeddings = []
        
        for text in texts:
            # Check cache
            cache_key = self._get_cache_key(text)
            cached = self._load_from_cache(cache_key)
            
            if cached:
                embeddings.append(cached)
            else:
                # Generate and cache
                response = openai.embeddings.create(
                    model=self.model,
                    input=text,
                )
                embedding = response.data[0].embedding
                self._save_to_cache(cache_key, embedding)
                embeddings.append(embedding)
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Content-based cache key (deterministic)"""
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
```

---

**Module 4: Qdrant Indexer** (1 hour)

**Use optimal parameters from experiments**:
```python
# src/ingestion/indexer.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, 
    ScalarQuantization, ScalarType, PointStruct
)
from typing import List

class QdrantIndexer:
    """Index documents in Qdrant with optimal config"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
    
    def create_collection(self, collection_name: str, use_quantization: bool = False):
        """
        Create collection with optimal HNSW parameters from experiments
        
        Decisions from Day 3-4 experiments:
        - M=16 (balance recall/memory)
        - ef_construct=200 (good build quality)
        - Scalar quantization if use_quantization=True
        """
        quantization_config = (
            ScalarQuantization(
                scalar=ScalarType.INT8,
                always_ram=True,
            )
            if use_quantization
            else None
        )
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,  # From Experiment 1: optimal balance
                ef_construct=200,  # From Experiment 1
            ),
            quantization_config=quantization_config,
        )
    
    def index_chunks(self, collection_name: str, chunks: List[Chunk], embeddings: List[List[float]]):
        """Bulk upload chunks with embeddings"""
        points = [
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata,
                },
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Batch upload (efficient)
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,  # Wait for indexing to complete
        )
```

---

### Day 6: Search Interface (4 hours)

**Module 1: Semantic Search** (1.5 hours)

```python
# src/search/semantic_search.py
from qdrant_client import QdrantClient
from typing import List, Dict

class SemanticSearch:
    """Pure vector similarity search"""
    
    def __init__(self, client: QdrantClient, embedder: CachedEmbedder):
        self.client = client
        self.embedder = embedder
    
    def search(
        self, 
        collection_name: str, 
        query: str, 
        limit: int = 10,
        ef_search: int = 100,
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            ef_search: From Experiment 1, 100 is good balance
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_batch([query])[0]
        
        # Search Qdrant (uses streaming internally)
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            search_params={"hnsw_ef": ef_search},  # From experiments
        )
        
        return [
            {
                "content": r.payload["content"],
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
            }
            for r in results
        ]
```

---

**Module 2: Filtered Search** (1.5 hours)

**Use streaming filtering from Experiment 3**:
```python
# src/search/filtered_search.py
from qdrant_client.models import Filter, FieldCondition, MatchValue

class FilteredSearch:
    """Semantic search with metadata filters"""
    
    def search_with_filter(
        self,
        collection_name: str,
        query: str,
        filters: Dict[str, str],
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search with metadata filtering
        
        Uses Qdrant's streaming filtering (from Experiment 3)
        - Guarantees 100% recall
        - No ef_search cutoff issues
        """
        query_embedding = self.embedder.embed_batch([query])[0]
        
        # Build filter conditions
        must_conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        
        # Search with filter
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=must_conditions),
            limit=limit,
        )
        
        return self._format_results(results)
```

---

**Module 3: Hybrid Search** (1 hour)

**Combine vector + keyword matching**:
```python
# src/search/hybrid_search.py
class HybridSearch:
    """Combine vector similarity with keyword matching"""
    
    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        keywords: List[str],
        limit: int = 10,
        vector_weight: float = 0.7,
    ) -> List[Dict]:
        """
        Hybrid search: vector similarity + keyword match
        
        Strategy:
        1. Get vector search results (score 0-1)
        2. Get keyword matches from payload (binary: match=1, no match=0)
        3. Combine: final_score = vector_weight * vector_score + (1-vector_weight) * keyword_score
        """
        # Vector search
        vector_results = self.semantic_search.search(
            collection_name, query, limit=limit*2  # Oversample
        )
        
        # Keyword scoring
        scored_results = []
        for result in vector_results:
            content = result["content"].lower()
            keyword_score = sum(1 for kw in keywords if kw.lower() in content) / len(keywords)
            
            # Combine scores
            final_score = (
                vector_weight * result["score"] + 
                (1 - vector_weight) * keyword_score
            )
            
            scored_results.append({
                **result,
                "hybrid_score": final_score,
                "keyword_matches": keyword_score,
            })
        
        # Sort by hybrid score, return top-k
        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_results[:limit]
```

---

### Day 7: CLI & Testing (2 hours)

**CLI Interface** (1 hour):
```python
# src/cli.py
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def cli():
    """Knowledge Base - Semantic search for your documents"""
    pass

@cli.command()
@click.argument("path")
@click.option("--collection", default="knowledge_base")
def add(path: str, collection: str):
    """Add document(s) to knowledge base"""
    # Implementation: Load → Chunk → Embed → Index
    console.print(f"[green]✓[/green] Indexed {path}")

@cli.command()
@click.argument("query")
@click.option("--collection", default="knowledge_base")
@click.option("--limit", default=10)
def search(query: str, collection: str, limit: int):
    """Semantic search"""
    results = semantic_search.search(collection, query, limit)
    
    # Display results in table
    table = Table(title=f"Results for: {query}")
    table.add_column("Score", style="cyan")
    table.add_column("Content", style="white")
    
    for r in results:
        table.add_row(
            f"{r['score']:.3f}",
            r['content'][:100] + "...",
        )
    
    console.print(table)

@cli.command()
@click.argument("query")
@click.option("--filter-key")
@click.option("--filter-value")
def filter_search(query: str, filter_key: str, filter_value: str):
    """Search with metadata filter"""
    # Implementation
    pass

@cli.command()
@click.option("--collection", default="knowledge_base")
def stats(collection: str):
    """Show collection statistics"""
    info = client.get_collection(collection)
    
    console.print(f"[bold]Collection:[/bold] {collection}")
    console.print(f"Vectors: {info.vectors_count:,}")
    console.print(f"Segments: {info.segments_count}")
    console.print(f"Status: {info.status}")

if __name__ == "__main__":
    cli()
```

**Usage examples**:
```bash
# Add documents
kb add week-01/notes/
kb add https://example.com/article.html

# Search
kb search "vector database architecture"

# Filtered search
kb filter-search "embeddings" --filter-key type --filter-value technical

# Stats
kb stats
```

---

**Testing** (1 hour):
```python
# tests/test_integration.py
def test_end_to_end():
    """Test complete pipeline: load → chunk → embed → index → search"""
    
    # Load sample document
    loader = DocumentLoader()
    doc = loader.load_file("data/documents/sample.txt")
    
    # Chunk
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 0
    
    # Embed
    embedder = CachedEmbedder()
    texts = [c.content for c in chunks]
    embeddings = embedder.embed_batch(texts)
    assert len(embeddings) == len(chunks)
    
    # Index
    indexer = QdrantIndexer()
    indexer.create_collection("test_collection")
    indexer.index_chunks("test_collection", chunks, embeddings)
    
    # Search
    searcher = SemanticSearch(indexer.client, embedder)
    results = searcher.search("test_collection", "sample query", limit=5)
    assert len(results) > 0
    assert results[0]["score"] > 0
```

---

## Week 2 Deliverables

### Documentation
1. **Running notes** (`week-02/notes/running-notes.md`): 3000+ words capturing all learnings
2. **Experiment results**: 3 detailed result files with analysis
3. **Architecture decisions**: Justified parameter choices from experiments
4. **README**: Complete project documentation

### Code
1. **Experiments**: 3 complete experiments with results
2. **Knowledge base**: Production-ready CLI tool with 6 modules
3. **Tests**: Integration tests validating end-to-end flow

### Learning Synthesis
1. **Blog post** (`week-02/blog-post.md`): "Vector Databases: From Theory to Production"
   - Section 1: The problem (scale, speed, production)
   - Section 2: HNSW explained (with your own diagrams)
   - Section 3: Quantization trade-offs (with experiment data)
   - Section 4: Qdrant in production (lessons learned)
   - Section 5: When to use what (decision framework)

---

## Success Metrics

By end of Week 2, you should be able to:

### Explain Precisely
- ✓ "HNSW graph is built at [index-time once per insert] and traversed at [query-time every search]"
- ✓ "Quantization compresses at [index-time via codebook] and decompresses at [query-time via lookup]"
- ✓ "M=16 provides [92% recall with 140MB for 10K vectors] vs M=32 [97% recall with 250MB]"

### Calculate
- ✓ Given N vectors, M edges, quantization → predict memory (show work)
- ✓ Compare compression ratios: Scalar (4x), BQ (32x), PQ (64x)

### Build
- ✓ Index 100+ personal documents in Qdrant
- ✓ Semantic search returns relevant results
- ✓ Filtered search maintains 100% recall

### Decide
- ✓ When to use M=16 vs M=32 (justify with data)
- ✓ When to enable quantization (at what scale?)
- ✓ When to use Qdrant vs other DBs (decision framework)

---

## Daily Checklist

### Day 1
- [ ] Read Pinecone vector DB article (1.5h)
- [ ] Read Pinecone HNSW article (2h)
- [ ] Document in running notes (1300+ words)
- [ ] Answer checkpoint questions

### Day 2
- [ ] Read Pinecone PQ article (1.5h)
- [ ] Read Timescale DiskANN article (1.5h)
- [ ] Document in running notes (1000+ words)
- [ ] Create comparison tables

### Day 3
- [ ] Setup Qdrant + experiment structure (1h)
- [ ] Run Experiment 1: HNSW tuning (3h)
- [ ] Document results with analysis

### Day 4
- [ ] Run Experiment 2: Quantization (2.5h)
- [ ] Run Experiment 3: Filtering (2.5h)
- [ ] Synthesize insights (30min)

### Day 5
- [ ] Build ingestion pipeline (4h)
- [ ] Test with sample documents

### Day 6
- [ ] Build search interface (4h)
- [ ] Test all search modes

### Day 7
- [ ] Build CLI (1h)
- [ ] Integration testing (1h)
- [ ] Write blog post draft

---

## Resources

### Primary Reading (in order)
1. [Pinecone: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
2. [Pinecone: HNSW Explained](https://www.pinecone.io/learn/series/faiss/hnsw/)
3. [Pinecone: Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/)
4. [Timescale: DiskANN & SBQ](https://www.tigerdata.com/blog/how-we-made-postgresql-as-fast-as-pinecone-for-vector-data/)

### Implementation References
5. [Qdrant Documentation](https://qdrant.tech/documentation/)
6. [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
7. [Qdrant Filtering Guide](https://qdrant.tech/documentation/concepts/filtering/)
8. [Qdrant Optimization Guide](https://qdrant.tech/documentation/guides/optimize/)

---

## Connection to Week 1

**Week 1 insights that carry forward**:
- Embeddings are pre-computed lookups → enables caching (reused in Week 2)
- Deterministic embeddings → enables content-based cache keys (reused)
- Function-level chunking → adapted to document-level chunking (similar strategy)
- ChromaDB basics → graduated to Qdrant production features

**Week 2 builds on Week 1**:
- Week 1: How to generate embeddings (OpenAI API)
- Week 2: How to search embeddings at scale (vector indexes)
- Week 1: Simple local storage (ChromaDB)
- Week 2: Production-ready database (Qdrant with HNSW, quantization, filtering)

---

## Notes

- All code should be modular and testable
- Document WHY decisions were made (reference experiments)
- Use rich terminal output for better UX
- Keep Week 1's caching patterns (proven to work)
- Prioritize understanding over speed (depth > breadth)

**Remember**: Qdrant is the vehicle, concepts are universal. Everything you learn applies to Pinecone, Weaviate, Milvus, etc.
