---
name: RAG Code QA Implementation
overview: Build a RAG-powered code Q&A tool in phases, implementing and testing each component incrementally to validate understanding of embeddings and the RAG pattern.
todos:
  - id: phase1-parser
    content: Implement parser.py with Python/JS/TS function extraction
    status: pending
  - id: phase2-chunker
    content: Implement chunker.py to create semantic chunks from functions
    status: pending
    dependencies:
      - phase1-parser
  - id: phase3-embedder
    content: Implement embedder.py with OpenAI API and caching
    status: pending
    dependencies:
      - phase2-chunker
  - id: phase4-storage
    content: Implement storage.py with ChromaDB integration
    status: pending
    dependencies:
      - phase3-embedder
  - id: phase5-retriever
    content: Implement retriever.py for semantic search
    status: pending
    dependencies:
      - phase4-storage
  - id: phase6-generator
    content: Implement generator.py with Claude API for answer generation
    status: pending
    dependencies:
      - phase5-retriever
  - id: phase7-cli
    content: Implement cli.py with all commands and create README
    status: pending
    dependencies:
      - phase6-generator
---

# RAG Code QA - Phased Implementation Plan

## Implementation Strategy

We'll build this in **7 phases**, each independently testable. After each phase, we'll validate understanding before moving forward.---

## Phase 1: Parser (Foundation)

**Goal**: Extract functions from code files with metadata**Files to create**:

- [`src/parser.py`](src/parser.py) - Core parsing logic
- [`tests/test_parser.py`](tests/test_parser.py) - Unit tests

**What we'll build**:

- `ParsedFunction` dataclass
- Python parser using `ast` module (requirements P1, P3, P4, P5)
- JavaScript/TypeScript parser using regex (requirement P2) - simpler for learning
- Recursive directory traversal (P8)
- Error handling (P7)

**Validation**: Can extract functions from sample Python/JS files with correct metadata---

## Phase 2: Chunker (Semantic Units)

**Goal**: Convert parsed functions into embeddable chunks**Files to create**:

- [`src/chunker.py`](src/chunker.py) - Chunk creation logic
- [`tests/test_chunker.py`](tests/test_chunker.py) - Unit tests

**What we'll build**:

- `Chunk` dataclass
- Format chunks with file path headers (C3)
- Generate deterministic IDs via content hashing (C4)
- Preserve metadata for retrieval (C5)

**Validation**: Chunks are properly formatted and have stable IDs---

## Phase 3: Embedder (OpenAI Integration)

**Goal**: Generate embeddings with caching layer**Files to create**:

- [`src/embedder.py`](src/embedder.py) - Embedding generation + cache
- [`tests/test_embedder.py`](tests/test_embedder.py) - Unit tests (mocked API)
- [`config/config.yaml`](config/config.yaml) - Configuration file

**What we'll build**:

- OpenAI API integration using `text-embedding-3-small` (E1, E2)
- File-based caching with SHA256 keys (E3, E6, E7)
- Rate limiting with exponential backoff (E5)
- Batch processing (E4)

**Validation**: Embeddings are cached correctly, API calls minimized on re-runs---

## Phase 4: Storage (ChromaDB)

**Goal**: Persist embeddings in vector database**Files to create**:

- [`src/storage.py`](src/storage.py) - ChromaDB management
- [`tests/test_storage.py`](tests/test_storage.py) - Unit tests

**What we'll build**:

- ChromaDB client with persistent storage (S1)
- Collection with cosine similarity (S2, S4)
- Upsert operations (S5)
- Utility methods: list_all(), stats(), delete_collection() (S6, S7, S8)

**Validation**: Can store and retrieve chunks with embeddings---

## Phase 5: Retriever (Semantic Search)

**Goal**: Find relevant chunks for queries**Files to create**:

- [`src/retriever.py`](src/retriever.py) - Similarity search
- [`tests/test_retriever.py`](tests/test_retriever.py) - Unit tests

**What we'll build**:

- `RetrievalResult` dataclass
- Query embedding + ChromaDB search (R1, R2, R3)
- Score calculation and filtering (R4, R5)
- Metadata filtering support (R6)

**Validation**: Retrieves semantically relevant chunks with correct scores---

## Phase 6: Generator (Claude Integration)

**Goal**: Generate grounded answers with citations**Files to create**:

- [`src/generator.py`](src/generator.py) - Answer generation
- [`tests/test_generator.py`](tests/test_generator.py) - Unit tests (mocked API)

**What we'll build**:

- Claude API integration with `claude-sonnet-4` (G1)
- Context formatting with file paths and line numbers (G2, G3)
- System prompt for citation enforcement (G4)
- Empty results handling (G5)
- Token limit management (G6)

**Validation**: Generates answers that cite specific code locations---

## Phase 7: CLI (User Interface)

**Goal**: Command-line interface for all operations**Files to create**:

- [`src/cli.py`](src/cli.py) - Click-based CLI
- [`README.md`](README.md) - Usage documentation

**What we'll build**:

- `index` command - Index codebase
- `query` command - Ask questions
- `list` command - Show indexed functions
- `stats` command - Display statistics
- Progress indicators and error messages

**Validation**: End-to-end flow works from indexing to querying---

## Testing Strategy

For each phase:

1. **Implement** the component
2. **Write tests** to validate behavior
3. **Run tests** to ensure correctness
4. **Discuss** what we learned before moving to next phase

---

## Dependencies Setup

Before Phase 1, we'll verify/update:

- [`requirements.txt`](requirements.txt) - Ensure all dependencies listed
- [`setup.py`](setup.py) - Package configuration
- Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

---

## Success Metrics

After Phase 7, we should be able to:

- Index a real codebase (e.g., this project itself)
- Ask questions and get grounded answers with citations
- Understand why each architectural decision was made