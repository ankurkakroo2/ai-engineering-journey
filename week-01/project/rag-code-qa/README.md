# RAG-Code-QA: Code Question Answering System

A CLI tool that answers natural language questions about codebases using Retrieval-Augmented Generation (RAG).

**Status**: 🚧 In Development (Component 3/7 Complete - **43% progress**)
**Learning Project**: Week 1, Day 5-6 of 8-week AI Engineering Journey

---

## What This Tool Does

```bash
# Index a codebase
rag-code-qa index ./my-project

# Ask questions about it
rag-code-qa query "How does authentication work?"

# Get answers grounded in actual code with citations
```

**How it works**:
1. Parse code files → Extract functions with metadata
2. Create semantic chunks → Embed using OpenAI
3. Store in vector database → ChromaDB with cosine similarity
4. Retrieve relevant chunks → Pass to Claude as context
5. Generate grounded answers → With file/line citations

---

## Architecture

```
User Query
    ↓
Parser (✅ Complete) → Extract functions from code
    ↓
Chunker (⏳ Next) → Create semantic chunks
    ↓
Embedder → Generate embeddings (OpenAI)
    ↓
Storage → Store in ChromaDB
    ↓
Retriever → Find relevant chunks
    ↓
Generator → Claude generates answer
    ↓
Answer with Citations
```

---

## Implementation Progress

### ✅ Component 1: Parser (Complete)
**Module**: `src/parser/` (modular package)
**Status**: ✅ Implemented and tested
**What it does**: Extracts functions from Python/JS/TS files with metadata

**Module structure**:
- `models.py` - ParsedFunction dataclass
- `python_parser.py` - AST-based Python parsing
- `javascript_parser.py` - Regex-based JS/TS parsing
- `dispatcher.py` - File extension routing strategy
- `directory_walker.py` - Recursive directory traversal
- `__init__.py` - Public API
- `README.md` - Comprehensive documentation (474 lines)
- `LLD.md` - Low-level design diagrams (635 lines)

**Key features**:
- Python parsing using AST (handles nested functions, class methods, async)
- JavaScript/TypeScript parsing using regex patterns
- Recursive directory traversal with smart filtering (skip node_modules, __pycache__, .git)
- Graceful error handling (one bad file doesn't crash pipeline)
- Preserves docstrings and line numbers for citations
- Absolute file paths for reliability
- Detailed LLD diagrams with file/method references

**Test results**: All edge cases handled, modular structure verified, ready for chunker input

📄 **Documentation**: See [src/parser/README.md](src/parser/README.md) for detailed docs and [src/parser/LLD.md](src/parser/LLD.md) for design diagrams

### ✅ Component 2: Chunker (Complete)
**Module**: `src/chunker/` (modular package)
**Status**: ✅ Implemented and tested
**What it does**: Converts ParsedFunction objects into embeddable chunks

**Module structure**:
- `models.py` - Chunk dataclass with deterministic IDs
- `formatter.py` - Three-layer formatting + intelligent truncation
- `token_counter.py` - tiktoken integration (accurate OpenAI tokenization)
- `hasher.py` - SHA-256 deterministic ID generation
- `chunker.py` - Main orchestrator (format → count → hash)
- `__init__.py` - Public API
- `README.md` - Comprehensive documentation (461 lines)
- `LLD.md` - Low-level design diagrams with complete flow

**Key features**:
- Three-layer format: location (spatial context) + docstring (semantics) + code (implementation)
- Deterministic content-based IDs enable change detection and incremental updates
- Accurate token counting using tiktoken (matches OpenAI's tokenizer)
- Intelligent truncation preserves semantic units (header + docstring, truncate code)
- 8000 token limit validation with graceful degradation
- Metadata preservation for citations (file path, line numbers, language, etc.)
- Batch processing with statistics and warnings

**Test results**: Metadata preservation verified, token counting accurate, deterministic ID generation validated

📄 **Documentation**: See [src/chunker/README.md](src/chunker/README.md) for detailed docs and [src/chunker/LLD.md](src/chunker/LLD.md) for design diagrams

### ✅ Component 3: Embedder (Complete)
**Module**: `src/embedder/` (modular package)
**Status**: ✅ Implemented and tested
**What it does**: Generates vector embeddings from chunks using OpenAI API with intelligent caching

**Module structure**:
- `models.py` - EmbeddingResult dataclass
- `api_client.py` - OpenAI API integration with error handling
- `cache.py` - File-based and in-memory caching layers
- `embedder.py` - Main orchestrator (batch processing + rate limiting)
- `__init__.py` - Public API
- `README.md` - Comprehensive documentation (16K)
- `LLD.md` - Low-level design diagrams (12K)

**Key features**:
- Batch processing for efficiency (~10x overhead reduction)
- Deterministic caching (same input = identical output, always)
- Exponential backoff for rate limit handling
- Dual-layer caching (in-memory for speed + file-based for persistence)
- Comprehensive error handling and retry logic
- 1536-dimensional vectors (text-embedding-3-small)

**Test results**: Caching validated, batch processing verified, rate limit handling tested

📄 **Documentation**: See [src/embedder/README.md](src/embedder/README.md) for detailed docs and [src/embedder/LLD.md](src/embedder/LLD.md) for design diagrams

### 📋 Component 4: Storage (Planned)
**File**: `src/storage.py`
**What it will do**: Manage ChromaDB collection

**Key decisions**:
- Distance metric: Cosine similarity
- Persistent storage in `./data/indexed`
- Upsert support for incremental updates

### 📋 Component 5: Retriever (Planned)
**File**: `src/retriever.py`
**What it will do**: Find semantically relevant chunks

**Key decisions**:
- Top-k: 5 chunks (default)
- Minimum score threshold: 0.3
- Return chunks with similarity scores

### 📋 Component 6: Generator (Planned)
**File**: `src/generator.py`
**What it will do**: Generate answers using Claude

**Key decisions**:
- Model: claude-sonnet-4
- Context: Retrieved chunks as working memory
- Citations: File paths and line numbers

### 📋 Component 7: CLI (Planned)
**File**: `src/cli.py`
**What it will do**: Command-line interface

**Commands**:
- `index <path>` - Index a codebase
- `query <question>` - Ask a question
- `list` - List indexed functions
- `stats` - Show index statistics

---

## Project Structure

```
rag-code-qa/
├── src/
│   ├── __init__.py
│   ├── parser/              ✅ Complete (modular package)
│   │   ├── __init__.py      (Public API)
│   │   ├── models.py        (ParsedFunction dataclass)
│   │   ├── python_parser.py (AST-based Python parsing)
│   │   ├── javascript_parser.py (Regex-based JS/TS parsing)
│   │   ├── dispatcher.py    (File routing)
│   │   ├── directory_walker.py (Recursive traversal)
│   │   ├── test_parser_manual.py (Manual test suite)
│   │   ├── README.md        (474 lines - learning context + examples)
│   │   └── LLD.md           (635 lines - flow diagrams + references)
│   ├── chunker/             ✅ Complete (modular package)
│   │   ├── __init__.py      (Public API)
│   │   ├── models.py        (Chunk dataclass)
│   │   ├── formatter.py     (Three-layer formatting + truncation)
│   │   ├── token_counter.py (tiktoken integration)
│   │   ├── hasher.py        (SHA-256 ID generation)
│   │   ├── chunker.py       (Orchestrator: format → count → hash)
│   │   ├── test_chunker_manual.py (Manual test suite)
│   │   ├── README.md        (461 lines - learning context + examples)
│   │   └── LLD.md           (Flow diagrams + component reference)
│   ├── embedder/            ✅ Complete (modular package)
│   │   ├── __init__.py      (Public API)
│   │   ├── models.py        (EmbeddingResult dataclass)
│   │   ├── api_client.py    (OpenAI API integration)
│   │   ├── cache.py         (Dual-layer caching: in-memory + file-based)
│   │   ├── embedder.py      (Orchestrator: batch processing + rate limiting)
│   │   ├── test_embedder_manual.py (Manual test suite)
│   │   ├── README.md        (16K - learning context + examples)
│   │   └── LLD.md           (12K - flow diagrams + reference)
│   ├── storage.py           📋 Next - ChromaDB management + persistence
│   ├── retriever.py         📋 Planned - Similarity search
│   ├── generator.py         📋 Planned - Claude + context
│   └── cli.py               📋 Planned - Click interface
├── tests/
│   ├── __init__.py
│   ├── test_parser.py       (Unit tests)
│   ├── test_chunker.py      (Unit tests)
│   ├── test_embedder.py     (Planned)
│   ├── test_storage.py      (Planned)
│   ├── test_retriever.py    (Planned)
│   └── test_generator.py    (Planned)
├── data/
│   ├── indexed/             (ChromaDB persistence - gitignored)
│   └── cache/               (Embedding cache - gitignored)
├── config/
│   └── config.yaml          (Configuration)
├── REQUIREMENTS.md          (Detailed specifications)
├── README.md                (This file)
├── setup.py
└── requirements.txt
```

**Progress**: 3/7 components complete (43%)

---

## Testing

### Manual Testing (Current)
```bash
cd week-01/project/rag-code-qa
python3 test_parser_manual.py
```

This tests the parser by parsing the rag-code-qa codebase itself.

### Unit Tests (Planned)
```bash
pytest tests/
```

---

## Learning Objectives

By building this project, you'll understand:

1. **Embeddings**: Pre-trained models as semantic lookups
2. **Vector databases**: High-dimensional similarity search
3. **Chunking strategies**: Why function-level > fixed-size
4. **RAG pattern**: Retrieved context prevents hallucinations
5. **Distance metrics**: Cosine vs Euclidean for code
6. **Working memory**: How LLMs use context

These concepts are validated by the 7 experiments in `../experiments/01_embeddings/`.

---

## Architecture Decisions

All decisions are validated by experiments:

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding Model | text-embedding-3-small | 7% accuracy improvement (1536 vs 384 dim) |
| Distance Metric | Cosine similarity | Handles length variation in code |
| Chunking | By function | Preserves semantic units |
| Top-k | 5 | Balance between context and noise |
| Caching | Enabled | Embeddings are deterministic |
| LLM | Claude Sonnet 4 | Strong reasoning for code |

See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed rationale.

---

## Dependencies

```bash
# Core
click>=8.0.0
pyyaml>=6.0

# Embeddings
openai>=1.0.0
numpy>=1.24.0

# Vector Database
chromadb>=0.4.0

# LLM
anthropic>=0.18.0

# Utilities
python-dotenv>=1.0.0

# Development
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## Next Steps

1. **Implement chunker.py** - Convert ParsedFunction → Chunk
2. **Implement embedder.py** - Generate embeddings with caching
3. **Implement storage.py** - ChromaDB integration
4. **Implement retriever.py** - Similarity search
5. **Implement generator.py** - Claude answer generation
6. **Implement cli.py** - Command-line interface
7. **End-to-end testing** - Validate full pipeline

---

## Resources

- **Session Summary**: `../SESSION_SUMMARY.md` - Days 1-4 progress
- **Requirements**: `REQUIREMENTS.md` - Detailed specifications
- **Parser Module**: `src/parser/README.md` - Component 1 documentation
- **Experiments**: `../experiments/01_embeddings/` - Architecture validation

---

**Last Updated**: January 3, 2026 (Day 5-6)
**Current Focus**: Parser ✅ & Chunker ✅ & Embedder ✅ complete, storage next
**Progress**: 3/7 components (43%) - modular architecture validated through three production modules
**Learning Mode**: Building step-by-step with comprehensive documentation
