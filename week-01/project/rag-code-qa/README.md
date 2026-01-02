# RAG-Code-QA: Code Question Answering System

A CLI tool that answers natural language questions about codebases using Retrieval-Augmented Generation (RAG).

**Status**: 🚧 In Development (Component 1/7 Complete)
**Learning Project**: Week 1 of 8-week AI Engineering Journey

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
**Status**: Tested and working
**What it does**: Extracts functions from Python/JS/TS files with metadata

**Module structure**:
- `models.py` - ParsedFunction dataclass
- `python_parser.py` - AST-based Python parsing
- `javascript_parser.py` - Regex-based JS/TS parsing
- `dispatcher.py` - File extension routing
- `directory_walker.py` - Recursive directory traversal
- `README.md` - Comprehensive documentation (474 lines)
- `LLD.md` - Low-level design diagrams (635 lines)

**Key features**:
- Python parsing using AST (handles nested functions, class methods, async)
- JavaScript/TypeScript parsing using regex
- Recursive directory traversal with smart filtering
- Graceful error handling
- Preserves docstrings and line numbers
- Well-documented with learning context
- Detailed LLD diagrams with file/method references

**Test results**: All edge cases handled, modular structure verified

📄 **Documentation**: See [src/parser/README.md](src/parser/README.md) for detailed documentation and [LLD.md](src/parser/LLD.md) for design diagrams

### ⏳ Component 2: Chunker (Next)
**File**: `src/chunker.py`
**Status**: Not started
**What it will do**: Convert ParsedFunction objects into chunks for embedding

**Key decisions** (from REQUIREMENTS.md):
- One chunk per function (semantic unit preservation)
- Format: `# {file_path}:{start_line}\n{code}`
- Deterministic chunk IDs from content hash
- Include docstrings in chunk content

### 📋 Component 3: Embedder (Planned)
**File**: `src/embedder.py`
**What it will do**: Generate embeddings via OpenAI API with caching

**Key decisions**:
- Model: text-embedding-3-small (1536 dimensions)
- Caching enabled (embeddings are deterministic)
- Batch API calls for efficiency

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
│   ├── parser/            ✅ Complete (modular package)
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── python_parser.py
│   │   ├── javascript_parser.py
│   │   ├── dispatcher.py
│   │   ├── directory_walker.py
│   │   ├── README.md      (474 lines - comprehensive docs)
│   │   └── LLD.md         (635 lines - detailed diagrams)
│   ├── chunker.py         ⏳ Next
│   ├── embedder.py        📋 Planned
│   ├── storage.py         📋 Planned
│   ├── retriever.py       📋 Planned
│   ├── generator.py       📋 Planned
│   └── cli.py             📋 Planned
├── tests/
│   └── (unit tests for each module)
├── data/
│   ├── indexed/           (ChromaDB persistence)
│   └── cache/             (Embedding cache)
├── config/
│   └── config.yaml        (Configuration)
├── REQUIREMENTS.md        (Detailed specifications)
└── README.md              (This file)
```

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

**Last Updated**: January 2, 2026
**Current Focus**: Parser complete, chunker next
**Learning Mode**: Building step-by-step with understanding
