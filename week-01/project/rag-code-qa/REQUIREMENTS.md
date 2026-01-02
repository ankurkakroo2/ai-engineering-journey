# RAG-Code-QA: Requirements Document

> **Context**: This is a learning project as part of an 8-week AI engineering journey. Week 1 focuses on embeddings fundamentals. This project is the culmination of Days 1-4 learning and experiment validation.

---

## 1. Project Overview

### What You're Building
A CLI tool that answers natural language questions about a codebase by:
1. Indexing code files into a vector database
2. Retrieving semantically relevant code chunks
3. Using Claude to generate answers grounded in actual code

### Why This Architecture
This implements the **RAG (Retrieval-Augmented Generation) pattern**:
- **Problem**: LLMs hallucinate when asked about code they haven't seen
- **Solution**: Retrieve relevant code chunks → pass as context → generate grounded answer
- **Key insight**: Retrieved chunks act as "working memory" for the LLM (validated in Experiment 1, Test 7)

### Learning Objectives
By building this, you validate understanding of:
- Pre-trained embeddings as semantic lookups (not runtime calculations)
- Cosine similarity for high-dimensional vector comparison
- Function-level chunking for semantic preservation
- RAG pattern for grounded generation

---

## 2. Architecture Decisions (Validated by Experiment 1)

These decisions come directly from the 7 validation tests in `experiments/01_embeddings/`:

| Decision | Choice | Validated By | Rationale |
|----------|--------|--------------|-----------|
| Embedding Model | `text-embedding-3-small` | Test 1, 3 | 1536 dimensions, good balance of quality/cost |
| Dimensions | 1536 (full) | Test 3 | 7%+ accuracy improvement over 384-dim truncation |
| Distance Metric | Cosine similarity | Test 4 | Handles length variation; Euclidean fails on short vs verbose code |
| Chunking Strategy | By function | Test 6 | Semantic units retrieve better than fixed-size chunks |
| Top-k Retrieval | 5 | Test 7 | Provides enough context without noise |
| Multi-language | Python, JS, TS | Test 2, 5 | Semantic clustering works across languages |
| Caching | Enabled | Test 1 | Embeddings are deterministic (same input = identical output) |
| LLM for Generation | Claude (claude-sonnet-4) | Test 7 | Working memory pattern requires strong reasoning |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                            │
│                  index | query | list | stats                   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  INDEX FLOW     │  │  QUERY FLOW     │  │  UTILITY FLOW   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    parser.py    │  │  retriever.py   │  │   storage.py    │
│ Extract funcs   │  │ Similarity srch │  │  List/Stats     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │
        ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│   chunker.py    │  │  generator.py   │
│ Create chunks   │  │ Claude + context│
└─────────────────┘  └─────────────────┘
        │                    │
        ▼                    │
┌─────────────────┐          │
│  embedder.py    │          │
│ OpenAI API      │          │
└─────────────────┘          │
        │                    │
        ▼                    │
┌─────────────────────────────────────────────────────────────────┐
│                    ChromaDB (storage.py)                        │
│                  Persisted vector database                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Requirements

### 4.1 Parser (`src/parser.py`)

**Purpose**: Extract functions from source code files with metadata.

**Input**: File path or directory path
**Output**: List of `ParsedFunction` objects

**Data Structure**:
```python
@dataclass
class ParsedFunction:
    name: str              # Function/method name
    code: str              # Full source code including signature
    docstring: str | None  # Extracted docstring if present
    file_path: str         # Absolute path to source file
    start_line: int        # Line number where function starts
    end_line: int          # Line number where function ends
    language: str          # "python" | "javascript" | "typescript"
```

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| P1 | Parse Python files using `ast` module | Must |
| P2 | Parse JavaScript/TypeScript files using regex or tree-sitter | Must |
| P3 | Extract function name, full code, docstring, line numbers | Must |
| P4 | Handle nested functions (include as separate entries) | Should |
| P5 | Handle class methods (include class name in function name) | Should |
| P6 | Skip files matching `.gitignore` patterns | Should |
| P7 | Return empty list (not error) for unparseable files | Must |
| P8 | Support recursive directory traversal | Must |

**Python Parsing Approach**:
```python
import ast

def parse_python_file(file_path: str) -> list[ParsedFunction]:
    with open(file_path) as f:
        source = f.read()
    tree = ast.parse(source)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Extract function details
            # Use ast.get_source_segment() for code extraction
            pass
    return functions
```

**JavaScript/TypeScript Parsing Approach**:
```python
# Option 1: Regex-based (simpler, less accurate)
FUNCTION_PATTERN = r'(function\s+\w+|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'

# Option 2: tree-sitter (more accurate, requires dependency)
# tree-sitter-python, tree-sitter-javascript packages
```

**Error Handling**:
- File not found → log warning, return empty list
- Parse error → log warning, return empty list
- Permission denied → log warning, return empty list

---

### 4.2 Chunker (`src/chunker.py`)

**Purpose**: Convert parsed functions into chunks suitable for embedding.

**Input**: List of `ParsedFunction` objects
**Output**: List of `Chunk` objects

**Data Structure**:
```python
@dataclass
class Chunk:
    id: str                # Unique identifier (hash of content + metadata)
    content: str           # Text to embed (formatted function)
    metadata: dict         # {file_path, function_name, start_line, end_line, language}
```

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| C1 | One chunk per function (semantic unit preservation) | Must |
| C2 | Include docstring in chunk content if present | Must |
| C3 | Format chunk as: `# {file_path}:{start_line}\n{code}` | Must |
| C4 | Generate deterministic chunk ID from content hash | Must |
| C5 | Preserve all metadata for retrieval display | Must |
| C6 | Handle functions > 8000 tokens (truncate with warning) | Should |

**Chunk Format**:
```python
def format_chunk(func: ParsedFunction) -> str:
    """
    Format: Include file path and line number as header for context
    """
    header = f"# {func.file_path}:{func.start_line}"
    if func.docstring:
        return f"{header}\n\"\"\"{func.docstring}\"\"\"\n{func.code}"
    return f"{header}\n{func.code}"
```

**Why Function-Level Chunking** (from Test 6):
- Fixed-size chunks (e.g., 512 tokens) break semantic units
- A query like "authenticate user" should retrieve the whole `authenticate_user()` function
- Experiment 1 validated: semantic chunking > fixed-size chunking

---

### 4.3 Embedder (`src/embedder.py`)

**Purpose**: Generate embeddings via OpenAI API with caching layer.

**Input**: List of `Chunk` objects
**Output**: List of embeddings (numpy arrays of shape `(1536,)`)

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| E1 | Use `text-embedding-3-small` model | Must |
| E2 | Return 1536-dimensional embeddings | Must |
| E3 | Implement caching layer (embeddings are deterministic) | Must |
| E4 | Batch API calls (max 2048 inputs per request) | Should |
| E5 | Handle rate limiting with exponential backoff | Must |
| E6 | Cache key = hash of chunk content | Must |
| E7 | Cache storage = local JSON or pickle file | Should |

**API Integration**:
```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env var

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
```

**Caching Strategy** (validated by Test 1):
```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("./data/cache")

def get_cached_embedding(text: str) -> list[float] | None:
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None

def cache_embedding(text: str, embedding: list[float]):
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps(embedding))
```

**Why Caching Works** (from Test 1):
- Embeddings are pre-trained lookups, not runtime calculations
- Same input always produces identical output (bitwise)
- Safe to cache indefinitely

---

### 4.4 Storage (`src/storage.py`)

**Purpose**: Manage ChromaDB collection for storing and retrieving embeddings.

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| S1 | Use ChromaDB with persistent storage | Must |
| S2 | Single collection named `code-chunks` | Must |
| S3 | Store embedding, chunk content, and metadata | Must |
| S4 | Use cosine similarity (ChromaDB default) | Must |
| S5 | Support upsert (update if exists, insert if new) | Must |
| S6 | Provide list_all() for debugging | Should |
| S7 | Provide stats() for collection info | Should |
| S8 | Support delete_collection() for re-indexing | Should |

**ChromaDB Integration**:
```python
import chromadb
from chromadb.config import Settings

def get_collection():
    client = chromadb.PersistentClient(
        path="./data/indexed",
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name="code-chunks",
        metadata={"hnsw:space": "cosine"}  # Cosine similarity
    )

def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]):
    collection = get_collection()
    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[c.metadata for c in chunks]
    )
```

**Why Cosine Similarity** (from Test 4):
- High-dimensional vectors need angle-based comparison
- Euclidean distance fails when comparing short vs verbose code
- Cosine normalizes for length, focuses on direction

---

### 4.5 Retriever (`src/retriever.py`)

**Purpose**: Find semantically relevant chunks for a query.

**Input**: Query string, top_k parameter
**Output**: List of `RetrievalResult` objects

**Data Structure**:
```python
@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    metadata: dict
    score: float          # Cosine similarity score (0-1)
    distance: float       # ChromaDB distance (1 - similarity)
```

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Embed query using same model as indexing | Must |
| R2 | Query ChromaDB with cosine similarity | Must |
| R3 | Return top_k results (default: 5) | Must |
| R4 | Include similarity score in results | Must |
| R5 | Filter by minimum score threshold (default: 0.3) | Should |
| R6 | Support filtering by language or file path | Should |

**Retrieval Flow**:
```python
def retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
    # 1. Embed the query
    query_embedding = get_embedding(query)

    # 2. Query ChromaDB
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Convert to RetrievalResult objects
    # Note: ChromaDB returns distance, not similarity
    # For cosine: similarity = 1 - distance
    return [
        RetrievalResult(
            chunk_id=results['ids'][0][i],
            content=results['documents'][0][i],
            metadata=results['metadatas'][0][i],
            distance=results['distances'][0][i],
            score=1 - results['distances'][0][i]
        )
        for i in range(len(results['ids'][0]))
    ]
```

---

### 4.6 Generator (`src/generator.py`)

**Purpose**: Generate answers using Claude with retrieved chunks as context.

**Input**: Query string, list of `RetrievalResult` objects
**Output**: Generated answer string with citations

**Requirements**:

| ID | Requirement | Priority |
|----|-------------|----------|
| G1 | Use Claude API (claude-sonnet-4) | Must |
| G2 | Format retrieved chunks as context in prompt | Must |
| G3 | Include file paths and line numbers in context | Must |
| G4 | Instruct Claude to cite sources in answer | Must |
| G5 | Handle case when no relevant chunks found | Must |
| G6 | Limit context to ~6000 tokens (leave room for response) | Should |

**Prompt Template**:
```python
SYSTEM_PROMPT = """You are a code assistant answering questions about a codebase.
You will be given relevant code snippets as context.
Answer based ONLY on the provided code. If the answer isn't in the context, say so.
Always cite which file and line number your answer comes from."""

def build_context(results: list[RetrievalResult]) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        file_info = f"{r.metadata['file_path']}:{r.metadata['start_line']}"
        context_parts.append(f"[{i}] {file_info}\n```\n{r.content}\n```")
    return "\n\n".join(context_parts)

def generate_answer(query: str, results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant code found for this query."

    context = build_context(results)

    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

**Why This Pattern** (from Test 7):
- Retrieved chunks = working memory for Claude
- Quality retrieval prevents hallucination
- Claude generates answers grounded in actual code, not imagination

---

### 4.7 CLI (`src/cli.py`)

**Purpose**: Command-line interface using Click framework.

**Commands**:

#### `index` - Index a codebase
```bash
rag-code-qa index <path> [--languages python,javascript,typescript]

# Examples:
rag-code-qa index ./src
rag-code-qa index ./src --languages python
rag-code-qa index /path/to/project --languages python,typescript
```

**Behavior**:
1. Recursively find all supported files in path
2. Parse each file → extract functions
3. Create chunks from functions
4. Generate embeddings (with caching)
5. Store in ChromaDB
6. Print summary: files processed, functions indexed, time taken

#### `query` - Ask a question
```bash
rag-code-qa query "<question>" [--top-k 5] [--show-context]

# Examples:
rag-code-qa query "How does authentication work?"
rag-code-qa query "Where is the database connection handled?" --top-k 10
rag-code-qa query "What does the calculate_tax function do?" --show-context
```

**Behavior**:
1. Embed the query
2. Retrieve top-k chunks from ChromaDB
3. Pass chunks + query to Claude
4. Print generated answer
5. If `--show-context`: also print retrieved chunks with scores

#### `list` - List indexed functions
```bash
rag-code-qa list [--file <path>] [--language <lang>]

# Examples:
rag-code-qa list
rag-code-qa list --language python
rag-code-qa list --file src/auth.py
```

**Behavior**:
1. Query ChromaDB for all chunks (or filtered)
2. Print table: function name, file path, line number

#### `stats` - Show index statistics
```bash
rag-code-qa stats
```

**Behavior**:
1. Query ChromaDB collection metadata
2. Print: total chunks, files indexed, languages, last updated

**CLI Implementation**:
```python
import click

@click.group()
def cli():
    """RAG-powered code Q&A tool"""
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--languages', default='python,javascript,typescript')
def index(path, languages):
    """Index a codebase for semantic search"""
    # Implementation here
    pass

@cli.command()
@click.argument('question')
@click.option('--top-k', default=5)
@click.option('--show-context', is_flag=True)
def query(question, top_k, show_context):
    """Ask a question about the indexed codebase"""
    # Implementation here
    pass

if __name__ == '__main__':
    cli()
```

---

## 5. Configuration

**Location**: `config/config.yaml`

```yaml
# OpenAI Configuration
openai:
  model: text-embedding-3-small
  dimensions: 1536
  # API key from environment: OPENAI_API_KEY

# Claude Configuration
claude:
  model: claude-sonnet-4-20250514
  max_tokens: 1024
  # API key from environment: ANTHROPIC_API_KEY

# ChromaDB Configuration
chromadb:
  persist_directory: ./data/indexed
  collection_name: code-chunks
  distance_metric: cosine

# Indexing Configuration
indexing:
  chunk_strategy: by_function
  languages:
    - python
    - javascript
    - typescript
  ignore_patterns:
    - "**/__pycache__/**"
    - "**/node_modules/**"
    - "**/.git/**"
    - "**/venv/**"

# Retrieval Configuration
retrieval:
  top_k: 5
  min_score: 0.3

# Cache Configuration
cache:
  enabled: true
  directory: ./data/cache
```

---

## 6. Directory Structure

```
rag-code-qa/
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── __init__.py
│   ├── parser.py             # P1-P8: Extract functions from code
│   ├── chunker.py            # C1-C6: Create semantic chunks
│   ├── embedder.py           # E1-E7: OpenAI API + caching
│   ├── storage.py            # S1-S8: ChromaDB management
│   ├── retriever.py          # R1-R6: Similarity search
│   ├── generator.py          # G1-G6: Claude + context
│   └── cli.py                # CLI with Click
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_storage.py
│   ├── test_retriever.py
│   └── test_generator.py
├── data/
│   ├── indexed/              # ChromaDB persistence (gitignored)
│   └── cache/                # Embedding cache (gitignored)
├── REQUIREMENTS.md           # This file
├── architecture.md           # Design decisions
├── README.md                 # Usage documentation
├── setup.py                  # Package setup
└── requirements.txt          # Dependencies
```

---

## 7. Dependencies

**requirements.txt**:
```
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

## 8. Testing Requirements

Each module should have corresponding tests in `tests/`:

### test_parser.py
- Test Python function extraction
- Test JavaScript/TypeScript function extraction
- Test handling of nested functions
- Test handling of class methods
- Test error handling for invalid files

### test_chunker.py
- Test chunk creation from parsed functions
- Test chunk ID determinism
- Test metadata preservation
- Test large function handling

### test_embedder.py
- Test embedding generation (mock API)
- Test caching (hit and miss)
- Test batch processing
- Test error handling

### test_storage.py
- Test ChromaDB initialization
- Test upsert operations
- Test query operations
- Test stats retrieval

### test_retriever.py
- Test query embedding
- Test result ranking
- Test score filtering
- Test metadata filtering

### test_generator.py
- Test prompt construction
- Test context formatting
- Test empty results handling
- Test answer generation (mock API)

---

## 9. Success Criteria

### Functional Requirements
- [ ] `index` command successfully indexes Python/JS/TS files
- [ ] `query` command returns relevant, grounded answers
- [ ] Answers cite specific file paths and line numbers
- [ ] Re-indexing updates existing chunks (upsert)
- [ ] Caching reduces API calls on re-index

### Quality Requirements
- [ ] Retrieval returns relevant chunks (validated by manual testing)
- [ ] Answers are grounded in actual code (no hallucinations)
- [ ] Response time < 5 seconds for queries
- [ ] Index time < 1 minute for 100-file project

### Learning Validation
- [ ] Architecture decisions match Experiment 1 findings
- [ ] Can explain why cosine > euclidean
- [ ] Can explain why function-level > fixed-size chunking
- [ ] Can explain the working memory pattern

---

## 10. Implementation Order

Recommended build sequence (each step is independently testable):

1. **parser.py** → Can parse files and extract functions
2. **chunker.py** → Can create chunks from parsed functions
3. **embedder.py** → Can generate embeddings with caching
4. **storage.py** → Can store and retrieve from ChromaDB
5. **retriever.py** → Can find relevant chunks for queries
6. **generator.py** → Can generate answers with context
7. **cli.py** → Can run all commands end-to-end

Each step: implement → test → verify → proceed.

---

## 11. Example End-to-End Flow

### Index Flow
```
User runs: rag-code-qa index ./myproject

1. parser.py: Scans ./myproject, finds 50 Python files
2. parser.py: Extracts 200 functions with metadata
3. chunker.py: Creates 200 chunks with formatted content
4. embedder.py: Checks cache → 50 hits, 150 misses
5. embedder.py: Calls OpenAI API for 150 new embeddings
6. embedder.py: Caches new embeddings
7. storage.py: Upserts 200 chunks into ChromaDB

Output: "Indexed 200 functions from 50 files in 12.3 seconds"
```

### Query Flow
```
User runs: rag-code-qa query "How does user authentication work?"

1. retriever.py: Embeds query via OpenAI
2. retriever.py: Queries ChromaDB (top_k=5)
3. retriever.py: Returns 5 chunks with scores [0.82, 0.78, 0.71, 0.65, 0.52]
4. generator.py: Formats chunks as context
5. generator.py: Sends to Claude with system prompt
6. generator.py: Returns answer with citations

Output:
"Based on the codebase, authentication is handled in `src/auth/login.py:45`.
The `authenticate_user` function validates credentials against the database
and returns a JWT token. See also `src/middleware/auth.py:12` for the
middleware that validates tokens on protected routes."
```

---

## 12. Notes for Implementation

### Things That Will Bite You
1. **ChromaDB persistence**: Make sure `./data/indexed` directory exists before first run
2. **API rate limits**: OpenAI has rate limits; implement backoff
3. **Token limits**: Functions > 8000 tokens need truncation
4. **Unicode in code**: Ensure encoding is handled (UTF-8 everywhere)

### Things That Might Surprise You
1. **Embeddings are deterministic**: You can cache aggressively (Test 1 validated this)
2. **Semantic > syntactic**: "Python add" and "JavaScript add" are similar (Test 2 validated this)
3. **Length doesn't matter for cosine**: Short and verbose versions of same code are similar

### Debugging Tips
1. Use `--show-context` flag to see what chunks are being retrieved
2. Check similarity scores - low scores (< 0.5) indicate poor matches
3. Use `stats` command to verify indexing worked
4. Check `./data/cache` to verify embedding caching is working

---

*Last updated: Week 1, Day 4*
*Validated by: Experiment 1 (7 tests)*
*Next: Run experiments → extract metrics → implement components*
