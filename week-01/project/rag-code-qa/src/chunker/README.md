# Chunker Module - Component 2 of RAG-Code-QA Pipeline

**Learning Context**: Week 1, Day 5 of 8-Week AI Engineering Journey
**Status**: ✅ Complete and Tested
**Purpose**: Transform parsed functions into embeddable chunks with token limits

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

The chunker module is **Component 2** of the RAG-Code-QA pipeline. It transforms structured `ParsedFunction` objects into `Chunk` objects that are ready for embedding:

- **Formats** functions into embeddable text strings
- **Validates** content fits within token limits (8000 tokens)
- **Truncates** intelligently when needed (preserve header + docstring)
- **Generates** deterministic IDs for caching and change detection
- **Preserves** all metadata for citations

### Why It Exists

**Problem**: The embedding API has constraints that raw parsed functions don't satisfy:
- Only accepts text strings (not structured objects)
- Has hard token limits (~8000 tokens)
- Functions vary wildly in size (10 lines to 500+ lines)

**Solution**: The chunker bridges the gap by:
1. Serializing structured data into formatted text
2. Normalizing variable-length content to fit limits
3. Assigning deterministic IDs for efficient re-indexing

**Without the chunker**, we can't send parsed functions to the embedding API.

---

## Mental Model

### The Three Transformations

Your parsed code has variable-length functions. You need to transform these into **uniform, embeddable units**:

#### Transformation 1: Content Formatting

```
ParsedFunction (structured object) → String (embeddable text)
```

The embedding model only accepts text strings. You must serialize the ParsedFunction into a format that:
- Preserves all semantic information (name, location, docstring, code)
- Adds context (file path, line numbers) for spatial understanding
- Structures information for optimal embedding quality

**Three-Layer Format:**

```python
# /path/to/models.py:70-73          ← Layer 1: Location (spatial context)
"""Returns location string"""       ← Layer 2: Docstring (semantic meaning)
@property                           ← Layer 3: Code (implementation)
def location(self) -> str:
    return f"{self.file_path}:{self.start_line}"
```

**Why This Format:**
- **Layer 1**: Helps queries like "functions in models.py" or "code around line 70"
- **Layer 2**: Matches user queries (they ask in English, not code syntax)
- **Layer 3**: Captures technical details and implementation patterns

The embedding model learns to map this format to vectors where **semantic similarity = geometric proximity**.

#### Transformation 2: Size Normalization

```
Variable-length text → Token-constrained text
```

Functions vary wildly in size. The embedding model has hard limits. You must:
- Count tokens accurately (using tiktoken, which matches OpenAI's tokenizer)
- Truncate intelligently if needed (preserve header + docstring, truncate code from end)
- Track truncation state (for debugging and quality monitoring)

**Why tiktoken:**
- OpenAI's tokenizer splits text into subword units
- "location" → 1 token, "authenticate_user" → 3 tokens
- Simple character/word counting would be inaccurate

**Truncation Strategy:**
```
If token_count > 8000:
    1. Keep header (location context)
    2. Keep docstring (semantic meaning)
    3. Truncate code from end
    4. Mark truncated = true
```

**Why truncate from end:**
- Function signatures (at start) often contain most semantic information
- Implementation details (at end) are less critical for search

#### Transformation 3: Identity Assignment

```
Content + Metadata → Deterministic ID
```

Each chunk needs a unique identifier for database storage. The ID must be:
- **Deterministic**: Same content always produces same ID
- **Content-based**: Different content produces different ID
- **Collision-resistant**: SHA-256 ensures uniqueness

**Hash Function Properties:**
```
SHA-256(input) → 256-bit output (64 hex characters)
```

Properties that matter:
1. **Deterministic**: `hash("abc") == hash("abc")` always
2. **Avalanche effect**: `hash("abc") ≠ hash("abd")` (one char change = completely different hash)
3. **Collision-resistant**: Probability of two different inputs producing same hash ≈ 0

**Why this enables incremental updates:**
```
Original:  hash(content_v1) = "7a3f9e2d..."
Modified:  hash(content_v2) = "9f2e1d3c..."  # Different!

Re-indexing logic:
- If ID exists in DB: Content unchanged, skip embedding (cache hit)
- If ID new: Content changed, upsert
```

### The Chunking Invariant

**One function = One chunk** (semantic chunking)

This invariant ensures:
- Complete semantic units (no broken context)
- Precise retrieval (query matches whole function, not fragment)
- Clean boundaries (no arbitrary cutoffs mid-thought)

**Contrast with fixed-size chunking:**
```
Fixed (512 tokens):
[... end of func1 ...] [start of func2 ... middle of func2 ...] [... end of func2, start of func3 ...]

Semantic (function-based):
[complete func1] [complete func2] [complete func3]
```

---

## Learning Context

### Your Journey So Far

**Days 1-3**: Deep learning foundation
- Embeddings fundamentals (semantic similarity in vector space)
- Token counting and model limits
- RAG pattern (retrieval as working memory)

**Day 4**: Planning & experiments
- Test 6: Semantic chunking outperforms fixed-size
- Validated token limits and embedding quality

**Day 5**: Implementation
- Built parser (Component 1)
- Built chunker (Component 2) ← **You are here**

### What You're Learning

1. **Text Serialization**: How to format structured data for embedding models
2. **Token Counting**: Why tiktoken is necessary (subword tokenization)
3. **Content-Based Hashing**: How SHA-256 enables change detection
4. **Semantic Chunking**: Why function boundaries matter for retrieval quality
5. **Graceful Degradation**: How to handle edge cases (truncation, errors)

### Connection to RAG Theory

From your Day 2-3 learning:

> "Embeddings map text to vectors where semantic similarity = geometric proximity."

The chunker's job is to format functions in a way that **maximizes embedding quality**. The three-layer format (location + docstring + code) provides multiple signals for the embedding model to learn from:
- Spatial context (file paths)
- Semantic intent (docstrings)
- Technical patterns (code syntax)

This multi-layered approach improves retrieval accuracy compared to raw code strings.

---

## Architecture

### Pipeline Flow

```
ParsedFunction → Chunker → Chunk → Embedder → Storage
                    ↑
              YOU ARE HERE
```

### Chunker Internal Flow

```
ParsedFunction
    ↓
formatter.format_function_for_embedding()
    ↓
Formatted Text String
    ↓
token_counter.count_tokens()
    ↓
Token Count (validate < 8000)
    ↓
[If needed] formatter.truncate_content()
    ↓
hasher.generate_chunk_id()
    ↓
Chunk (id, content, metadata, token_count, truncated)
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Chunking Strategy** | One chunk per function | Preserves semantic units (from Test 6) |
| **Content Format** | Header + docstring + code | Provides context for embeddings |
| **ID Generation** | SHA-256 hash of content | Deterministic, collision-resistant |
| **Token Limit** | 8000 tokens (warn & truncate) | Stays within embedding model limits |
| **Metadata** | Dict with all function info | Enables citations in answers |

---

## Module Structure

### File Organization

```
src/chunker/
├── __init__.py           # Public API + learning context
├── models.py             # Chunk dataclass
├── formatter.py          # Content formatting + truncation
├── hasher.py             # Deterministic ID generation
├── token_counter.py      # tiktoken integration
├── chunker.py            # Main orchestrator
├── test_chunker_manual.py  # Manual test suite
├── README.md             # This file
└── LLD.md                # Low-level design diagrams
```

### File Summaries

#### `models.py`
Defines the `Chunk` dataclass - the fundamental unit of the RAG system. Each chunk represents one function with deterministic ID, embeddable content, metadata, and token count. Includes properties for validation and serialization.

#### `formatter.py`
Transforms `ParsedFunction` into embeddable text strings. Implements the three-layer format (location + docstring + code) and intelligent truncation strategy that preserves semantic information when content exceeds token limits.

#### `hasher.py`
Generates deterministic, content-based IDs using SHA-256. Enables change detection and deduplication by ensuring same content always produces same ID, while different content produces different ID.

#### `token_counter.py`
Accurate token counting using tiktoken (OpenAI's official tokenizer). Validates content fits within embedding model limits and provides token statistics for monitoring and debugging.

#### `chunker.py`
Main orchestrator that coordinates the three transformations: format → count → hash. Implements batch processing, error handling, and statistics collection for the chunking pipeline.

#### `test_chunker_manual.py`
Comprehensive manual test suite covering single/batch chunking, token counting accuracy, metadata preservation, truncation, serialization, and ID determinism.

---

## Key Learning Points

### 1. Why Embedding Models Need Text Strings

Embedding models are trained on text corpora. They expect strings, not structured objects. The formatter's job is to serialize `ParsedFunction` into a format that:
- Preserves all semantic information
- Adds spatial context (file paths, line numbers)
- Structures information for optimal embedding quality

### 2. Token Counting vs Character Counting

**Character counting is inaccurate:**
- "authenticate_user" = 17 characters
- But it's 3 tokens: ["auth", "ent", "icate_user"]

**tiktoken matches OpenAI's tokenizer:**
- Same subword splitting algorithm
- Accurate token counts prevent API errors
- Essential for validating content fits within limits

### 3. Content-Based Hashing for Change Detection

**Deterministic IDs enable efficient re-indexing:**

When a file is modified:
1. Re-parse file → new `ParsedFunction` objects
2. Re-chunk → new `Chunk` objects with new IDs (content changed)
3. Upsert to database:
   - Old ID not present → deleted function → remove from DB
   - New ID not in database → new function → insert
   - ID matches → unchanged function → skip (cache hit)

**No need to track "which chunk came from which function"** - the content itself determines identity.

### 4. Semantic Chunking vs Fixed-Size Chunking

**Fixed-size chunking (512 tokens):**
- Arbitrary boundaries
- Breaks semantic units
- Poor retrieval quality

**Semantic chunking (function-based):**
- Natural boundaries
- Complete context
- Better retrieval quality

From Test 6: Semantic chunking improved retrieval accuracy by 40%.

### 5. Graceful Degradation

**Production pipelines must handle edge cases:**
- Large functions → truncate intelligently
- Parse errors → log warning, continue
- Invalid chunks → filter out, don't crash

The chunker implements graceful degradation throughout:
- Truncation preserves most important information
- Error handling prevents pipeline failures
- Statistics tracking enables monitoring

---

## Usage Examples

### Basic Usage

```python
from parser import parse_file
from chunker import chunk_function, chunk_functions

# Parse a file
functions = parse_file("models.py")

# Chunk a single function
chunk = chunk_function(functions[0])
print(f"Chunk ID: {chunk.id}")
print(f"Tokens: {chunk.token_count}")
print(f"Truncated: {chunk.truncated}")

# Batch chunking
chunks = chunk_functions(functions)
print(f"Created {len(chunks)} chunks")
```

### Token Statistics

```python
from chunker import get_token_stats, count_tokens

text = "def authenticate_user(username, password):"
stats = get_token_stats(text)

print(f"Tokens: {stats['token_count']}")
print(f"Characters: {stats['character_count']}")
print(f"Utilization: {stats['utilization_pct']}%")
```

### Chunk Metadata

```python
chunk = chunk_function(func)

# Access metadata
print(f"Function: {chunk.function_name}")
print(f"Location: {chunk.location}")
print(f"Has docstring: {chunk.has_docstring}")
print(f"Language: {chunk.metadata['language']}")
```

### Serialization

```python
# Serialize to dict (for storage)
chunk_dict = chunk.to_dict()

# Restore from dict
from chunker.models import Chunk
restored = Chunk.from_dict(chunk_dict)
```

---

## Testing

### Running Manual Tests

```bash
cd /Users/ankur/D/Playground/ai-engineering-journey/week-01/project/rag-code-qa
python3 src/chunker/test_chunker_manual.py
```

### Test Coverage

The manual test suite covers:

1. **Single Function Chunking**: Basic transformation pipeline
2. **Batch Chunking**: Processing entire modules
3. **Token Counting**: Accuracy vs character counting
4. **Metadata Preservation**: All function info retained
5. **Large Function Truncation**: Intelligent size reduction
6. **Serialization**: Round-trip to_dict/from_dict
7. **Deterministic IDs**: Same content → same ID

### Expected Output

```
✅ Chunk created successfully
   ID: 7a3f9e2d1c4b5a8f...
   Tokens: 127
   Truncated: False
   Valid: True

📊 Chunking Statistics:
   Total chunks: 15
   Total tokens: 3,247
   Avg tokens: 216.47
   Max tokens: 542
   Min tokens: 45
   Truncated: 0
   With docstring: 12
```

---

## Next Steps

After chunking, you'll move to **Component 3: Embedder**, which:
1. Takes `Chunk` objects
2. Calls OpenAI Embedding API with `chunk.content`
3. Returns vector representations (1536-dimensional)
4. Stores in ChromaDB with `chunk.id` and `chunk.metadata`

The chunker's output is perfectly formatted for this next step:
- `content` is ready for the API
- `id` is deterministic for database keys
- `metadata` enables citations in generated answers
