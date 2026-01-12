# Generator Module - Answer Generation with Claude

## Learning Context

This is **Component 6** of your RAG-Code-QA pipeline (Week 1, Day 8-9).

After retrieving relevant code chunks, you now use Claude to generate natural language answers **grounded in actual code**. This is where the RAG pattern comes together to prevent hallucination.

**Previous Components:**
1. ✅ Parser - Extract functions from code
2. ✅ Chunker - Create embeddable units
3. ✅ Embedder - Generate vector embeddings
4. ✅ Storage - Persist to ChromaDB
5. ✅ Retriever - Semantic search

**This Component:**
6. **Generator** - Generate answers with Claude

**Next Component:**
7. CLI - Command-line interface

---

## Mental Model: How RAG Prevents Hallucination

### The Problem: LLM Hallucination

Without RAG, LLMs make up answers about code they haven't seen:

```
User: "How does authentication work in this codebase?"
LLM (without RAG): "The system uses JWT tokens with bcrypt hashing..."
                   ↑ HALLUCINATION - made up, not from your code
```

**Why this happens:**
- LLMs are trained on public code, not YOUR codebase
- They generate plausible-sounding answers based on patterns
- No way to verify if the answer is correct for YOUR code

### The Solution: RAG Pattern

With RAG, LLMs answer based on actual retrieved code:

```
User: "How does authentication work?"
   ↓
Retriever: Finds relevant chunks (login_handler, verify_token, etc.)
   ↓
Generator: Passes chunks as context to Claude
   ↓
Claude: "Based on login_handler() in auth.py:45-67,
         authentication uses session tokens stored in Redis..."
         ↑ GROUNDED - cites actual code from YOUR codebase
```

**Why this works:**
- Claude sees the actual code in the prompt
- Answers based on what it sees, not what it imagines
- Can cite specific files and line numbers for verification

### Retrieved Chunks as "Working Memory"

Think of it like an open-book exam:

**Without RAG (closed book):**
- Claude guesses based on training data
- May reference patterns from other codebases
- No way to verify accuracy
- High risk of hallucination

**With RAG (open book):**
- Claude has the exact code in front of it
- Answers based on what it sees
- Can cite specific files and line numbers
- Low risk of hallucination (if prompted correctly)

**The key:** Retrieved chunks act as Claude's "working memory" for this specific question.

### The Context Window Budget

Claude has a large context window (~200K tokens), but we need to be strategic:

```
Context Budget: ~8000 tokens
├── System prompt: ~200 tokens (instructions for Claude)
├── Retrieved code: ~6000 tokens (5 chunks × ~1200 tokens each)
├── User question: ~100 tokens
└── Answer: ~1700 tokens (reserved for response)
```

**Why limit context?**
1. **Better focus:** Too much context = confusion. Quality over quantity.
2. **Faster responses:** Less context = faster generation.
3. **Cost efficiency:** Tokens cost money. Be strategic.
4. **Leave room for answer:** Need space for detailed responses.

**Strategy:**
- Retriever returns top 5-10 most relevant chunks
- Context builder formats and truncates to fit budget
- Claude focuses on most relevant code

### Citation-Based Answers

Every claim should reference source code:

```
Bad Answer:
"The system validates tokens."
↑ No citation - can't verify

Good Answer:
"The system validates tokens in verify_token() (auth/tokens.py:23-41),
which checks expiration using datetime.now() and signature using
hmac.compare_digest() with the SECRET_KEY."
↑ Cites file:line - can verify in codebase
```

**Why citations matter:**
- **Verifiability:** Can check if answer is correct
- **Trust:** Shows answer is grounded in code
- **Navigation:** Helps user find relevant code
- **Debugging:** If wrong, can see which code was used

---

## Module Architecture

### High-Level Flow

```
Query + RetrievalResults
         ↓
    Generator Module
         ↓
    ┌────────────────┐
    │ Context Builder│  Format chunks with file:line
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Prompt Builder │  Build system + user prompts
    └────────────────┘
         ↓
    ┌────────────────┐
    │ Claude Client  │  Call Anthropic API
    └────────────────┘
         ↓
    GenerationResult (answer + metadata)
```

### Module Structure

```
src/generator/
├── __init__.py              # Public API
├── models.py                # GenerationResult dataclass
├── context_builder.py       # Format retrieved chunks for Claude
├── prompt_builder.py        # Build system and user prompts
├── claude_client.py         # Anthropic API integration
├── generator.py             # Main orchestrator (facade)
├── README.md                # This file
├── LLD.md                   # Low-level design diagram
└── test_generator_manual.py # Manual validation script
```

### File Responsibilities

**models.py:**
- `GenerationResult`: Answer + metadata (tokens, timing, citations)
- `check_citations()`: Detect file:line references
- `extract_code_blocks()`: Find code in answer
- Statistics and formatting helpers

**context_builder.py:**
- `format_chunk()`: Format single chunk with [1] file.py:10-20
- `build_context()`: Format all chunks, check token count
- `truncate_results()`: Fit chunks within token budget
- Token counting and validation

**prompt_builder.py:**
- `SYSTEM_PROMPT`: Instructions for Claude (cite sources, no hallucination)
- `build_user_prompt()`: Context + Question format
- `build_messages()`: Complete message list for API
- `build_no_context_prompt()`: Handle empty results

**claude_client.py:**
- `create_claude_client()`: Initialize from ANTHROPIC_API_KEY
- `call_claude_api()`: Make API call with messages
- `call_with_retry()`: Exponential backoff for rate limits
- `estimate_cost()`: Calculate API cost from tokens

**generator.py:**
- `generate_answer()`: Main entry point (facade)
- `generate_with_retrieval()`: Complete RAG pipeline
- Statistics tracking and logging

---

## Key Learning Points

### 1. RAG Prevents Hallucination

**Without RAG:**
```python
# Claude has never seen your code
answer = claude.ask("How does auth work?")
# Result: Generic answer about JWT/OAuth/etc. (hallucination)
```

**With RAG:**
```python
# Retrieve actual code
results = retrieve("How does auth work?", top_k=5)

# Claude sees YOUR code
answer = generate_answer("How does auth work?", results)
# Result: Specific answer citing your auth.py:45-67 (grounded)
```

### 2. Context Formatting Matters

**Bad formatting:**
```
Here's some code: def login(u,p): return check(u,p)
And more: def verify(t): return valid(t)
Question: How does auth work?
```

**Good formatting:**
```
Context:
[1] auth/handlers.py:45-67
```python
def login_handler(username, password):
    # Validate credentials
    ...
```

[2] auth/tokens.py:23-41
```python
def verify_token(token):
    # Check expiration
    ...
```

Question: How does authentication work?
```

**Why good formatting helps:**
- Clear structure (Context: ... Question: ...)
- Numbered references ([1], [2]) for citations
- File:line locations for navigation
- Code blocks preserve formatting

### 3. Prompt Engineering is Critical

**System prompt sets behavior:**
```python
SYSTEM_PROMPT = """
1. Answer based ONLY on the provided code
2. Always cite file and line numbers
3. If answer isn't in context, say so
"""
```

**Key principles:**
- Be explicit about rules
- Emphasize citations
- Forbid hallucination
- Use temperature=0.0 for deterministic responses

### 4. Token Budget Management

**Strategy:**
```python
# Retriever gets top 10 (might be too much)
results = retrieve(query, top_k=10)

# Context builder truncates to fit budget
truncated = truncate_results(results, max_tokens=6000)

# Only use what fits
context = build_context(truncated)
```

**Trade-offs:**
- More context = better coverage, but slower/costlier
- Less context = faster/cheaper, but might miss relevant code
- Sweet spot: 5-7 chunks (~6000 tokens)

### 5. Citation Detection

**Regex pattern:**
```python
pattern = r'([a-zA-Z0-9_/\-\.]+\.py):\s*(\d+)\s*-\s*(\d+)'
# Matches: file.py:10-20, path/to/file.py:45-67
```

**Quality check:**
```python
if result.has_citations:
    print("✓ Answer is grounded")
else:
    print("⚠ Answer may be hallucinated")
```

---

## Usage Examples

### Basic Generation

```python
from src.retriever import retrieve
from src.generator import generate_answer

# Step 1: Retrieve relevant code
query = "How does authentication work?"
results = retrieve(query, top_k=5, min_score=0.3)

# Step 2: Generate answer
answer = generate_answer(query, results)

# Step 3: Display
print(answer.answer)
print(f"\nSources cited: {answer.sources_cited}")
print(f"Tokens used: {answer.total_tokens}")
print(f"Generation time: {answer.generation_time:.2f}s")
```

### Complete RAG Pipeline (One-Shot)

```python
from src.generator import generate_with_retrieval

# Retrieve + Generate in one call
answer = generate_with_retrieval(
    query="How does authentication work?",
    top_k=5,
    min_score=0.3
)

print(answer.answer)
```

### Handling Empty Results

```python
# Query with no relevant code
answer = generate_answer("How to bake a cake?", [])

# Result: Helpful message about no code found
print(answer.answer)
# "I couldn't find any relevant code for your question..."
```

### Custom Token Budgets

```python
# More context, shorter answer
answer = generate_answer(
    query=query,
    retrieval_results=results,
    max_context_tokens=8000,  # More context
    max_answer_tokens=1024    # Shorter answer
)

# Less context, longer answer
answer = generate_answer(
    query=query,
    retrieval_results=results,
    max_context_tokens=4000,  # Less context
    max_answer_tokens=3000    # Longer answer
)
```

### Monitoring Statistics

```python
from src.generator import get_generation_stats

# After multiple generations
stats = get_generation_stats()

print(f"Total generations: {stats['total_generations']}")
print(f"Average tokens: {stats['avg_tokens_per_generation']:.1f}")
print(f"Average time: {stats['avg_time_per_generation']:.2f}s")
print(f"Citation rate: {stats['citation_rate']:.1f}%")
```

---

## Integration Points

### Input: RetrievalResult from Retriever

```python
from src.retriever.models import RetrievalResult

# Generator expects List[RetrievalResult]
results: List[RetrievalResult] = retrieve(query)
answer = generate_answer(query, results)
```

### Output: GenerationResult for CLI

```python
from src.generator.models import GenerationResult

result: GenerationResult = generate_answer(query, results)

# CLI can use:
result.answer              # Display to user
result.sources_cited       # Show citations
result.format_for_display() # Pretty print
```

### Environment Variables

Required in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
```

Optional:
```bash
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MAX_TOKENS=2048
ANTHROPIC_TEMPERATURE=0.0
```

---

## Testing

### Manual Test Script

Run the manual test:
```bash
cd /Users/ankur/D/Playground/ai-engineering-journey/week-01/project/rag-code-qa
python src/generator/test_generator_manual.py
```

**Test cases:**
1. Basic generation (with good retrieval results)
2. Empty results handling (no relevant code)
3. Citation verification (answer includes file:line)
4. Token limit handling (truncate long context)
5. Error handling (invalid API key)
6. Statistics tracking

### Expected Output

**With good results:**
```
Query: How does the visitor pattern work?
Answer: The visitor pattern is implemented in FunctionVisitor
        (parser/python_parser.py:23-67), which traverses the AST...

Sources: ['parser/python_parser.py:23-67', 'parser/models.py:15-30']
Tokens: 1523 (prompt: 1234, completion: 289)
Time: 2.34s
```

**With no results:**
```
Query: How to bake a cake?
Answer: I couldn't find any relevant code for your question...

Sources: []
Tokens: 0
Time: 0.01s
```

---

## Next Steps

After completing the generator:
1. **CLI Module** - Command-line interface for `index` and `query` commands
2. **End-to-End Testing** - Full pipeline from code to answer
3. **Documentation** - Update project README with complete workflow
4. **Demo** - Record video showing the tool in action

The generator completes the RAG pattern. You can now ask questions and get grounded, verifiable answers!
