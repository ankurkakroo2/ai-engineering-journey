# Day 3: Hands-On Experimentation - Running Notes

**Date**: Monday, December 30, 2025
**Focus**: Validating Day 1-2 learning through practical experiments with embeddings, similarity metrics, storage, and chunking strategies
**Strategy**: Detailed running notes during experiments, will synthesize in `experiments.md` and daily-log at end of day

⚠️ **What This Is**: Hands-on experimentation notes—capturing observations, "aha moments," code behavior, and connections as they happen. Not just templates, but filled with actual findings.

---

## Foundation Coming Into Day 3

### From Day 1: Core Concepts Validated
**Embeddings & Semantic Meaning:**
- Embeddings are learned vectors (768 dimensions for transformers, 1536 for OpenAI)
- Similar semantic concepts cluster together in embedding space
- Distance metrics (cosine similarity) measure semantic relatedness

**Transformers & Attention:**
- Attention mechanism creates weighted relationships between all tokens
- Query/Key/Value matrices enable flexible, context-aware lookups
- 12 layers of progressive refinement build contextual understanding

**Learning & Weight Updates:**
- Parameters learned through gradient descent over training examples
- Training: weeks/months; Inference: milliseconds with frozen weights
- Larger embeddings naturally produce stronger signals (LSTM insight manifesting)

### From Day 2: Complete LLM Pipeline Understanding

**Checkpoints 1-2 (Completed):**
- **Pretraining**: Text → BPE Tokenization → 768-dim embeddings → 12 layers of attention → next token prediction
- **Post-Training**: Base model (internet simulator) → Fine-tuning on conversation data → Helper behavior learned through human labeling preferences
- **Key Insight**: Quality of labeled training data > Quantity (InstructGPT paper)

**Checkpoints 3+ (Currently Watching):**
- **Hallucinations**: No "I don't know" state—models predict plausibly even when wrong
  - More likely with uncommon/obscure information
  - No internal uncertainty mechanism
- **Tool Use**: Special tokens trigger orchestrator → actual tool execution → context updated → model continues
  - Examples: Calculator, Web Search
  - Expands model beyond static training data
- **Knowledge vs. Working Memory**:
  - Knowledge = pre-trained parameters (vague, compressed)
  - Working Memory = current prompt context (specific, current)
  - Quality improves with working memory over knowledge
- **Models Need Tokens to Think**: Tokens ARE the thinking process, not byproduct
  - Chain-of-Thought works because intermediate tokens provide multiple passes through the transformer
  - Fewer tokens = fewer passes = lower accuracy
  - More tokens = more context for 12 layers = higher probability of right answer
  - Labelers prefer detailed reasoning (learned in post-training)

### Why Day 3 Matters: Theory Meets Practice

**The Connection You're Building:**
- Day 1 taught: How transformers work (mechanisms)
- Day 2 taught: How LLMs are trained (pretraining → post-training → alignment)
- Day 3 proves: All of this actually works in real code

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

## CHECKPOINT 3: Hallucinations, Tool Use, Knowledge & Working Memory (1:20:32 onwards)

### Hallucinations: Models Always Predict Something (1:20:32)

**The Fundamental Truth:**
Models are text prediction machines. They predict the next token based on learned patterns from training data, NOT based on truth or understanding. This is critical to understand.

**Why Hallucinations Happen:**

When a model encounters information it:
- Doesn't know well
- Hasn't seen in training data
- Finds contradictory in training data
- Finds ambiguous

The model still MUST output something. It has no internal "I don't know" state. Instead, it predicts a plausible continuation based on patterns learned from training data.

**Probability of Hallucinations:**
- **Common information** (e.g., "What is 2+2?"): Low hallucination (patterns are clear)
- **Uncommon/obscure information** (e.g., "Name a specific person in this room"): High hallucination (patterns are vague or absent)
- **Specific facts with few training examples**: Higher hallucination (learned patterns are weaker)

**The Mathematical Truth:**
With 50,277 possible next tokens and softmax probabilities, the model outputs its best guess even with low confidence. If confidence for true answer is 0.3% and confidence for plausible-sounding hallucination is 0.8%, the model outputs the hallucination.

**Implications for Day 3:**
- Your embedding search might retrieve correct code, but if model doesn't know context, it will hallucinate relationships
- Tool use (providing updated context) helps counteract this
- Working memory (current prompt context) is more trusted than knowledge (parameters)

### Tool Use: Special Tokens as API Calls (1:24:25)

**The Architecture You Need to Understand:**

Models learn (during post-training) to output special tokens that trigger actions:
- Model outputs: `<CALCULATOR>`
- System intercepts this token
- System runs actual calculator
- System appends result to context: `<CALCULATOR_RESULT: 147714</CALCULATOR_RESULT>`
- Model continues generating with updated context

**Why This Works:**

1. **Post-training data is general enough**: Human labelers in post-training wrote conversations that included tool use examples naturally
   - "I need to calculate 323 * 458... let me use a calculator... [result]"
   - Model learned these patterns

2. **Hidden system prompt describes tools**: Before user input, system adds:
   ```
   You have access to: calculator, web_search, code_executor
   To use a tool, output [TOOL_NAME:argument]
   ```

3. **Orchestrator pattern**:
   - Model outputs tool token → Orchestrator sees it → Executes tool → Returns result → Model sees result in context
   - Flow: Model → Special Token → Orchestrator → Tool → Updated Context → Model

**Examples in Practice:**

**Calculator Tool:**
- User: "What is 323 * 458?"
- Model (thinking): I should use the calculator
- Model outputs: `[CALCULATOR:323*458]`
- System returns: `[CALCULATOR_RESULT:147,714]`
- Model continues: "The answer is 147,714"

**Web Search Tool:**
- User: "What's the weather in New York today?"
- Model: "Let me search for current weather"
- Model outputs: `[WEB_SEARCH:weather in New York today]`
- System returns: `[SEARCH_RESULT:Sunny, 72°F, 45% humidity]`
- Model continues: "The weather in New York today is sunny, 72°F..."

**Why This Expands Capability:**
- Overcomes static training data limitation
- Provides real-time information
- Enables accurate computation beyond LLM's math abilities
- Shows how pre-trained knowledge + tool use = better assistant

**Implication for Day 3:**
- Your semantic search tool is like a "retrieval tool"
- Good chunking strategy = good tool output = better model response
- Bad chunking strategy = irrelevant code chunks = model hallucinates

### Knowledge vs. Working Memory: Two Types of Information (1:31:00)

**Knowledge: What Lives in Parameters (1:31:30)**

This is information the model learned during pre-training and stored in its 1.6 billion parameters.

**Characteristics:**
- Compressed and statistical: "cats are furry animals that meow"
- Vague and probabilistic: Model can't distinguish "cat" the animal from "cat" the name
- Unreliable for specific facts: If model saw something 1000 times in training, it's fuzzy
- Examples: General facts, language patterns, common sense reasoning

**How it's stored:**
- Not as explicit knowledge like "Paris is the capital of France"
- As learned patterns in weight matrices: Patterns that activate when discussing French cities tend to mention Paris

**Limitation:**
- Hallucinations come from knowledge: Model doesn't KNOW, so it guesses based on patterns
- Outdated: Training data cutoff (April 2024) means no knowledge of recent events
- Vague connections: Can't be certain about specific relationships

**Working Memory: What's in the Current Prompt (1:33:00)**

This is the information provided in the current conversation—the full context window available to the model right now.

**Characteristics:**
- Specific and current: "In this document, cats are defined as..."
- Reliable: Model can reference exact text from prompt
- Immediate: Available for this conversation, doesn't decay
- Examples: Uploaded documents, previous messages, system instructions

**Why Quality Improves with Working Memory:**
If knowledge says "cats meow" but your prompt says "In this company, 'cat' is our mascot named Mr. Whiskers who barks," the model trusts the prompt.

Model learns in post-training: "Information in working memory is more specific and reliable than general knowledge."

**In-Context Learning Relies on Working Memory (1:35:00):**

This is the ability to learn from examples in the current prompt:

```
Example from prompt:
"Here's how to translate English to Pig Latin:
- hello → ello-hay
- computer → omputer-cay

Now translate: programmer"
```

Model learns the pattern from these 2 examples in the prompt (working memory), not from its pre-trained knowledge. It didn't see "Pig Latin" in training data with confidence, but the examples in working memory teach it the pattern.

**The Tradeoff (1:39:00):**

**Larger context windows** (more working memory):
- **Pros**: More information, better reasoning, less hallucination, can reference longer documents
- **Cons**: More computation, slower processing, higher API costs

This is why:
- ChatGPT has 128K token context window (massive working memory)
- But it's slower and more expensive than smaller context windows
- Trade-off: Accuracy vs. Speed/Cost

**Implication for Day 3:**
- Your prompt + retrieved code chunks = working memory
- If chunks are relevant (good chunking strategy) = good working memory = better answer
- If chunks are irrelevant (bad chunking strategy) = garbage working memory = hallucination

### Knowledge of Self: Pure Imitation (1:41:46)

**The Honest Truth:**

When a model says "As a large language model, I...", it's not expressing self-awareness. It's completing a learned text pattern.

**Why Models Say This:**

In training data (especially from websites discussing AI), humans write things like:
- "As a large language model, I can..."
- "As an AI assistant, my purpose is..."
- "I'm Claude, an AI made by Anthropic..."

Model learned these phrases from MILLIONS of examples. They're natural continuations in datasets.

**It's Imitation, Not Consciousness:**

Model is not thinking about itself. It's completing:
- `User: "Who are you?"`
- `Model sees pattern: Similar questions usually followed by "I am..."`
- `Model outputs: "I am Claude, an AI assistant"`

No self-awareness involved. Pure learned pattern matching.

**Why This Matters:**

Helps you debug model behavior:
- When model says "I can't do X," it learned this phrase from training data
- When model makes mistakes about itself ("I was created in 2020"), it's repeating training data patterns
- No hidden "self" to appeal to, just learned statistical patterns

**Implication for Day 3:**
- Don't treat model as having preferences or desires
- Treat tool outputs (your code chunks) as information the model will process
- Quality of information = quality of model response

### Models Need Tokens to Think: The Chain-of-Thought Revolution (1:46:56)

**Core Insight: Tokens ARE the Thinking**

The model's "thought process" is not happening before it generates tokens. The generation of tokens IS the thinking.

**Why This Matters:**

To solve a complex problem, the model needs to output intermediate tokens that represent intermediate steps.

**Example Problem:**
```
Q: If you have 5 apples and someone gives you 3 more, then you give away 2,
what's the final count?

Direct answer attempt:
- Model might predict: "3" (wrong—didn't do math)

Step-by-step attempt:
- Model predicts: "I have 5 apples initially"
- Tokens represent "5 apples" concept
- Model then predicts: "Plus 3 more = 8 apples"
- Tokens represent "8 apples" concept, which influences next prediction
- Model then predicts: "I give away 2"
- Token represents "2 given away"
- Model finally predicts: "Final count = 6"
```

**Chain-of-Thought Prompting (1:48:00):**

Technique: Explicitly instruct the model to "think step by step"

```
System: "Let's work through this step-by-step"
User: "100 + 200 * 3 = ?"

Without CoT:
- Model might predict: "900" (wrong—didn't follow order of operations)

With CoT:
- Model: "First, I need to follow order of operations (PEMDAS)"
- Model: "Multiplication comes before addition"
- Model: "So 200 * 3 = 600"
- Model: "Then 100 + 600 = 700"
```

**Why Counting Problems Fail: The Inverse of CoT**

This is the famous strawberry example Karpathy highlighted:

```
Question: "How many r's are in the word 'strawberry'?"

Direct attempt (no intermediate tokens):
- Model tries to count in a single forward pass
- No intermediate thinking steps to constrain the answer
- Result: Often gets it wrong (0, 1, or 2 instead of correct answer: 3)

Step-by-step attempt:
- Model: "Let me spell it out: s-t-r-a-w-b-e-r-r-y"
- Tokens represent each letter as context
- Model: "Now counting the r's: position 3 (r), position 8 (r), position 9 (r)"
- Model: "Total: 3 r's in strawberry"
```

**Why This Happens:**

The model doesn't have a built-in "counter" component. To solve counting, it must:
1. Generate intermediate tokens representing the counting process
2. Feed those tokens back through the 12-layer transformer
3. Use the enriched context to predict the next step
4. Repeat until the count is complete

**If you force single-pass prediction:**
- No intermediate tokens = no thinking process
- Model must predict the answer without "computational" steps
- This is not how the transformer was trained (training data had step-by-step examples)
- Result: Hallucination or wrong count

**The Deep Truth:**

Models don't "count" the way we think. They predict what tokens would typically come next after seeing similar patterns in training data. Counting is not a native operation—it's something the model learned to simulate through token generation.

---

**Why CoT Works: YOUR KEY INSIGHT**

- Tokens are inputs to the transformer
- Transformer (12 layers) is the brain
- Each pass through the 12 layers processes tokens as context

**The Passes Through Brain:**
- **Fewer tokens** = Fewer things for 12 layers to process = Less contextual refinement = Lower accuracy
- **More tokens** = More context fed into each pass = 12 layers have richer input to work with = Higher probability of correct answer
- **No intermediate tokens** (forcing single-pass counting) = Model has no "thinking space" = Forced to guess without reasoning = Higher error rate

**Why Labelers Prefer Detailed Answers:**

In post-training, humans labeled thousands of examples. They naturally preferred:
- Answers with reasoning: "Let me think... Step 1... Step 2... Therefore..."
- Over direct answers: "42"

Model learned this preference. Longer, token-rich responses with intermediate steps scored higher in training.

This is WHY CoT works: It's mimicking what human labelers preferred in training data.

**Computational Cost:**
- More tokens = longer context window = more computation
- Must be processed through all 12 layers
- Slower and more expensive
- But: Higher accuracy justifies cost for complex reasoning

**Implication for Day 3:**
- When retrieving code chunks (your "thought tokens"), quality matters
- Multiple relevant chunks = richer context = better reasoning
- Single chunk = limited context = higher hallucination risk
- This guides your chunking strategy decision

---

## CHECKPOINT 4: Remaining Sections (Placeholder for Notes as You Watch)

### Tokenization Revisited: Where Models Struggle (Coming 02:04:53)
- [ ] How models struggle with spelling
- [ ] Edge cases where tokenization breaks
- [ ] Why token boundaries matter for reasoning
- **Notes to capture**: Examples of tokenization failures, implications for semantic search

### Jagged Intelligence: Why Models Are Inconsistent (Coming 02:07:28)
- [ ] Why models are brilliant at some tasks, terrible at others
- [ ] Domain expertise unevenness
- [ ] How training data distribution creates "jagged" capabilities
- **Notes to capture**: Examples across domains, how this affects code search

### Supervised Fine-Tuning to Reinforcement Learning (Coming 02:07:28)
- [ ] Transition from SFT to RL training
- [ ] Why RL needed after SFT
- [ ] Difference in training signals
- **Notes to capture**: How this affects model behavior

### Reinforcement Learning (Coming 02:14:42)
- [ ] How RL optimizes model behavior
- [ ] Reward signals and learning
- [ ] Practice makes perfect concept
- **Notes to capture**: Why RL outputs better behavior

### DeepSeek-R1 & AlphaGo (Coming 02:27:47 onwards)
- [ ] Real-world examples of reinforcement learning
- [ ] How reasoning emerges from RL
- [ ] Search and planning in LLMs
- **Notes to capture**: Practical implications for your system

### RLHF: Alignment Through Human Feedback (Coming 02:48:26)
- [ ] How humans guide model behavior
- [ ] Preference learning from human rankings
- [ ] Aligning models with human values
- **Notes to capture**: Why models behave helpfully

### Preview & Future Directions (Coming 03:09:39 onwards)
- [ ] Where LLM research is heading
- [ ] Emerging capabilities
- [ ] Open problems
- **Notes to capture**: Future considerations for your work

### Summary & Key Takeaways (Coming 03:21:46 onwards)
- [ ] Synthesis of entire training pipeline
- [ ] Key insights to remember
- **Notes to capture**: Final mental model

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
From Day 3 goal: Can embeddings capture semantic meaning of code across languages?

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
- [ ] Implications for Day 4 implementation:

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
- [ ] Decision for Day 4 semantic search: Use ___ because ___

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

From Day 2: "Chunking strategy is critical—impacts both storage and search quality"

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
- [ ] Will adjust for Day 4 by: ___

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

### Critical Insights for Day 4 Implementation

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

### Confidence & Readiness for Day 4

- Confidence in understanding: High / Medium / Low
- Ready to implement? Yes / Need more investigation
- Key uncertainties remaining: ___

---

*These notes document the hands-on validation of Day 1-2 theory. You're not just learning how embeddings work—you're seeing them work in real code. This bridges the gap between understanding and building.*
