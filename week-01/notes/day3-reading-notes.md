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

### From Day 2 (Deep Dive Video - Pretraining & Post-Training & Beyond) 🔄 IN PROGRESS
**Completed Checkpoints:**
- Complete LLM pipeline: Text → Tokens → Embeddings → 12 Transformer Layers → Output
- Tokenization via BPE (Byte Pair Encoding)
- Query/Key/Value attention mechanism with softmax
- Post-training teaches helpful assistant behavior
- Base model = internet simulator → Post-training = helpful labeler simulator

**Coming from remaining video sections:**
- How models hallucinate and what causes them
- Tool use and how models leverage external knowledge
- Knowledge/working memory limitations and patterns
- Tokenization struggles with spelling and edge cases
- Why model intelligence is "jagged" across domains
- Full training spectrum: Supervised fine-tuning → Reinforcement Learning → RLHF
- Real-world examples: DeepSeek-R1, AlphaGo, and alignment strategies
- Practical applications and where to find/track LLMs

### Why Day 3 Matters (After Complete Day 2 Foundation)
**Theory to Practice Translation:**
- You'll understand BOTH:
  - How LLMs are trained (pretraining → fine-tuning → RL → RLHF alignment)
  - Why they behave the way they do (hallucinations, tool use, knowledge limits)
- You'll see how chunking strategy directly affects semantic search quality
- You'll understand the tradeoffs: embedding dimensions, similarity metrics, storage approaches
- You'll build intuition for Day 4-5 implementation decisions based on real constraints

**Practical Validation:**
- See embeddings, similarity metrics, storage, and chunking work in real code
- Connect "jagged intelligence" concept to code snippet relevance
- Understand why certain chunking strategies fail (tokenization issues)
- Build confidence that theory works in practice

---

## CHECKPOINT 3: Hallucinations, Tool Use, Knowledge/Working Memory (1:20:32 onwards)

### Hallucinations (1:20:32)
- Models are text prediction machines - predict next token based on patterns, not truth
- When models don't know: still predict plausible continuation → "hallucinations" (factually incorrect/nonsensical)
- Hallucinations more likely with obscure/uncommon information from training data
- **Key insight**: No internal "I don't know" state - just probability-based prediction
- This is fundamental to how transformers work - always output something

### Tool Use (1:24:25)
**Mental Model Established:**
- Every response flows through special tokens
- Post-training data is general enough to recognize when tools are needed
- Hidden system prompt describes available tools
- Flow: Model outputs tool token → Orchestrator intercepts → Calls actual tool → Passes updated context back to model → Gets final answer with current information

**Examples:**
- Calculator (1:24:50): Model outputs token for calculator tool, system runs calculation, inserts result back into context
- Web Search (1:26:00): Model outputs search token, system performs search, results fed back as context, model continues with current info
- **Outcome**: Expands capabilities beyond static training data, improves accuracy for factual/real-time queries

### Knowledge vs. Working Memory (1:31:00)
**Knowledge (1:31:30):**
- Information "memorized" in parameters during pre-training
- Compressed, statistical recollection of internet
- Vague and probabilistic

**Working Memory (1:33:00):**
- Current context in the prompt
- Short-term memory analogy
- **Key insight**: Quality/specificity improves when in working memory vs. relying on stored knowledge

**In-Context Learning (1:35:00):**
- Model learns new patterns from examples in current prompt
- Relies on working memory

**Tradeoff (1:39:00):**
- Larger context windows = more computation but better reasoning, less hallucination
- This explains why longer prompts with examples help

### Knowledge of Self (1:41:46)
- Model has no "self" or consciousness
- Statements like "As a large language model..." are text patterns from training data
- Common phrases learned from datasets where people describe AI
- **Fundamental truth**: Imitation and mimicry based on observed text, not self-awareness

### Models Need Tokens to Think (1:46:56)
**Core Insight:**
- Model's "thought process" = sequential token generation
- To solve complex problems: needs to output intermediate tokens
- **Tokens ARE the thinking**, not a byproduct

**Chain-of-Thought Prompting (1:48:00):**
- Instruct model to "think step by step"
- Intermediate thought tokens improve accuracy
- Example: 100 + 200 * 3 (direct answer might fail, step-by-step works)

**YOUR KEY INSIGHT:**
- If tokens are inputs and model (12 layers) is the brain
- Predicting answer quickly = fewer tokens = fewer passes through brain
- Giving model time to think = more tokens = multiple passes through 12 layers
- **Result**: Significantly higher probability of reaching right answer
- **Why labelers prefer this**: In post-training, human labelers naturally prefer detailed reasoning responses
- This is why CoT prompting works - it's mimicking what the model learned from human preferences

**Computational Cost:**
- More complex thoughts = more tokens
- Longer context windows + more computation
- Trade-off between cost and accuracy

---

## CHECKPOINT 4: (Continuing from 1:46:56 onwards - Placeholder for remaining sections)
- [ ] Tokenization revisited: spelling struggles
- [ ] Jagged intelligence
- [ ] Supervised fine-tuning to reinforcement learning
- [ ] Reinforcement learning
- [ ] DeepSeek-R1
- [ ] AlphaGo
- [ ] RLHF
- [ ] Preview of things to come
- [ ] Keeping track of LLMs
- [ ] Where to find LLMs
- [ ] Grand summary

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
