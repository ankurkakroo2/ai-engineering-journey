# Day 1 Reading Notes - Raw Brain Dump
**Date**: Saturday, Dec 28, 2025

⚠️ **What This Is**: Raw, unpolished notes from reading. This is a brain dump—capturing insights, "aha moments," questions, and connections as they happen. Not structured yet. Will be refined into a proper mental model later.

**Why It Looks Like This**: Incomplete sentences, jumping between topics, lots of details mixed with key insights—that's intentional. This is thinking-in-progress.

## Stack Overflow: Intuitive Introduction to Text Embeddings

### Key Learnings

**Distance Metrics - "Oh, this is so good!"**
- **Euclidean distance**: The straight line between two points
- **Cosine distance**: The circle and the angle between the two vectors
- This analogy makes it so clear how different similarity measures work!

**Examples**
- **Cat and dog example**: Great illustration of how embeddings capture semantic relationships

### THE KEY INSIGHT: Pre-Trained Models, Not Runtime Computation

**This was the hard part to comprehend:**
- ❌ WRONG: Computing embeddings at runtime
- ✅ RIGHT: Models are pre-trained. You just look up embeddings.

**How it actually works:**
1. Model is **trained once** (beforehand) on massive data
2. During training, the model discovers latent dimensions automatically
3. The model learns to assign each word specific values across those dimensions
4. At runtime: you just do word-by-word lookup → get pre-computed embedding values

**Key realization:** The dimensions are NOT pre-created by humans. The model automatically figures out what dimensions are important based on the training data and method.

---

### Latent Dimensions - The "Aha" Moment

**The Problem (Before Embeddings):**
```
Tracking 5 words: dog, cat, feline, canine, pet
Book A: (dog=2, cat=1, feline=0, canine=0, pet=1)
= 5 numbers per book

With 30,000 words in vocabulary:
= 30,000 numbers per book (slow and wasteful!)
```

**The Solution (Embeddings):**
Realize that words cluster around hidden concepts:
- "cat" and "feline" mean similar things → compress to "Catness"
- "dog" and "canine" mean similar things → compress to "Dogness"
- "pet" relates to both → split contribution

**The Magic:**
```
Book A becomes: (Catness=1.9, Dogness=1.5)
Just 2 numbers instead of 5!
```

**Why This Matters:**
- Fewer numbers = faster computation
- Captures meaning (not raw counts)
- Dimensionality reduction groups similar concepts together
- This is how embeddings work at scale

---

### Why Different Embedding Models Are Different

**All embeddings models follow the same logic, but they differ in:**

1. **Number of dimensions**
   - Fewer dimensions (e.g., 384) = Less nuance, but faster computation
   - More dimensions (e.g., 1536) = More nuance, more accurate, but slower

2. **Training data**
   - What corpus was the model trained on?
   - Different data = different discovered dimensions

3. **Training method**
   - How was the model optimized during training?

4. **Architecture and design**
   - Different neural network architectures
   - ⚠️ This is a rabbit hole - we'll skip this for now

**Example:** text-embedding-3-small vs text-embedding-3-large are both OpenAI models but have different dimension counts and accuracy/speed tradeoffs.

---

### ⚠️ RABBIT HOLE: LSA vs LDA (Avoid for now, but captured)

**Two approaches to finding latent topics:**
1. **LSA** (Latent Semantic Analysis/Allocation): Mathematical approach
2. **LDA** (Latent Dirichlet Allocation): Statistical approach

**Ultra-simple explanation:**

**LSA**: "Look at all words in all books at once and find patterns. Words that go together = same topic."
- Dump all LEGO bricks on table → Notice patterns → Organize all at once

**LDA**: "Make a guess about topics, then keep checking and fixing until it looks right."
- Grab handful randomly → Say "These are red" → Check and re-sort → Keep improving

**Bottom line**: Both find the same topics, but LSA does it mathematically all at once, LDA does it statistically with iterative guessing and checking.

**Status**: Understand the basic distinction but NOT diving deep into algorithms yet. This is a rabbit hole for Week 4+ when we build RAG systems.

---

### Why Neural Networks Are Necessary

**The Problem with Simple Normalization:**
Normalization (cosine distance) fixes the magnitude problem but misses semantic meaning.

**Example - Same word counts, opposite meanings:**
```
"The cat chased the dog"   = (cat:1, dog:1, chased:1)
"The dog chased the cat"   = (cat:1, dog:1, chased:1)

After normalization:
Both → (cat:33%, dog:33%, chased:33%)
They look identical! But they mean opposite things.
```

**What normalization solves:**
- ✅ Magnitude problem: (dog:10, cat:1) vs (dog:200, cat:1) now look similar
- ❌ Still misses: Word order, word relationships, context

**What neural networks solve:**
- ✅ Learn word relationships and context automatically
- ✅ Understand word order matters
- ✅ Know which words appear near each other matters
- ✅ Capture semantic meaning (not just counts)

**The progression:**
1. **Bag of words** → Raw counts (loses all context)
2. **Normalization** → Percentages (fixes magnitude, but still loses meaning)
3. **Neural networks** → Semantic embeddings (understands "what words mean together")

**Bottom line:** Normalization fixed the "how many words" problem. Neural networks fixed the "what do the words mean together" problem.

---

### How Neural Networks Train

**The Parent/Child Analogy:**
- **Parent (You)** = Provides the input AND the correct answer (label)
- **Baby (Network)** = Makes guesses and adjusts based on feedback
- **Training data** = Thousands of examples with correct answers

**The Training Process:**

1. **Start with Random Vectors**
   - Network begins with random, meaningless vectors for each word/input
   - Example: "cat" = [0.2, 0.8, 0.1], "dog" = [0.7, 0.3, 0.9]

2. **Feed Input/Label Pairs**
   - You provide: Review A ("about cats") + Review B ("about dogs") + Label: "DIFFERENT"
   - The network doesn't decide what to compare—you decide

3. **Make a Prediction**
   - Network measures distance between the two vectors
   - Example: Distance = 2.5

4. **Check Against Label**
   - Network compares prediction to your label using a threshold you set
   - Example: "If distance > 2.0, they're DIFFERENT"
   - Network predicted 2.5, label says DIFFERENT, 2.5 > 2.0 ✅ Correct

5. **Adjust and Repeat**
   - If wrong, adjust vectors slightly
   - Process thousands of different pairs
   - Vectors gradually improve

6. **When to Stop**
   - After processing all training data multiple times (epochs)
   - When accuracy is acceptable
   - When you manually stop it

7. **The Network Learns**
   - Vectors stabilize over iterations
   - Similar items end up close together
   - Different items end up far apart
   - **Meaningful embeddings created!**

---

### Word2vec: Creating Word Embeddings

**What It Does:**
Creates embeddings for individual words. Each word gets a small vector (~300 numbers) that captures semantic meaning and relationships.

**Training Process:**

1. **Raw Text (No Labels!)**
   - Start with large unlabeled corpus (Wikipedia, books, articles)
   - Text itself provides the training signal—no manual labeling

2. **Sliding Context Window**
   - Slide fixed-size window through text. Each window = one training example.
   - Example (window size 3, "the cat sat on the mat"):
     ```
     Position 1: (the, cat, sat)
     Position 2: (cat, sat, on)
     Position 3: (sat, on, the)
     ```
   - **Signal**: Words in same window are probably related

3. **Refine Individual Word Embeddings**
   - Each word has ONE embedding vector (shared across all contexts)
   - When "cat" appears in different windows, its embedding gets pulled in different directions
   - Refined by all contexts it appears in

4. **Multiple Passes (Epochs)**
   - Repeat sliding window across entire text multiple times
   - Pass 1: Rough adjustments from random vectors
   - Pass 2: Finer adjustments based on pass 1 results
   - Pass 3+: Even finer refinements
   - **Why?** Embeddings change, so repeating improves quality

5. **Stop at Convergence**
   - After ~5-10 epochs, embeddings stabilize
   - Done training

**The Magic: Vector Arithmetic**

Embeddings capture semantic relationships as mathematical dimensions.

```
king - man + woman ≈ queen
```

**Why it works**: Network learned gender as a measurable dimension. Remove maleness, add femaleness, get queen.

**Key Advantages:**
- ✅ No labeled data needed
- ✅ Uses raw text as-is
- ✅ Learns rich semantic relationships
- ✅ Simple context-based learning creates meaningful embeddings

---

### 🐰 RABBIT HOLE: Recurrent Neural Networks (RNNs)

**What They Do:**
Process text token-by-token (word-by-word) sequentially, accumulating context as they move through the sequence. Each new word's meaning depends on all previous words.

**How They Work (The Forward Process):**

Example: "The dog was cute" → Predict sentiment

Pre-requisite: Word embeddings are pre-stored
- "The" → [0.1, 0.5, 0.2]
- "dog" → [0.3, 0.7, 0.1]
- "was" → [0.4, 0.2, 0.8]
- "cute" → [0.6, 0.9, 0.3]

1. **Token 1 ("The")**: Context = [0.1, 0.5, 0.2]
2. **Token 2 ("dog")**: Combine previous context + new embedding using learned weights
3. **Token 3 ("was")**: Combine new context + "was" embedding
4. **Token 4 ("cute")**: Combine context + "cute" embedding
5. **Predict**: Pass final context through classifier → POSITIVE (85% confidence)

**What Are Weights?**

- **Embeddings**: Fixed vectors (pre-learned via Word2vec) - don't change during RNN training
- **Weights**: Learned multipliers (one per dimension, NOT per word) that define how to combine embeddings
  - Start as random numbers
  - Improve through training by example
  - Final weights like [0.8, 0.8, 0.7, ...] (300 weights for 300 dimensions)

**Critical Insight**: "Each dimension has the same weight for ALL words."
- Dimension 1: Weight 0.8 applies to "dog," "cat," "the," everything
- Dimension 2: Weight 0.8 applies to all words
- etc.

**The Vanishing Gradient Problem (CRITICAL):**

RNNs fail at long-distance relationships because error signals shrink as they travel backward.

**Concrete Example:**
```
Sentence: "The dog that I adopted from the pound five years ago won the pet competition."
Problem: Network must connect "dog" (position 2) to "won" (position 15) — 13 steps apart
```

**How Error Travels Backward:**
```
Position 15 (where error happens): Error = 1.0
Position 14: Error × 0.8 = 0.8
Position 13: Error × 0.8 = 0.64
Position 12: Error × 0.8 = 0.512
...
Position 2 (the important "dog"): Error × 0.8^13 = 0.055 (VANISHED!)
```

**Why This Happens:**
Weights are multiplied together going backward. With weights < 1:
- 0.8^5 = 0.33 (shrinks to 1/3)
- 0.8^10 = 0.107 (shrinks to 1/10)
- 0.8^13 = 0.055 (shrinks to nearly nothing)

**The Proportional Blame Insight:**
"Weights represent how much that word mattered, so you multiply the error by them to assign proportional blame"
- Weight 0.8 → contributed 80% → gets 80% of the blame
- Weight 0.2 → contributed 20% → gets 20% of the blame

This is logically correct, but it makes early words stop learning.

**Why Weights Can't Be Larger:**
- If weights > 1: Error signal explodes (exploding gradients) → network unstable
- If weights < 1: Error signal vanishes (vanishing gradients) → can't learn long-distance
- Catch-22: There's no perfect solution with this architecture

**Bottom Line:**
RNNs can't learn that "dog" at position 2 is important when the error signal reaches it as 0.055 instead of 1.0. This is why **LSTMs and Transformers were invented** — they bypass the multiplication for certain information, creating direct pathways so error signals don't vanish.

**Status**: Deep rabbit hole, but now understand why RNNs are limited and why better architectures exist.

---

### 🐰 RABBIT HOLE: Long Short-Term Memory (LSTMs)

**What Solves RNN's Problem:**
Instead of information traveling through pure multiplication (which vanishes), LSTMs maintain a separate **long-term memory vector** that travels through the entire sequence, preserving important information.

**The Memory Cell: A Running Vector**

Memory is a vector with same dimensions as embeddings (~300 dimensions). It accumulates information across tokens:

```
Token 1 ("The"): Memory = [0.092, 0.46, 0.164] (small—unimportant)
Token 2 ("dog"): Memory = [0.3496, 1.012, 0.2132] (grew—important added)
Token 3 ("that"): Memory = [0.481, 1.366, 0.3286] (grows, but controlled)
...
Token 15 ("won"): Memory still holds [0.3496, 1.012, 0.2132] from "dog"! ✅
```

**The Three Gates Managing Memory:**

1. **Input Gate**: "Should I add this word to memory?"
   - Important words (larger embeddings) → produce larger amounts → more gets stored
   - Example: "dog" [0.3, 0.7, 0.1] × weights [0.92, 0.92, 0.82] = [0.276, 0.644, 0.082] (large)
   - Example: "the" [0.1, 0.5, 0.2] × weights [0.92, 0.92, 0.82] = [0.092, 0.46, 0.164] (smaller)

2. **Forget Gate**: "What percentage of old memory should I keep?"
   - Multiplies old memory by weights (typically 0.7-0.9)
   - Gently degrades old info while preserving important patterns
   - Example: Memory [0.3496, 1.012, 0.2132] × [0.85, 0.85, 0.85] = [0.2972, 0.8602, 0.1812]
   - Notice: "dog" info mostly preserved, slightly reduced

3. **Output Gate**: "What part of memory should I expose for prediction?"
   - Multiplies memory by weights to expose relevant information
   - Applied at EVERY token, not just the end
   - Early tokens see limited memory (not much accumulated yet)
   - Late tokens see rich memory (accumulated everything)

**Complete Memory Cycle (Each Token):**
1. Candidate transformation: embedding → candidate gate → candidate vector
2. Input gating: How much to add? (candidate × input weights)
3. Forget gating: How much of old memory to keep? (old memory × forget weights)
4. Update memory: kept memory + amount to add = new memory
5. Output gating: What to expose? (memory × output weights)
6. Pass to predictor

**Why This Preserves Long-Distance Information:**

Example: "The dog...won" (13 steps apart)

- **Token 2 ("dog")**: Information stored in memory [0.3496, 1.012, 0.2132]
- **Tokens 3-14**: Forget gates keep 0.80-0.85 of old memory, filler words add little
  - Memory stays around [0.3-0.35, 0.95-1.05, 0.19-0.22]
- **Token 15 ("won")**: Memory STILL has [0.3496, 1.012, 0.2132]!
  - Predictor can access "dog" as subject

**Compare to RNN:**
- RNN: "dog" info × 0.8^13 = 0.055 (nearly vanished) ❌
- LSTM: "dog" info preserved in memory with gates [0.80-0.85] (strong signal) ✅

**Why It Works (Not Magic):**
- **Input gate filters**: Only important words (larger embeddings) produce large stored amounts
- **Forget gate filters**: Old memory gradually degraded but important patterns preserved
- **Output gate filters**: Only relevant info exposed for current predictions
- All three work together to manage what gets in, what stays, what gets used

**Memory Limitations (Why Transformers Are Better):**
- **Single vector bottleneck**: ALL long-distance info must fit in one vector [?, ?, ?]
- **Crowding**: In very long docs (1000+ tokens), important info can get overwritten
- **No direct paths**: Token 2 only connects to token 15 through memory cell. If memory degrades, connection weakens

Transformers solve this by maintaining explicit relationships between ALL token pairs, creating multiple direct paths instead of routing everything through one memory cell.

**Status**: Understand how LSTMs overcome RNN limitations. Memory is a managed vector, not magic. Understand why Transformers are the next evolution.

---

### 🐰 RABBIT HOLE: Transformers - Complete Understanding

**The Core Problem Transformers Solve:**
- LSTMs process tokens sequentially (slow) + route all information through single memory cell (bottleneck)
- Transformers solve both: all tokens process simultaneously + each token directly queries any other token

**Key-Value Store Analogy:**
- Vendor "dog" has: storefront sign (key) = "noun, concrete, subject" + inventory (value) = semantic info
- Customer "won" has: shopping list (query) = "looking for subjects that perform actions"
- "won" compares query to all keys, finds "dog" matches (0.95 similarity), takes 0.95 of "dog"'s value
- Final purchase = weighted blend of everything, weighted by query-key similarity

**The Four Things Per Token (Using "dog" as Example):**

1. **Original Embedding** (Fixed, pre-learned)
   - [0.3, 0.7, 0.1] (768 dimensions)
   - Learned during Word2vec training, never changes
   - Fundamental meaning of the word

2. **Key Vector** (Pre-computed once, reused across all 12 layers)
   - [0.3, 0.7, 0.1] × KeyMatrix = [0.29, 0.69, 0.08]
   - Computed at runtime start, stored, never recalculated
   - Vendor's storefront sign—how other tokens find you
   - Stable because based on original embedding

3. **Value Vector** (Pre-computed once, reused across all 12 layers)
   - [0.3, 0.7, 0.1] × ValueMatrix = [0.28, 0.68, 0.09]
   - Computed at runtime start, stored, never recalculated
   - Vendor's inventory—what you offer when found
   - Stable because based on original embedding

4. **Query Vector** (Computed fresh each layer, changes)
   - Layer 1: [0.3, 0.7, 0.1] × QueryMatrix(L1) = [0.29, 0.68, 0.09]
   - Layer 2: [0.35, 0.72, 0.15] × QueryMatrix(L2) = [0.30, 0.69, 0.10] (DIFFERENT!)
   - Layer 3: [0.37, 0.73, 0.18] × QueryMatrix(L3) = [0.31, 0.69, 0.11] (DIFFERENT AGAIN!)
   - Shopping list—what you're looking for right now
   - Changes because computed from evolving contextual understanding

5. **Contextual Representation** (Updated each layer, flows through)
   - Layer 1 input: [0.3, 0.7, 0.1] (original)
   - Layer 1 output: [0.35, 0.72, 0.15] (updated by attention)
   - Layer 2 input: [0.35, 0.72, 0.15] (flows to next layer)
   - Layer 2 output: [0.37, 0.73, 0.18] (updated again)
   - ... continues through all 12 layers ...
   - Layer 12 output: [0.40, 0.76, 0.25] (final enriched representation)

**Why This Design Works:**
- Key & value stay same: represent what you fundamentally are (stable anchors)
- Query changes: represents what you currently need (evolves with context)
- Pre-computed & reused key/value: efficiency (compute once, use in all 12 layers)
- Fresh queries each layer: flexibility (search differently as context accumulates)

**How Attention Works:**
```
Attention = Σ(similarity_score × value_vector)

High query-key similarity → High weight → That token's value contributes more
Low query-key similarity → Low weight → That token's value contributes less
```
Answer: "How much should I listen to each token?" based on query-key match

**Why Layers Matter:**
- Layer 1: Basic patterns (articles with nouns, adjectives with nouns)
- Layers 2-5: Medium patterns (grammatical relationships, word pairs)
- Layers 6-9: Complex patterns (semantic roles, word sense in context)
- Layers 10-12: Subtle nuances (pragmatic meaning, contextual interpretation)

**Parallelization:**
All tokens process simultaneously on GPU. "The" computes attention while "dog" computes attention while "ran" computes—all at same time. Makes transformers ~100x faster than LSTMs.

**Parameters (Learned Numbers):**
- Word embeddings: 50K words × 768 dims = 38.4M parameters
- Query matrices: 12 layers × 768×768 = ~7M parameters
- Key/Value matrices: 768×768 = ~0.6M each (same across all layers)
- Others: ~billions more
- **Total**: ~1-2 billion parameters
- **Key insight**: Embedding dimension has exponential impact (doubling dimension quadruples matrix size)

**Scaling:**
- 1M params: 70% accuracy
- 100M params: 75% accuracy
- 1B params: 78% accuracy
- 10B params: 82% accuracy
- 100B params: 85% accuracy
- **No ceiling**: Self-supervised task (predict hidden word) is inexhaustible

**End-to-End Example: "The big dog ran" → Predict next word**

*Step 1: Look up embeddings*
- "The" → [0.1, 0.5, 0.2]
- "big" → [0.2, 0.6, 0.25]
- "dog" → [0.3, 0.7, 0.1]
- "ran" → [0.5, 0.3, 0.6]

*Step 2: Compute key & value ONCE*
- "dog" Key = [0.3, 0.7, 0.1] × KeyMatrix = [0.29, 0.69, 0.08] (stored)
- "dog" Value = [0.3, 0.7, 0.1] × ValueMatrix = [0.28, 0.68, 0.09] (stored)
- (same for all tokens)

*Layer 1:*
- Query: [0.3, 0.7, 0.1] × QueryMatrix(L1) = [0.29, 0.68, 0.09]
- Attention similarities:
  - "The" key: 0.18 similarity
  - "big" key: 0.78 similarity ← good match
  - "dog" key: 0.95 similarity ← excellent match
  - "ran" key: 0.35 similarity
- Weighted values: 0.18×[The value] + 0.78×[big value] + 0.95×[dog value] + 0.35×[ran value]
  - Result: [0.615, 1.303, 0.535]
- Layer 1 output: [0.3, 0.7, 0.1] + [0.615, 1.303, 0.535] = [0.35, 0.72, 0.15]
- "dog" learned it's described as "big" and related to "ran"

*Layer 2:*
- Query (different input!): [0.35, 0.72, 0.15] × QueryMatrix(L2) = [0.30, 0.69, 0.10]
- New similarities (different from Layer 1):
  - "ran" key: 0.80 similarity ← much higher now!
- "dog" recognizes "ran" is even more relevant
- Layer 2 output: [0.37, 0.73, 0.18]

*Layers 3-12: Same pattern - each layer refines understanding*

*Final representation after Layer 12:*
- "dog" final: [0.40, 0.76, 0.25]
- Contains: "I'm a noun, described by 'big', subject of 'ran', central to this sentence"
- "ran" final: [0.48, 0.71, 0.45]
- Contains: "I'm a verb, subject is 'dog', past tense, what follows?"

*Prediction:*
- [0.48, 0.71, 0.45] → prediction layer → probabilities
- "fast": 45% ← selected
- "away": 25%
- "to": 15%
- others: 15%

*Output: "The big dog ran fast"*

**Why This Works:**
12 layers progressively refined understanding. All tokens learned simultaneously. Key/value provided stability (knowing who you are). Query provided flexibility (searching differently as context accumulated). Attention weighted by relevance. Result: model understands complete semantic structure.

**Status**: Understand transformers end-to-end, from embedding lookup to final prediction. Understand parallelization, attention, layers, and scaling.

---

*Note: These are raw notes. Will polish and create structured mental model later.*
