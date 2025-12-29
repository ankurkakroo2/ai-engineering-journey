# Day 2: Andrej Karpathy Deep Dive - Running Notes

**Video**: "Deep Dive into LLMs like ChatGPT"
**Status**: In Progress (~2 hours in)
**Strategy**: Taking detailed notes during viewing, will summarize in daily-log.md at end of day

---

## CHECKPOINT 1: Understanding the Complete Pretraining Flow

### THE COMPLETE PRETRAINING FLOW

#### Step 1: Download and Preprocess the Internet
- Collect raw text data (100 billion tokens for GPT-2)
- Clean and format into sequences
- Source: Internet documents, books, web pages

#### Step 2: Tokenization - The Critical Insight
Convert text into shorter, diverse symbols (tokens)

**Why it matters:**
- "running", "runs", "ran" → shared component "run"
- Bounded vocabulary (~50k tokens instead of millions of words)
- More efficient than word-level encoding
- Each token gets a unique ID (91, 860, 287, 11579, etc.)

**How it works:**
- Start with stream of bytes (256 tokens)
- Run Byte Pair Encoding (BPE) algorithm
- Iteratively merge the most common token pair to new tokens
- Produces a vocabulary of ~50k tokens

**Resources to explore:**
- Pic Tokenizer website - visualization
- BBCroft.netllm - visualization

#### Step 3: Neural Network Training
Learn billions of parameters that predict the next token

**Input to the network:**
- Sequence of tokens: [91, 860, 287, 11579] ("The big dog ran")
- Up to 8,000 tokens per sequence

**The "Giant Mathematical Expression":**
This is where everything from Day 1 (embeddings, transformers, attention) comes together:

Parameters (weights) w: [0.2, -0.5, 1.6, 0.8, 0.0, -0.9, ...]

**Computation:**
1. Token lookup → embedding vectors (768 dimensions)
2. Layer 1 Attention:
   - Query Matrix (w_q) × embedding → what am I looking for?
   - Key Matrix (w_k) × embeddings → identity of each token
   - Value Matrix (w_v) × embeddings → what each token offers
   - Softmax: exp(scores) / sum(exp(scores)) → attention weights
   - Weighted sum → contextual representation
3. Layers 2-12: Same pattern with DIFFERENT weight matrices
4. Output layer: 768 → 50,277 vocabulary scores
5. Softmax over vocabulary → probability distribution
6. Result: 50,277 probabilities (one for each token in vocabulary)

#### Step 4: Inference (Using the Model)
Once trained, matrices are frozen. At runtime:
- User types text
- Tokenize to [91, 860, 287, 11579]
- Run through frozen network (milliseconds)
- Get probability distribution
- Sample highest probability token (3962 = "fast")
- Repeat to generate next tokens

---

## KEY PARAMETERS & WEIGHTS

### Where they come from
- Learned during training through gradient descent
- ALL learned numbers in the model

### What they are
- Embedding table: 50,277 tokens × 768 dimensions
- 12 attention layers × 4 matrices (Q/K/V/output)
- Feed-forward networks in each layer
- Output projection: 768 → 50,277
- **Total for GPT-2: 1.6 billion parameters**

### How they're used

**During training (weeks/months):**
- Start as random numbers
- Feed data through network
- Compare prediction to correct answer
- Error flows backward (backprop)
- Each weight adjusts slightly
- Repeat millions of times → optimized weights

**During inference (milliseconds):**
- All weights frozen
- Pure computation: matrix multiplies and adds
- No learning happens

---

## THE SOFTMAX BREAKTHROUGH

Converting raw attention scores to probabilities:

```
Scores: [0.3, 0.8, 1.2, 0.5]
           ↓ (Step 1: Exponentiate)
exp():  [1.35, 2.23, 3.32, 1.65]  (all positive, amplified differences)
           ↓ (Step 2: Normalize)
Softmax: [0.16, 0.26, 0.39, 0.19]  (sum to 1, interpretable as percentages)
```

**Meaning:**
- "ran" pays 39% attention to "dog"
- 26% attention to "big"
- 19% attention to itself
- 16% attention to "The"

**Why exponential?**
- Amplifies important differences
- If score for "dog" is 1.2 and "big" is 0.8:
  - Raw difference: 0.4
  - After exp: difference of 1.09 (more amplification)
  - After softmax: "dog" gets ~1.5× more attention than "big"

---

## THE FULL END-TO-END EXAMPLE

### Input: "The big dog ran"

**Tokenization:** [91, 860, 287, 11579]

**Embedding Lookup:**
- Token 91 → [0.2, -0.3, 0.5, 0.1, ...] (768 dims)
- Token 860 → [0.1, 0.8, -0.2, 0.4, ...]
- Token 287 → [-0.5, 0.1, 0.7, 0.9, ...]
- Token 11579 → [0.3, -0.1, 0.6, 0.2, ...]

**Layer 1 Attention (for "ran"):**
- Query: W_q × [0.3, -0.1, 0.6, 0.2] = [-0.15, -0.22, 0.05, 0.60]
- Dot with all Keys → scores [0.3, 0.8, 1.2, 0.5]
- Softmax → weights [0.16, 0.26, 0.39, 0.19]
- Weighted sum of Values → new contextual rep for "ran"

**Layers 2-12:**
- Same process with different matrices
- Evolving contextual representation

**Layer 12 Output:**
- Final contextual rep [0.40, 0.76, 0.25, ...]
- Output Layer: W_out × final rep → 50,277 scores
- Softmax: Probability distribution over vocabulary
  - Token 3962 ("fast"): 4% probability (highest)
  - Other tokens: lower probabilities
- **Prediction: "fast"**

---

## CONNECTING TO DAY 1 LEARNING

### From Phase 2 (LSTM Insight)
**"Important words have larger embeddings. Same weights applied uniformly. Larger embeddings naturally produce larger outputs."**

This is EXACTLY what happens in transformers:
- Larger embeddings → larger Query/Key products → higher attention scores
- Higher attention scores (after softmax) → more influence on next representation
- This all flows naturally from mathematics, no magic needed

### From Phase 3 (Transformer Clarity)
**Keys/Values computed once at start (from original embeddings)**
**Queries evolve per layer (from contextual representations)**

This caching efficiency now makes sense:
- Keys don't change because they're identity/anchors
- Queries change because they represent evolving context

---

## STILL TO WATCH

- [ ] Post-training: Fine-tuning, RLHF, instruction following
- [ ] Inference optimization: How to make predictions faster
- [ ] Scaling laws: Why bigger models are better
- [ ] Context window: Why there's a token limit

---

## KEY INSIGHTS SO FAR

✅ **No magic**: Every operation grounded in mathematics
- Embeddings → matrix multiplies → attention → softmax → repeat 12 times → output

✅ **The "Giant Mathematical Expression" is just**:
- Repeated matrix operations with different learned weights

✅ **Validation of Day 1**:
- The entire LLM pipeline validates RNN/LSTM/Transformer understanding
- Every concept from yesterday manifests here

---

## VISUALIZATION RESOURCES TO EXPLORE

- Pic Tokenizer website - See how tokenization actually works
- BBCroft.netllm - Interactive visualization of transformer internals
