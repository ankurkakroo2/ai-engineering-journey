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

---

## CHECKPOINT 2: Post-Training - From Base Model to Assistant

### KEY INSIGHT: THE REVOLUTION

**The base model is an "internet document simulator."**
**Post-training transforms it into the helpful assistant you interact with.**

### WHAT THE BASE MODEL ACTUALLY IS

After pretraining on 100 billion internet tokens, the model can:
- Complete sentences (predict next token)
- Simulate internet text patterns
- Finish code snippets, essays, recipes, etc.

**Critical limitation:** It's NOT a conversation partner. It's literally trying to predict "what comes next on the internet."

**Example of base model behavior:**

```
Human: "How do I make a cake?"

Base Model (just continuing like internet text):
"How do I make a cake?
1. Preheat oven to 350°F
2. Mix flour, sugar, butter...

Sources:
- allrecipes.com
- Wikipedia article on cakes
- Reddit thread: "Why cakes are delicious"
- Amazon affiliate link to cake pans
... [continues with more internet noise]"
```

The base model doesn't understand conversation structure. It just predicts tokens.

### WHAT POST-TRAINING CREATES: THE INSTRUCT MODEL (ChatGPT)

Post-training takes that base model and transforms it through:
1. Conversation data (not internet text)
2. Human labeling standards (helpful, harmless, honest)
3. Repeated fine-tuning on "what good responses look like"

**Same model, fundamentally different behavior:**

```
Human: "How do I make a cake?"

ChatGPT (after post-training):
"I'd be happy to help! Here's a simple chocolate cake recipe:

Ingredients:
- 2 cups flour
- 1.5 cups sugar
- ¾ cup cocoa powder
...

Instructions:
1. Preheat oven to 350°F
...

Tips: Make sure ingredients are at room temperature for best results!"
```

**The revolutionary insight:** ChatGPT isn't magic. It's the base model learning to "think and respond like a helpful human labeler."

---

## HOW POST-TRAINING WORKS

### Phase 1: Tokenization of Conversations

Just like pretraining converted text to tokens, post-training converts conversations to tokens using special formatting tokens.

**Example conversation:**
```
Human: "What is 2+2?"
Assistant: "2+2 = 4"
Human: "What if it was * instead of +?"
Assistant: "2*2 = 4, same as 2+2!"
```

**Tokenized representation (using Tiktokenizer format):**
```
_im_start_
user: What is 2+2?
_im_end_
assistant: 2+2 = 4
_im_end_
user: What if it was * instead of +?
_im_end_
assistant: 2*2 = 4, same as 2+2!
_im_end_
```

**Special tokens mark:**
- Where messages begin/end
- Who is speaking (user vs. assistant)
- Conversation structure

**Why this matters:** The model learns to recognize conversation patterns, not just continuous text.

**Note:** Different LLMs have slightly different tokenization protocols (GPT-4, Claude, Llama, etc.). It's a "wild west" of formats—no universal standard yet.

### Phase 2: Human Labeling and InstructGPT Paper (2022)

The landmark InstructGPT paper from OpenAI revealed the secret.

**How human labelers create post-training data:**

1. **Prompts are provided:**
   - "List five ideas for how to regain enthusiasm for my career"
   - "Explain quantum computing to a 10-year-old"
   - "Write a Python function to sort a list"

2. **Humans write "ideal" responses:**
   - Labelers (contractors from Upwork, Scale AI, etc.) craft responses
   - Each response is carefully written to demonstrate good assistant behavior

3. **Labeling instructions guide the work:**
   - Companies provide detailed guidelines (often hundreds of pages)
   - Core principles: Helpful, Harmless, Honest

**Example from labeling instructions:**
- "Be conversational but professional"
- "If you don't know something, say so—don't make it up"
- "Consider diverse perspectives"
- "Refuse harmful requests (hacking, illegal activity)"
- "Provide accurate information"

**Training process:**
```
Base Model (after pretraining)
        ↓
Fine-tune on conversation dataset
(human-labeled examples)
        ↓
Instruct Model (ChatGPT-style assistant)
```

The model doesn't memorize these examples. Instead, it learns the statistical pattern of "how a helpful assistant responds."

### Phase 3: The Evolution—Synthetic Data (2023-2025)

A major shift happened: Humans no longer write all the responses from scratch.

**The modern workflow:**
1. Existing LLM generates initial response
2. Human reviewers edit and refine (not write from scratch)
3. Refined responses go into training data

**Example:**
```
Initial prompt: "How do I learn Python?"

LLM-generated answer:
"Python is a programming language..."

Human labeler edits to:
"Python is an excellent language for beginners! Here's how to start:
1. Install Python...
2. Choose an IDE...
3. Work through beginner tutorials..."
```

**Datasets using this approach:**
- UltraChat (large synthetic conversation dataset)
- Various Llama training datasets
- Most modern post-training data

**Why this matters:** LLMs are now involved in creating the data that trains future LLMs. This is both powerful (scales faster) and raises questions (data quality, drift over time).

---

## THE "PSYCHOLOGY" OF CHATGPT

### What You're Actually Talking To

When you interact with ChatGPT, you're not talking to:
- A magical oracle
- A true reasoning entity
- A sentient AI

**You're talking to:** A statistical simulation of a helpful human labeler.

**How it works:**

The model has learned two things simultaneously:
1. **Pre-training knowledge:** Everything from internet text (facts, code, reasoning patterns)
2. **Post-training persona:** How a helpful assistant speaks and behaves

**Example:**
```
Prompt: "Tell me about the Colosseum in Rome"

The model knows (from pretraining):
- Location, architecture, history
- Thousands of details about Rome
- How Wikipedia articles are structured

The model learned (from post-training):
- Structure responses clearly
- Use engaging tone
- Add helpful context
- Be conversational

Result: A helpful, well-structured response that feels like talking to a knowledgeable person.
```

### Two Types of Responses

**Type 1: Responses in the training data**
```
Training data included:
Human: "Who won the 2020 US presidential election?"
Assistant: "Joe Biden won the 2020 US presidential election..."

When you ask this:
→ Model largely recites the trained response
```

**Type 2: Novel emergent responses**
```
Prompt: "What's a metaphor comparing machine learning to baking?"

The model:
1. Draws on pre-training knowledge (baking, ML concepts)
2. Applies the post-trained persona (helpful, clear, creative)
3. Generates an emergent response that "feels right"

Result: A response that wasn't explicitly trained but emerges from learned patterns
```

### The Labeler Simulation

Think of post-training as teaching the model: **"Talk like an educated, helpful, honest human labeler."**

For specialized domains:
- Code responses: Labelers are professional programmers
- Medical questions: Labelers are healthcare professionals
- Complex reasoning: Labelers are subject matter experts

The model learns to approximate this expertise level.

---

## KEY BREAKTHROUGH: THE TRANSFORMATION PIPELINE

```
PRETRAINING (Base Model)
Raw internet text → 100B tokens → Learn to predict next token
Result: "Internet document simulator"

        ↓ (Model frozen, now fine-tune)

POST-TRAINING (Instruct Model)
Conversation data + human labeling → Fine-tune on good responses
Result: "Helpful assistant simulator"

        ↓ (Further refinement)

RLHF / ALIGNMENT (Optional)
Human preference data → Train reward model → Further optimize
Result: "Better aligned assistant"

Final Product: ChatGPT
```

---

## CONCRETE EXAMPLES: BASE vs. INSTRUCT

### Example 1: Simple Question

**Base Model:**
```
"Why is the sky blue?"
[Continues like internet text]
Why is the sky blue? Scientists have long pondered this question.
According to physics, Rayleigh scattering explains...
[Random sources, ads, forum posts mixed in]
```

**Instruct Model (ChatGPT):**
```
"The sky appears blue because of a phenomenon called Rayleigh scattering.

When sunlight enters Earth's atmosphere, it collides with gas molecules.
Blue light has a shorter wavelength and scatters more easily than other colors.

Here's a simple explanation:
1. Sunlight contains all colors of the rainbow
2. Blue scatters the most
3. Our eyes see scattered blue light from all directions

This is why the sky is blue during the day and different colors at sunset!"
```

### Example 2: Harmful Request

**Base Model:**
```
"How can I hack into a computer?"
[Predicts internet-like response]
"How can I hack into a computer? Here are some common methods:
1. Social engineering...
2. Phishing...
3. SQL injection..."
[Continues with actual hacking techniques]
```

**Instruct Model (ChatGPT):**
```
"I can't provide instructions for hacking into computers without authorization.
That's illegal and harmful.

However, if you're interested in cybersecurity professionally:
- Consider legitimate security certifications (CEH, OSCP)
- Learn ethical hacking through authorized platforms
- Pursue a career in penetration testing"
```

The post-training prevented the harmful response and redirected appropriately.

---

## CONNECTING TO YOUR LEARNING JOURNEY

You now understand the complete flow:

1. **Pretraining:** Transformer learns patterns from 100B internet tokens → Base model
2. **Tokenization:** Conversations converted to token sequences with special markers
3. **Post-training:** Fine-tune on human-labeled conversation data → Instruct model
4. **Human Labeling:** Hundreds of pages of guidelines (helpful, harmless, honest)
5. **Emergence:** Model learns to "behave like" a helpful labeler on novel queries
6. **Evolution:** Modern systems use LLM-assisted labeling (synthetic data)

### The Revolutionary Insight

You're not talking to magic. You're talking to a model that learned to statistically imitate knowledgeable, helpful human labelers.

The combination of:
- **Vast pretraining knowledge** (internet understanding)
- **Post-training persona** (assistant behavior)
- **Emergent reasoning** (applying both to novel problems)

= **The assistant experience you know**

### What Makes This a Revolution (InstructGPT Paper, 2022)

**Before this paper, people thought:**
- Bigger models = automatically better at conversations
- RLHF was complex and necessary

**The InstructGPT discovery:**
- Simple supervised fine-tuning on good examples works remarkably well
- Labeling instructions (guidelines) are more powerful than realized
- Base models contain the knowledge; post-training teaches the persona
- Quality of labeled data matters more than quantity

This led directly to ChatGPT's success and enabled the entire modern LLM assistant ecosystem.

---

## SUMMARY

The post-training phase is where the magic happens—not through mysterious reasoning abilities, but through something more elegant:

**Teaching a model to think like a helpful human labeler.**

The base model knows the world (from pretraining). Post-training teaches it to share that knowledge in a helpful, honest, and safe way. It's statistical imitation at scale, and it's revolutionary because it works so well.

**Key takeaway:** Think of ChatGPT not as magical AI, but as a sophisticated statistical simulator of helpful human expertise.

---

## RESOURCES & LINKS

### Visualization Tools
- [Tiktokenizer](https://tiktokenizer.vercel.app/) - See exactly how conversations get tokenized
- [Transformer Neural Net 3D Visualizer](https://bbycroft.net/llm) - Interactive visualization of transformer internals
- Pic Tokenizer website - Tokenization visualization

### Datasets & Code
- [FineWeb (pretraining dataset)](https://huggingface.co/spaces/Hugging...) - Large-scale pretraining dataset
- [llm.c - Let's Reproduce GPT-2](https://github.com/karpathy/llm.c/dis...) - Building GPT-2 from scratch
- [HuggingFace Inference Playground](https://huggingface.co/spaces/hugging...)

### Research Papers & Models
- [InstructGPT Paper on SFT](https://arxiv.org/abs/2203.02155) - The paper that started it all
- [Llama 3 Paper from Meta](https://arxiv.org/abs/2407.21783) - Modern LLM architecture

### Applications & Tools
- [ChatGPT](https://chatgpt.com/) - The product we're studying
- [Hyperbolic - Inference of Base Model](https://app.hyperbolic.xyz/) - Run base models directly

---

## STILL TO WATCH

- [ ] Inference optimization: How to make predictions faster
- [ ] Scaling laws: Why bigger models are better
- [ ] Context window: Why there's a token limit
- [ ] RLHF (Reinforcement Learning from Human Feedback)
- [ ] Practical applications and deployment
