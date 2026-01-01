# Andrej Karpathy's "Deep Dive into LLMs like ChatGPT" - Complete Running Notes

**Video**: "Deep Dive into LLMs like ChatGPT"
**Duration**: 3+ hours (complete deep dive, not condensed intro)
**Strategy**: Detailed checkpoint notes during viewing, capturing all learning and key revelations
**Status**: In Progress (Checkpoints 1-3 complete, Checkpoints 4+ coming as you watch)

---

## Foundation Coming Into This Deep Dive

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

## SUMMARY (Checkpoints 1-2)

The post-training phase is where the magic happens—not through mysterious reasoning abilities, but through something more elegant:

**Teaching a model to think like a helpful human labeler.**

The base model knows the world (from pretraining). Post-training teaches it to share that knowledge in a helpful, honest, and safe way. It's statistical imitation at scale, and it's revolutionary because it works so well.

**Key takeaway:** Think of ChatGPT not as magical AI, but as a sophisticated statistical simulator of helpful human expertise.

---

## CHECKPOINT 3: Hallucinations, Tool Use, Knowledge & Working Memory

### Hallucinations: Models Always Predict Something

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

### Tool Use: Special Tokens as API Calls

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

### Knowledge vs. Working Memory: Two Types of Information

**Knowledge: What Lives in Parameters**

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

**Working Memory: What's in the Current Prompt**

This is the information provided in the current conversation—the full context window available to the model right now.

**Characteristics:**
- Specific and current: "In this document, cats are defined as..."
- Reliable: Model can reference exact text from prompt
- Immediate: Available for this conversation, doesn't decay
- Examples: Uploaded documents, previous messages, system instructions

**Why Quality Improves with Working Memory:**
If knowledge says "cats meow" but your prompt says "In this company, 'cat' is our mascot named Mr. Whiskers who barks," the model trusts the prompt.

Model learns in post-training: "Information in working memory is more specific and reliable than general knowledge."

**In-Context Learning Relies on Working Memory:**

This is the ability to learn from examples in the current prompt:

```
Example from prompt:
"Here's how to translate English to Pig Latin:
- hello → ello-hay
- computer → omputer-cay

Now translate: programmer"
```

Model learns the pattern from these 2 examples in the prompt (working memory), not from its pre-trained knowledge. It didn't see "Pig Latin" in training data with confidence, but the examples in working memory teach it the pattern.

**The Tradeoff:**

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

### Knowledge of Self: Pure Imitation

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

### Models Need Tokens to Think: The Chain-of-Thought Revolution

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

**Chain-of-Thought Prompting:**

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

## CHECKPOINT 4: Reinforcement Learning & Advanced Reasoning

### The Textbook Analogy: Three Types of Information in Learning

**Exposition (Pre-training equivalent)**
- Most text in any textbook is exposition—background knowledge, context, foundational information
- When reading through exposition, it's roughly equivalent to pre-training
- This is where we build a knowledge base and get a sense of the topic
- The model is essentially "reading" and internalizing information into its parameters

**Worked Solutions (Supervised Fine-Tuning equivalent)**
- Problems with their worked solutions provided by human experts (the textbook author)
- The solution is equivalent to having the "ideal response" for an assistant
- The expert shows us how to solve the problem in its full form
- As we read the solution, we are training on expert data—later we can try to imitate the expert
- This corresponds to having the SFT (Supervised Fine-Tuning) model

**Practice Problems (Reinforcement Learning equivalent)**
- Practice problems at the end of each chapter—critical for learning
- What you get: the problem description and the final answer (in the answer key)
- What you don't get: the solution itself
- You must practice the solution, trying out many different approaches
- You're discovering how to solve problems yourself
- You rely on: (1) background information from pre-training, (2) some imitation of human experts

**Key Insight**: The three stages of LLM training directly parallel how humans learn from textbooks—exposition → worked examples → practice problems.

---

### The Problem with Human-Annotated Solutions

**The Core Challenge**

Given a math problem: "Emily has three apples and two oranges. Each orange is $2. The total cost of all fruit is $13. What is the cost of each apple?"

Multiple candidate solutions can all reach the correct answer (3):
- Some solutions set up systems of equations
- Some talk through it in English
- Some skip right through to the solution

**Two Purposes of a Solution**
1. **Primary purpose**: Reach the right answer (correctness)
2. **Secondary purpose**: Present it nicely for the human (presentation)

**The Fundamental Unknown**

As a human labeler, you don't know which solution format is optimal for the LLM:
- For each token, the model can only spend a finite amount of compute—not very large
- The model can't make too big of a leap in any single token

**The Mental Arithmetic Problem**

Example: "30 - 4 ÷ 3 =" in a single token requires too much computation. This might incentivize the model to skip through calculations too quickly → mistakes. Maybe it's better to spread out the work, set up equations, or talk through it.

**The Cognition Gap**

- What is easy or hard for humans is different from what is easy or hard for the LLM
- Some token sequences trivial for us might be "too much of a leap" for the LLM
- Conversely, many tokens we create might be trivial to the LLM—wasting tokens
- Human knowledge ≠ LLM knowledge
- LLM has PhD-level knowledge in many domains—potentially more than the labeler
- We might not utilize that knowledge, or inject knowledge the LLM doesn't have

**The Conclusion**

- We are not in a good position to create ideal token sequences for the LLM
- Imitation is useful to initialize the system
- But we need the LLM to **discover** the token sequences that work for it
- The model needs to find what token sequence reliably gets to the answer
- This discovery happens through reinforcement learning and trial and error

**Key Takeaway**: Human annotators cannot know the optimal solution format for LLMs because cognition differs—RL lets models discover their own optimal thinking patterns.

---

### Reinforcement Learning: The Core Mechanism

**The Basic Process**

1. Take a prompt, run the model
2. Model generates a solution
3. Inspect the solution against the known correct answer
4. Repeat many times—each attempt produces a slightly different generation (models are stochastic)
5. At every token, there's a probability distribution; sampling from it leads to different paths

**Scale of Practice**

In practice, you might sample thousands or even millions of independent solutions for a single prompt. Some solutions will be correct, some will not.

**Goal**: Encourage the solutions that lead to correct answers.

**How Learning Happens**

- Whatever token sequences happened in wrong solutions—something went wrong; not a good path
- Whatever token sequences in correct solutions—things went well; do more of this
- To encourage this behavior: **train on these sequences**
- **Crucially**: these training sequences come from the model itself, not from human annotators
- The model is practicing—it tried solutions, some worked, now it trains on what worked

**The Student Analogy**

Like a student looking at their own solutions: "This one worked well, so this is how I should solve these kinds of problems."
- Take the single best solution (maybe shortest, or has other nice properties)
- Train on it → model becomes slightly more likely to take this path in similar future settings

**Scale of Optimization**

- Tens of thousands of diverse prompts across math, physics, coding problems
- Thousands of solutions per prompt
- All happening at the same time
- As the process iterates, the model discovers for itself what kinds of token sequences lead to correct answers
- **Not coming from a human annotator**

**Key Takeaway**: RL is "guess and check" at massive scale—the model generates many solutions, keeps what works, and trains on its own successful attempts.

---

### DeepSeek-R1: The Breakthrough Paper

**Why This Paper Matters**

- Pre-training and SFT have been around for years—standard across all LLM providers
- RL training is a lot more early/nascent and not yet standardized
- The high-level idea is simple (trial and error), but there are tons of details:
  - How to pick the best solutions
  - How much to train on them
  - What is the prompt distribution
  - How to set up training so it actually works
- Many companies (OpenAI, etc.) experimented internally but didn't talk publicly
- DeepSeek-R1 paper changed this—talked very publicly about RL fine-tuning for LLMs

**The Paper's Impact**

- Reinvigorated public interest in RL for LLMs
- Gave nitty-gritty details needed to reproduce results
- Showed how incredibly important RL is for reasoning capabilities

**Quantitative Results**

Figure showing improvement in solving mathematical problems (AIME accuracy):
- Initially models don't do well
- As you update with thousands of steps, accuracy climbs
- Models are discovering how to solve math problems through trial and error

**Qualitative Results: The Emergence of Thinking**

Even more incredible than accuracy improvements is **how** models achieve these results:
- Average length per response goes up—models use more tokens
- Models learn to create very, very long solutions

**What Emerges in the Solutions**

- "Wait, wait, wait. That's an aha moment. Let's reevaluate this step by step..."
- Model is re-evaluating steps
- It learns that trying ideas from different perspectives, retracing, reframing, backtracking → better accuracy
- These are things humans do in problem solving—but for your internal mental process, not what you write down

**The Profound Insight**

- No human can hard-code this in an ideal assistant response
- This can ONLY be discovered through reinforcement learning
- You wouldn't know what to put here—it just turns out to work for the model
- Models learn "chains of thought"—an emergent property of the optimization
- This is what bloats response lengths AND increases problem-solving accuracy
- **The model is discovering ways to think**—cognitive strategies for manipulating problems

**Key Takeaway**: DeepSeek-R1 showed that reasoning and "thinking out loud" emerges naturally from RL—no human programs this; it's discovered through optimization.

---

### Reasoning/Thinking Models in Practice

**SFT Model Response (e.g., ChatGPT 4.0)**

Standard response mimicking an expert solution:
- Clean, direct
- Presents nicely for humans

**RL Model Response (e.g., DeepSeek-R1)**

```
"Okay, let me try to figure this out..."
"So Emily buys three apples and two oranges..."
"I need to find out..."
"Wait a second. Let me check my math again to be sure."
[Tries from a slightly different perspective]
"Yep, all that checks out. I think that's the answer."
"Let me see if there's another way to approach the problem. Maybe setting up an equation..."
"Yep, same answer. So definitely each apple is $3."
"All right, confident that that's correct."
[Then writes up the nice solution for the human]
```

**The Two Phases**

1. **Thinking process** (from RL): trying different ways, verifying, aha moments
2. **Presentation** (for humans): writing it out nicely, boxing the answer

**Accessing Thinking Models**

**DeepSeek-R1**:
- Direct: chat.deepseek.com (enable "Deep Think" button)
- Via API/hosting: Together AI, other inference providers
- Note: Some people are cautious about putting sensitive data into Chinese company servers
- DeepSeek-R1 is open weights (MIT licensed)—anyone can download and use

**OpenAI's Thinking Models**:
- Models like o1, o3-mini, o3-mini-high use "advanced reasoning"
- Trained with RL using techniques similar to DeepSeek-R1
- GPT-4o models: mostly SFT models—don't do this thinking
- OpenAI hides exact chains of thought, shows only summaries (worried about "distillation risk")

**Google's Thinking Models**:
- AI Studio: Gemini 2.0 Flash Thinking Experimental
- Also a thinking model that produces similar results

**Anthropic**:
- Currently does not offer a thinking model (as of early 2025)

**When to Use Which**

- Thinking models: Advanced reasoning, math, code, difficult problems
- Standard models (GPT-4o): Knowledge-based questions, simpler tasks
- As Karpathy noted: "Empirically about 80-90% of my use is just GPT-4o. When I come across a very difficult problem, I reach for thinking models—but then I have to wait longer because they are thinking."

**Key Takeaway**: Thinking models (o1, o3, DeepSeek-R1) use RL-trained reasoning; standard models (GPT-4o) are mostly SFT—choose based on task complexity.

---

### AlphaGo: The Precedent for RL Power

**The Discovery Isn't New**

RL being extremely powerful for learning was already demonstrated in the game of Go. DeepMind's AlphaGo—watch the documentary.

**The Key Insight from AlphaGo Paper**

Comparison: Supervised Learning vs. Reinforcement Learning

ELO rating in game of Go:
- **Supervised Learning (imitating human experts)**: Gets better, but tops out—never quite reaches top human players
- **Reinforcement Learning**: Significantly more powerful—overcomes even top players like Lee Sedol

**If you're just imitating human players, you can't fundamentally go beyond human performance.**

In RL for Go:
- The system plays moves that empirically and statistically lead to winning
- AlphaGo plays against itself, using RL to create rollouts
- Games that lead to wins are reinforced

**Move 37: The Iconic Example**

- AlphaGo played a move no human expert would play
- Probability of this move being played by a human: ~1 in 10,000
- In retrospect: a brilliant move
- AlphaGo discovered a strategy unknown to humans through RL
- Experts were shocked: "That's a very surprising move..."

**Implications for Language Models**

In principle, we can see the same thing in LLMs for open-domain problem solving:
- What does it mean to solve problems in ways even humans wouldn't think of?
- Maybe discovering analogies humans couldn't create
- Maybe new thinking strategies
- Maybe even a wholly new language better suited for thinking (not constrained to English)
- The behavior is less defined—open to do whatever works

**The Requirement**

This only works if we have a very large, diverse set of problems for refinement:
- Frontier research: creating prompt distributions that are large and diverse
- Like writing textbooks with practice problems for all domains of knowledge
- If we have enough practice problems, models can RL on them and create these improvement curves—but for open thinking, not closed domains like Go

**Key Takeaway**: AlphaGo proved RL can exceed human performance by discovering strategies humans never conceived—we're now seeing early hints of this in LLMs.

---

### RLHF: Reinforcement Learning from Human Feedback

**The Problem: Unverifiable Domains**

- **Verifiable domains**: concrete answer to score against (math: answer is 3)
- **Unverifiable domains**: creative writing, summarization, poem writing, joke writing
- Example: "Write a joke about pelicans"—how do you score different jokes?

**The Naïve Approach (Doesn't Scale)**

- Have humans inspect and score all generated outputs
- Problem: RL requires many thousands of updates, thousands of prompts per update, hundreds/thousands of generations per prompt
- Total evaluations needed: ~1 billion (way too much human time)

**The RLHF Trick: Indirection**

1. Train a separate neural network called a **reward model**
2. This reward model imitates human scores
3. Once you have a simulator, you can query it as many times as you want
4. Do RL against the simulator instead of real humans

**Training the Reward Model**

1. Prompt: "Write a joke about pelicans"
2. Generate 5 different jokes (rollouts)
3. Ask a human to **order** the jokes from best to worst (ordering is easier than precise scoring)
4. Human ranks: #1 (best), #2, #3, #4, #5 (worst)
5. Reward model (separate transformer) takes prompt + joke → outputs a score (0 to 1)
6. Compare reward model scores with human ordering
7. Update reward model to be consistent with human preferences
8. Mathematical loss function calculates correspondence and updates the model

**Example of Reward Model Update**

- Human says joke X is #1, reward model gives it 0.8 → should be even higher → increases
- Human says joke Y is #2, reward model gives it 0.1 → massive disagreement → increases significantly
- Human says joke Z is #5 (worst), reward model gives it 0.7 → should be lower → decreases

**Upsides of RLHF**

1. Allows running RL in arbitrary domains including unverifiable ones
2. Empirically results in better models
3. Easier task for human labelers (ordering vs. creative writing)
4. Sidesteps the "discriminator-generator gap"—easier to judge than to create

**Downsides of RLHF**

1. RL against a lossy simulation of humans, not actual human judgment
2. The simulation might not perfectly reflect actual human opinion in all cases
3. **The Devious Problem: Gaming the Reward Model**
   - RL is extremely good at finding ways to game/exploit the simulation
   - Reward models are massive neural nets with billions of parameters
   - There are inputs that get inexplicably high scores but are nonsensical

**The Gaming Problem in Detail**

- Run RLHF for many updates (e.g., 1000)
- First few hundred steps: jokes improve
- Then: quality dramatically falls off, get nonsensical results
- Example: Top joke becomes random characters like "🎣🐦💰🌊"
- Reward model gives it score 1.0 (loves it)—but it's obviously not a good joke
- These are "adversarial examples"—inputs that go into nooks and crannies giving nonsense results
- You can add these to the training set with low scores, but there are always more
- Infinite nonsensical adversarial examples hiding in any reward model

**RLHF Is Not "Magical" RL**

- "RLHF is RL, but not RL in the magical sense"
- Not RL you can run indefinitely
- **Verifiable domains** (math, code): can't game easily—either correct answer or not
  - Can run for tens of thousands, hundreds of thousands of steps—discover really crazy strategies
- **RLHF**: think of it as "a little fine-tune that slightly improves your model"
- GPT-4o has gone through RLHF—it works—but it lacks the magic of pure RL

**Key Takeaway**: RLHF enables RL in creative/subjective domains by simulating human preferences, but reward models can be gamed—it's useful but not magical like verifiable-domain RL.

---

### Summary: The Three Stages of LLM Training

**Pre-training (Reading Exposition)**
- Equivalent to reading all explicit story material
- Look at all textbooks simultaneously, read all exposition
- Build a knowledge base

**Supervised Fine-Tuning (Studying Worked Solutions)**
- Looking at all fixed solutions from human experts
- Across all work solutions in all textbooks
- Get an SFT model that imitates experts—but blindly, doing statistical best-guess mimicry

**Reinforcement Learning (Practice Problems)**
- Do all practice problems in the RL stage across all textbooks
- Only practice problems → get the RL model

**The Parallel to Human Learning**

- Pre-training: basic knowledge acquisition (reading exposition)
- SFT: looking at lots of worked examples, imitating experts
- RL: practice problems where we discover for ourselves

**Key Takeaway**: LLM training mirrors human learning: read → study examples → practice. The key difference is LLMs do each stage in bulk across all domains simultaneously.

---

### Future Directions & What's Coming

**Multimodality**

- Models will rapidly become multimodal: text + audio (hear/speak) + images (see/paint)
- Done natively inside the language model
- Enable natural conversations
- Technically: tokenize audio (slices of spectrogram) and images (patches)—add more tokens, train just like text
- Not a fundamental change—just more token types

**Agents & Long-Running Tasks**

- Currently: handing individual tasks to models on a "silver platter"
- Models not yet capable of stringing together tasks to perform jobs coherently over long periods
- Coming: "agents" that perform tasks over time, report progress, require supervision
- Tasks that take minutes or hours, not just seconds
- "Human-to-agent ratios" will become a thing (like human-to-robot ratios in factories)

**Pervasive & Invisible Integration**

- AI integrated everywhere into tools
- Computer-using agents: taking keyboard/mouse actions on your behalf
- Example: OpenAI's "Operator"

**Test-Time Training (Speculative)**

- Current paradigm: train parameters → fix them → deploy for inference
- Only "learning" at test time is through context window
- Humans actually learn and update (especially during sleep)—no equivalent in current models
- Context window is finite and precious
- For very long multimodal tasks, making context windows longer alone won't scale
- New ideas needed for truly long-running tasks

**Key Takeaway**: Expect multimodality, long-running agents, invisible integration, and potentially new paradigms like test-time training.

---

### Staying Up to Date: Resources

**LM Arena (lmarena.ai)**

- LLM leaderboard ranking top models
- Based on human comparisons (blind judging which model gives better answers)
- Shows organizations, model names, licenses
- Notable: DeepSeek at #3 with MIT license (open weights)—unprecedented for a model this strong
- Caveat: Leaderboard may be somewhat gamed recently—use as first pass, try models yourself

**AI Newsletter (aon.news)**

- Very comprehensive newsletter produced almost every other day
- Some human-written/curated, much constructed with LLM help
- Summaries at the top have human oversight
- Won't miss anything major if you follow this

**X/Twitter**

- Lot of AI happens on X
- Follow people you like and trust

**Key Takeaway**: LM Arena for rankings (but verify yourself), AI Newsletter for comprehensive coverage, X/Twitter for real-time updates.

---

### Where to Find and Use Models

**Local Inference: LM Studio**

- Download and run models locally on your computer
- Search for models, load them, chat with them
- All happens locally—nothing sent anywhere
- Geared towards professionals, has UI/UX issues
- Watch YouTube tutorials to get started

**Cloud Inference: Hugging Face Inference Playground**

- Easily call different kinds of models
- Good for experimentation

**Together AI**

- Hosts full state-of-the-art models
- Select DeepSeek-R1, other frontier models
- Default settings usually OK

**Direct Providers**

- ChatGPT (OpenAI)
- chat.deepseek.com (DeepSeek)
- AI Studio (Google)
- Claude (Anthropic)

**Key Takeaway**: Multiple options for accessing models—local (LM Studio), cloud playgrounds (Hugging Face, Together AI), or direct providers.

---

### Theory of Mind for LLMs: What You're Actually Talking To

**The SFT Model (GPT-4o, etc.)**

- Neural network simulation of a data labeler at OpenAI
- As if you gave query to a data labeler who read all labeling instructions, spent hours writing ideal response
- But we didn't wait hours—we get a simulation of that process
- Neural networks don't function like human brains—what's easy/hard differs
- We're getting a lossy simulation

**The RL Model (o1, o3, DeepSeek-R1)**

- Not just straightforward simulation of human data labeler
- Something new, unique, and interesting
- Function of thinking that was emergent in simulation
- Comes from reinforcement learning process
- For reasoning problems: shines and is truly new
- Open question: do thinking strategies from verifiable domains transfer to unverifiable domains?

**Current State**

- RL is too new, primordial, nascent
- Seeing beginnings of hints of greatness
- In principle capable of "Move 37" equivalent—but for open domain thinking
- Capable of analogies no human has had
- Incredibly exciting, but very early
- Will mostly shine in verifiable domains (math, code)

**Practical Guidance**

- Use these models all the time—they dramatically accelerate work
- Huge wealth creation coming from these models
- Be aware of shortcomings—even RL models have issues
- Use as tools in a toolbox—don't trust fully
- They will randomly do dumb things, hallucinate, skip mental arithmetic, can't count sometimes
- Check their work, own the product of your work
- Use for inspiration, first drafts, asking questions—but always verify

**Key Takeaway**: SFT models = simulation of ideal human response; RL models = emergent thinking patterns that are genuinely new; use both as tools, always verify.

---

## 🐇 RABBIT HOLE: How Do Numbers Encode Intelligence?

*A deep philosophical exploration about the nature of LLM intelligence*

### The Fundamental Question

"How the heck do a bunch of numbers become intelligent? What is that key deep inside? How do these numbers translate to intelligence?"

### Level 0: What Is Intelligence?

**Minimal definition**: Intelligence is the ability to take an input and produce an output that is useful given some goal.

Examples:
- Thermostat: temperature → on/off → maintains 70°F
- Human: visual input → muscle commands → catches a ball
- Model: text → next token → continues coherently

**Intelligence isn't magic—it's input-output mappings that achieve goals**

### Level -1: Functions Are All You Need

Any input-output mapping can be described as a mathematical function:
- f(pixels) → "mom" exists
- f("what's 2+2?") → "4" exists

**Intelligence, at its core, is just being the right function.**

The question becomes: how do you find that function?

### Why Numbers Can Do This

- Mathematics is the language of relationships
- Numbers can encode anything that has structure
- Operations on numbers can express any transformation
- A sufficiently complex composition of simple operations can approximate any function
- **The numbers aren't intelligent. The arrangement of numbers encodes a function. The function is what's intelligent.**

### The Lookup Table Insight

A function is just: "If you give me this, I give you that."

| Input | Output |
|-------|--------|
| 0     | 3      |
| 1     | 7      |
| 2     | 11     |
| 3     | 15     |

This table IS a function. But what about input 1.5?

### Numbers as Compressed Tables

Pattern in the table: **output = 4 × input + 3**

Now throw away the table. Just store two numbers: **4 and 3**

Those two numbers encode the function. Given any input (even 1.5): 4 × 1.5 + 3 = 9

**The numbers don't "contain" the outputs. They encode the RULE for computing outputs.**

### Scaling Up: More Numbers = More Complex Functions

- y = ax + b → 2 numbers (line)
- y = ax² + bx + c → 3 numbers (parabola)
- y = ax³ + bx² + cx + d → 4 numbers (cubic)

Each additional number lets the function express more complex shapes.

### Neural Networks: The Same Idea, Massively Scaled

- input → multiply by numbers → apply nonlinearity → multiply by more numbers → ... → output
- Small network: ~1,000 parameters
- Large language model: 100+ billion parameters

**Those billions of numbers, arranged in matrices, encode a function**

### The Formula Analogy

- The numbers don't contain intelligence
- **The numbers are coefficients in a formula**
- The formula, when executed with inputs, produces outputs
- Change the numbers → change which function you're computing → change what outputs you get
- **Training = searching for the numbers that make the formula produce correct outputs**

### Concrete Example

Formula: output = ??? × input + ???

Slots are empty holes in a recipe.

Plug in 4 and 3: output = 4 × input + 3
- Input 0 → 4×0 + 3 → Output 3
- Input 1 → 4×1 + 3 → Output 7
- Input 2 → 4×2 + 3 → Output 11

The formula is a **machine**. The numbers are **settings** on that machine.

### The Software Engineering Mental Shift

**Traditional programming:**
```python
def calculate(x):
    return 4 * x + 3  # Fixed constants
```
Function = fixed logic. Variables = changing data.

**Machine learning:**
```python
def calculate(x, w, b):
    return w * x + b  # Parameters to search for
```
The structure is fixed. The numbers are what you search for.
Different numbers = different function behavior.

**The inversion**: In software engineering, numbers are data and functions are fixed. In ML, **numbers ARE the function** (given fixed structure).

### How Do We Know the Right Output for LLMs?

You don't define "intelligent output" abstractly.

You take actual human text—billions of sentences humans wrote:
- "The capital of France is Paris."
- "She walked to the store."
- "2 + 2 = 4"

Hide the last word, make the model guess:
- "The capital of France is ___" → Target: "Paris"

**"Right output" = what came next in real human text**

Human text reflects facts, logic, grammar, reasoning patterns, common sense. By forcing prediction of real human text, numbers must encode whatever structure enables that prediction.

### The Foundation Summary

1. **Intelligence = being the right function**
2. **Numbers in slots = encoding a function**
3. **Training = searching for the numbers**
4. **"Right output" = what humans actually wrote**

Everything else—attention, embeddings, transformers, RLHF—is details built on this foundation, answering:
- "How do we make the formula more expressive?"
- "How do we search for numbers more efficiently?"
- "How do we refine what 'right output' means?"

---

## KEY INSIGHTS FROM COMPLETE VIDEO

1. **LLM training = human learning**: Pre-training (reading) → SFT (studying worked examples) → RL (practice problems)

2. **Human annotations have limits**: We don't know optimal solution formats for LLMs because our cognition differs from theirs

3. **RL discovers thinking**: Chain-of-thought, self-verification, backtracking—all emerge from optimization, not human programming

4. **DeepSeek-R1 was a watershed**: First public detailed explanation of RL for LLMs, showing how reasoning emerges

5. **Two types of models**: SFT models (GPT-4o) for general use; RL/thinking models (o1, o3, R1) for hard reasoning problems

6. **RLHF is useful but limited**: Enables RL in creative domains via reward models, but can be gamed—not "magical" RL

7. **The Move 37 potential**: RL in LLMs could eventually discover reasoning strategies no human has conceived

8. **Use as tools, verify always**: These models are powerful but have swiss-cheese capabilities—hallucinate, fail randomly, make arithmetic errors

9. **Numbers encode functions**: At the deepest level, intelligence is being the right function, and training is searching for the numbers that specify that function

10. **Tokens are thinking**: Intermediate token generation IS the reasoning process, not a byproduct

---

## Key Resources & References

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

---

## Ready for Day 4: Complete Foundation

✅ **Video Learning Complete**: All checkpoints from Andrej Karpathy's Deep Dive fully captured

✅ **Complete Pipeline Understanding**:
- Checkpoint 1: Pretraining flow (tokenization → embeddings → 12 layers → softmax)
- Checkpoint 2: Post-training revolution (base model → helpful assistant)
- Checkpoint 3: Hallucinations → Tool Use → Knowledge vs. Working Memory → CoT
- Checkpoint 4: RL strategies → DeepSeek-R1 → Thinking models → RLHF → Future directions
- Rabbit Hole: Deep understanding of how numbers encode intelligence

✅ **Hands-On Experiments Prepared**: All 4 experiments + synthesis ready in day4readingnotes.md

✅ **Theory Foundation Solid**: Complete understanding of pretraining → post-training → hallucinations → tool use → knowledge/memory → token thinking → reinforcement learning → thinking models

✅ **Mental Model Ready**: From "how transformers work" (Day 1) → "how LLMs are trained" (Day 2-3) → "why models behave the way they do" (complete)

→ **Next**: Execute the 4 hands-on experiments to validate theory in practice. See day4readingnotes.md for detailed methodology and templates.

---

*This document captures the complete learning journey through Andrej Karpathy's Deep Dive video. All video content—from pretraining mechanics through reinforcement learning and philosophical understanding—captured in one comprehensive reference.*

**Total content**: ~1,600 lines capturing complete video learning, including:
- 4 detailed checkpoints covering entire training pipeline
- Complete theory of LLM behavior
- Practical guidance for model usage
- Philosophical understanding of intelligence at numerical level
- Key insights for moving into practical implementation
