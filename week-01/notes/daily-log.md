# Week 1 Daily Log

## Day 1 - Saturday, Dec 28, 2025

### Goals (Original)
- [ ] Read Pinecone vector search guide
- [ ] Read OpenAI embeddings documentation
- [ ] Watch Andrej Karpathy video
- [ ] Create mental model diagram

### What I Actually Did ✅
1. **Claude Code Quickstart** (30 min)
   - Learned basic commands, shortcuts, permission modes
   - Ready to use as development partner throughout the journey

2. **Stack Overflow: Intuitive Intro to Embeddings** (3-4 hours)
   - Distance Metrics: Euclidean vs Cosine distance
   - Latent Dimensions: Dimensionality reduction (30K words → 2 dimensions)
   - Pre-Trained Models: KEY INSIGHT - models are pre-trained, embedding lookup at runtime
   - Word2vec: Context windows, sliding through text, embedding accumulation
   - Neural Networks: Why semantic understanding is necessary beyond counting

3. **Rabbit Hole - RNNs** (1+ hour)
   - Sequential token-by-token processing
   - Vanishing gradient problem (0.8^13 = 0.055)
   - Error signal multiplication and proportional blame assignment
   - Why RNNs fail at long-distance relationships

4. **Rabbit Hole - LSTMs** (45+ min)
   - Three gates: input, forget, output
   - Memory cell as running vector
   - How gates manage what gets stored, forgotten, and exposed
   - Solved RNN vanishing gradient through memory preservation

5. **Rabbit Hole - Transformers** (1+ hour)
   - Key-value store analogy
   - Five components per token: embedding, key, value, query, contextual representation
   - Attention mechanism and similarity scoring
   - 12 layers of progressive refinement
   - Complete end-to-end example: "The big dog ran" → predict "fast"

### What I Learned
**Major Insights:**
- Embeddings are dimensionality reduction capturing semantic meaning
- Pre-training vs runtime computation distinction - CRITICAL mental shift
- Architecture evolution: each solves previous problems
  - RNNs: Sequential but vanishes long-distance info
  - LSTMs: Memory gates preserve important info
  - Transformers: Parallel + no memory bottleneck
- Why each architecture works: stability (key/value) + flexibility (query) + relevance (attention)

**Technical Depth Gained:**
- Understand weights as multipliers per dimension (not per word)
- Proportional blame: error × weight = how much to blame that component
- Catch-22 of RNNs: weights < 1 → vanishing, weights > 1 → exploding
- Pre-computed and reused vectors (key, value) vs dynamic vectors (query)
- How attention creates weighted blend of all tokens based on similarity

**Questions Answered:**
- Why do we need neural networks?
- Where do embeddings come from?
- How do networks learn?
- Why do RNNs fail?
- How do LSTMs fix it?
- How do transformers work end-to-end?

### Challenges
- **Went much deeper than planned** - rabbit holes took 2+ extra hours
  - Initial estimate: 15 min per article
  - Actual: 45 min - 1.5 hours per section with deep dives
  - But this depth is exactly the right learning style for me

- **Late night energy** - too tired for 1.5-hour Karpathy video
  - Better to do it with fresh mind, not force it late

- **Material is complex** - transformers section was dense and required re-reading
  - But now have concrete understanding, not superficial

### Wins
- ✅ Completed 5 hours of deep, meaningful learning
- ✅ Created comprehensive raw notes with examples
- ✅ Understood the complete evolution: embeddings → RNNs → LSTMs → Transformers
- ✅ Captured concrete examples (cat/dog, "The big dog ran fast", error signals)
- ✅ Gained confidence with Claude Code tool
- ✅ Built strong foundation for understanding LLMs

### Tomorrow's Plan
- Watch Andrej Karpathy's LLM intro video (with fresh mind)
- Create mental model diagram synthesizing all learning
- Continue with optional readings (Jay Alammar, Weaviate, Pinecone) if energy allows
- Prepare for Day 3 coding (semantic-code-search project)

---

## Day 2 - Sunday, Dec 29, 2025

### Goals
- [x] Claude Code Tutorial (Complete)
- [x] Complete mental model (STRONG foundation via Deep Dive)
- [x] Document understanding (400+ lines in running notes)
- [x] Prepare for hands-on work ✅

### What I Actually Did ✅

1. **Claude Code Tutorial - Fundamentals** (1-2 hours)
   - Learned basic commands and slash commands (/help, /clear, /agents, /mcp, /doctor)
   - Understood permission modes (read, edit, write, execute)
   - Explored available agents: general-purpose, Explore, Plan, claude-code-guide, fact-validator-agent
   - Practiced git workflows and commit operations
   - Gained confidence with Claude Code as primary development partner

2. **Andrej Karpathy: Deep Dive into LLMs like ChatGPT** (3.5+ hours) - STRATEGIC CHOICE
   - **Decision**: Chose depth (3.5h deep dive) over quick intro (1h)
   - **Checkpoint 1: Complete Pretraining Flow**
     - Text → Tokenization (BPE: 256 bytes → 50k tokens)
     - Token embeddings → 12-layer transformer with attention mechanism
     - Query/Key/Value matrices, softmax, weighted aggregation
     - 1.6B parameters for GPT-2
     - End-to-end: "The big dog ran" → "fast" (full trace through model)
     - Softmax breakthrough: exp(scores) → normalized probabilities

   - **Checkpoint 2: Post-Training Revolution** (KEY INSIGHT)
     - Base model = "internet document simulator" (predicts next token like web text)
     - Post-training = teaches model to "think like helpful human labeler"
     - InstructGPT paper (2022): quality labeling > quantity of data
     - Three phases: conversation tokenization → human labeling → synthetic data evolution
     - Labeling instructions (100+ pages): helpful, harmless, honest framework
     - Modern trend (2023-2025): LLM-assisted refinement, not scratch writing
     - Concrete examples: base model vs. instruct model on harmful requests
     - The psychology: ChatGPT = statistical simulation of helpful experts

3. **Created Running Notes Structure**
   - Created `day2-deep-dive-notes.md` (400+ lines)
   - Two complete checkpoints with diagrams and examples
   - All resource links captured (Tiktokenizer, visualizers, papers)
   - Approach validated: detailed notes during learning, summary in daily-log at end of day

### What I Learned
**Pretraining Insights:**
- Tokenization strategy is critical (BPE efficiency)
- Attention mechanism = learned weighted lookup of relevant tokens
- Softmax amplifies important differences (exp function is brilliant)
- Parameters learned through gradient descent over millions of iterations
- Training: weeks/months; Inference: milliseconds (frozen weights)

**Post-Training Insights:**
- The base model knows EVERYTHING (internet knowledge)
- Post-training teaches it HOW to share that knowledge helpfully
- Human labeling guidelines (100+ pages) encode company values
- Modern systems use LLM assistance for data generation (meta!)
- Quality > quantity: fewer high-quality labeled examples > many mediocre ones

**Mental Model Achievement:**
- Complete flow: Raw Internet → Tokens → Embeddings → 12 Transformer Layers → Output Distribution
- Connection: Day 1 insights (RNNs → LSTMs → Transformers) manifest directly in GPT architecture
- Why it works: Attention mechanism elegantly solves long-distance dependency problem
- The revolution: Post-training is where the "magic" happens (not pretraining as commonly thought)

### Challenges
- None - the depth was exactly right for learning style
- Initial concern about "wasting time" on deep dives was unfounded
- Deep dives create understanding, not distraction

### Wins
- ✅ Completed Deep Dive video (3.5+ hours) - chose depth over speed
- ✅ Created comprehensive running notes (400+ lines, structured)
- ✅ Built strong mental model of COMPLETE LLM pipeline (pretraining → post-training)
- ✅ Understanding shifted from "how does it work?" to "why does it work this way?"
- ✅ All learnings connected Day 1 insights to real implementation
- ✅ Made 4 GitHub commits with progress
- ✅ Prepared Day 3 hands-on work with solid foundation

### Still to Complete (Optional - can skip if time-pressed)
- Jay Alammar's Illustrated Word2Vec (visual reinforcement)
- Weaviate: Vector Embeddings Explained (practical context)
- Pinecone: Vector Database (architecture details)
- Create formal mental model diagram

**DECISION**: Skip the remaining readings. Day 3 hands-on will solidify knowledge better than more reading.

### Tomorrow's Plan (Day 3)
- Start hands-on experimentation
- Experiment 1: Embeddings generation with OpenAI API
- Experiment 2: Vector similarity calculations (cosine vs euclidean)
- Experiment 3: ChromaDB local storage and retrieval
- Experiment 4: Chunking strategy comparisons
- Document findings in `notes/experiments.md`

---

## Day 3 - Monday, Dec 30, 2025

### Goals
- [ ] Set up development environment
- [ ] Experiment with OpenAI embeddings
- [ ] Test ChromaDB locally
- [ ] Design system architecture

### What I Did
-

### What I Learned
-

### Challenges
-

### Wins
-

### Tomorrow's Plan
-

---

## Day 4 - Tuesday, Dec 31, 2025

### Goals
- [ ] Implement code parser
- [ ] Build chunking logic
- [ ] Create embedding pipeline
- [ ] Implement search function

### What I Did
-

### What I Learned
-

### Challenges
-

### Wins
-

### Tomorrow's Plan
-

---

## Day 5 - Wednesday, Jan 1, 2026

### Goals
- [ ] Build CLI interface
- [ ] Add Rich formatting
- [ ] Write documentation
- [ ] Test with real codebase

### What I Did
-

### What I Learned
-

### Challenges
-

### Wins
-

### Tomorrow's Plan
-

---

## Day 6 - Thursday, Jan 2, 2026

### Goals
- [ ] Write blog post
- [ ] Code cleanup and tests
- [ ] Polish documentation

### What I Did
-

### What I Learned
-

### Challenges
-

### Wins
-

### Tomorrow's Plan
-

---

## Day 7 - Friday, Jan 3, 2026

### Goals
- [ ] Final polish
- [ ] Publish to GitHub
- [ ] Share blog post
- [ ] Reflect on week

### What I Did
-

### What I Learned
-

### Challenges
-

### Wins
-

### Week Reflection
-

---

## Weekly Summary

### Total Hours
-

### Key Achievements
1.
2.
3.

### Most Important Learning
-

### Biggest Challenge
-

### What I'd Do Differently
-

### Excitement for Next Week
-
