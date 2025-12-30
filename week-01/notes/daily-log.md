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
- [ ] Complete mental model (In progress - Deep Dive video ~50% done)
- [ ] Document understanding (Building in running notes)
- [ ] Prepare for hands-on work

### What I Actually Did ✅ (In Progress)

1. **Claude Code Tutorial - Fundamentals** (1-2 hours)
   - Learned basic commands and slash commands (/help, /clear, /agents, /mcp, /doctor)
   - Understood permission modes (read, edit, write, execute)
   - Explored available agents: general-purpose, Explore, Plan, claude-code-guide, fact-validator-agent
   - Practiced git workflows and commit operations
   - Gained confidence with Claude Code as primary development partner

2. **Andrej Karpathy: Deep Dive into LLMs like ChatGPT** (In Progress - 50% complete)
   - **Decision**: Chose depth (3.5h+ deep dive) over quick intro (1h)

   - **✅ CHECKPOINT 1: Complete Pretraining Flow** (Watched & Documented)
     - Text → Tokenization (BPE: 256 bytes → 50k tokens)
     - Token embeddings → 12-layer transformer with attention mechanism
     - Query/Key/Value matrices, softmax, weighted aggregation
     - 1.6B parameters for GPT-2
     - End-to-end: "The big dog ran" → "fast" (full trace)
     - [Full details in day2-deep-dive-notes.md]

   - **✅ CHECKPOINT 2: Post-Training Revolution** (Watched & Documented)
     - Base model = "internet document simulator"
     - Post-training = "think like helpful human labeler"
     - InstructGPT paper (2022): quality > quantity
     - Three phases: tokenization → labeling → synthetic data
     - [Full details in day2-deep-dive-notes.md]

   - **🔄 CONTINUING NOW:**
     - [ ] Hallucinations, tool use, knowledge/working memory
     - [ ] Knowledge of self
     - [ ] Models need tokens to think
     - [ ] Tokenization revisited: spelling struggles
     - [ ] Jagged intelligence
     - [ ] Supervised fine-tuning to reinforcement learning
     - [ ] Reinforcement learning
     - [ ] DeepSeek-R1
     - [ ] AlphaGo
     - [ ] RLHF (Reinforcement Learning from Human Feedback)
     - [ ] Preview of things to come
     - [ ] Keeping track of LLMs
     - [ ] Where to find LLMs
     - [ ] Grand summary

3. **Running Notes Being Updated in day2-deep-dive-notes.md**
   - Will add checkpoints 3-8+ as video continues
   - All findings documented with resource links
   - Approach: detailed notes during watching, summary here at end of day

### What I Learned So Far (Checkpoints 1-2)
**Pretraining Insights:**
- Tokenization (BPE) is the critical first step
- Attention mechanism elegantly solves long-distance dependencies
- Softmax amplifies important differences (exp function insight)
- Training takes weeks/months; inference is milliseconds

**Post-Training Insights:**
- Base model knows everything from internet; post-training teaches helpfulness
- Quality of labeling instructions > quantity of data
- Modern systems using LLM-assisted synthetic data
- InstructGPT paper (2022) was the breakthrough

**Still to Learn in Remaining Sections:**
- How models hallucinate and how to detect it
- Tool use and knowledge/working memory implications
- Tokenization limitations (spelling struggles)
- Why intelligence is "jagged" across domains
- Full spectrum: Fine-tuning → Reinforcement Learning → RLHF
- Real-world models: DeepSeek-R1, AlphaGo patterns
- Future directions and where to find/track LLMs

### Challenges
- None yet - deep dive format is engaging and clear

### Wins So Far
- ✅ Chose depth (3.5h+) over lightweight intro
- ✅ Created running notes system (day2-deep-dive-notes.md)
- ✅ Captured first 2 major checkpoints thoroughly
- ✅ Connected Day 1 insights to Day 2 implementation details
- ✅ Made 5 GitHub commits tracking progress

### Plan for Rest of Day 2
- Continue Deep Dive video from hallucinations section
- Add checkpoints 3-8+ to day2-deep-dive-notes.md as you watch
- Finish entire video before end of day
- At end: finalize daily-log with complete summary
- Then start Day 3 hands-on experiments tomorrow

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
