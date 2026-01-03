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
- [x] Complete Andrej Karpathy Deep Dive video
- [x] Finalize mental model and synthesis
- [x] Prepare experiment structure for Day 4

### What I Actually Did ✅

1. **Completed Karpathy Deep Dive Video** (Remaining 6+ sections)
   - Hallucinations: Models filling gaps when context is insufficient
   - Tool Use and Knowledge vs Working Memory: External tools vs parameters
   - Models Need Tokens to Think: Chain-of-thought and intermediate steps
   - Tokenization Revisited: Spelling struggles, non-English, edge cases
   - Jagged Intelligence: Domain-specific competencies and failures
   - Reinforcement Learning: From SFT to RLHF
   - DeepSeek-R1, AlphaGo patterns, future directions

2. **Finalized Learning Documentation**
   - Updated day1-reading-notes.md with complete Day 1 insights
   - Created Date2-3-Deep-Dive-Notes.md with all 8+ checkpoints
   - Mapped theory to practical implications

3. **Established Experiment 1 Framework**
   - Defined 7-test structure mapping to Day 1-3 learning
   - Identified what each test validates
   - Connected theory to hands-on validation approach

### What I Learned
**Integration Insights:**
- Hallucinations stem from insufficient context, not model failure
- Tool use and working memory are critical for accurate answers
- Tokenization is a fundamental limitation affecting model behavior
- Chain-of-thought works because intermediate tokens enable thinking
- Intelligence is domain-specific, not uniform across all knowledge

**Architecture Clarity:**
- Why RAG systems work: providing working memory prevents hallucinations
- Why chunking matters: good chunks = good context = good answers
- Why caching works: embeddings are pre-computed and stable

### Challenges
- None - learning trajectory was smooth and building well

### Wins
- ✅ Completed comprehensive 8+ hour deep dive on transformer architecture
- ✅ Created structured learning notes for future reference
- ✅ Identified clear connection between theory and Day 4 experiments
- ✅ Prepared detailed experiment framework with 7 validation tests

### Tomorrow's Plan
- Implement 7 embeddings validation tests
- Run experiments and document findings
- Extract architecture decisions for rag-code-qa project

---

## Day 4 - Tuesday, December 31, 2025

### Goals
- [x] Create Experiment 1 structure with clear specifications
- [x] Define 7 validation tests mapping to Day 1-3 theory
- [x] Prepare test templates and learning objectives
- [x] Create detailed learning documentation

### What I Actually Did ✅

1. **Established Experiment 1 Testing Standards**
   - Created SPEC.md: Detailed PRD with purpose, background, and test specifications
   - Created README.md: Setup guide and test execution instructions
   - Defined clear methodology: theory → hypothesis → test → validation

2. **Built 7 Test Files with Complete Structure**
   - test_01_pretraining.py: Embeddings determinism (caching validation)
   - test_02_semantic_clustering.py: Multi-language semantic similarity
   - test_03_dimensionality.py: 1536 vs 384 dimension trade-offs
   - test_04_distance_metrics.py: Cosine vs Euclidean comparison
   - test_05_relationships.py: Programming paradigm clustering
   - test_06_chunking.py: Function-level vs fixed-size chunking
   - test_07_working_memory.py: RAG retrieval quality validation
   - Each with large docstring headers mapping to Day 1-3 concepts

3. **Created Supporting Infrastructure**
   - run_all.py: Orchestrator to run all 7 tests in sequence
   - results.md template: Structure for documenting findings
   - requirements.txt: Dependencies (openai, numpy, python-dotenv)

4. **Comprehensive Learning Documentation**
   - day4readingnotes.md: Detailed notes for each of 7 tests
   - Each test documented with core insight, what you learn, why it matters, connections to theory
   - Format: practical learning objectives rather than abstract specifications

### What I Learned (Through Testing Framework)
**Test Design Insights:**
- Test 1: Embeddings are pre-computed lookups → safe to cache
- Test 2: Semantic meaning transcends syntax → multi-language works
- Test 3: Dimensionality trade-off quantified (7% improvement)
- Test 4: Cosine distance optimal for embeddings → semantic matching
- Test 5: Language paradigms are encoded in embeddings
- Test 6: Chunking strategy is highest-impact architectural decision
- Test 7: RAG pattern works → good retrieval prevents hallucinations

**Architecture Decisions Informed:**
- Embedding model: text-embedding-3-small (1536 dims)
- Distance metric: cosine similarity
- Chunking strategy: by function (semantic units)
- Caching: safe and necessary for performance
- Multi-language support: viable across Python, JavaScript, TypeScript

### Challenges
- None - standards-based approach clarified exactly what tests should do
- Tangential learning on embedding strategies, multi-language architecture, chunking in real products

### Wins
- ✅ Created standards-based testing framework mapping theory to practice
- ✅ 7 tests with clear docstrings and TODO templates ready for implementation
- ✅ Detailed learning objectives for each test prepared
- ✅ Complete infrastructure for running and documenting experiments
- ✅ Clear methodology established: validate theory before building architecture

### Tomorrow's Plan
- Run all 7 tests with actual OpenAI API calls
- Document findings in results.md
- Extract concrete architecture decisions for rag-code-qa/

---

## Day 5 - Wednesday, January 1, 2026

### Goals
- [x] Implement parser module (Component 1)
- [x] Implement chunker module (Component 2)
- [x] Create modular package structure
- [x] Document with README & LLD for each module

### What I Actually Did ✅

1. **Parser Module - Complete Implementation** (src/parser/)
   - Modular package structure with 6 focused files
   - `models.py`: ParsedFunction dataclass
   - `python_parser.py`: AST-based Python parsing
   - `javascript_parser.py`: Regex-based JS/TS parsing
   - `dispatcher.py`: File routing strategy
   - `directory_walker.py`: Recursive traversal with smart filtering
   - `__init__.py`: Public API
   - **Comprehensive documentation**: README.md (474 lines) + LLD.md (635 lines)
   - **Features**: Nested functions, class methods, async functions, docstrings preserved, absolute paths, graceful error handling

2. **Chunker Module - Complete Implementation** (src/chunker/)
   - Modular package structure with 6 focused files
   - `models.py`: Chunk dataclass with deterministic IDs
   - `formatter.py`: Three-layer format (location + docstring + code) + truncation
   - `token_counter.py`: tiktoken integration for accurate counting
   - `hasher.py`: SHA-256 deterministic ID generation
   - `chunker.py`: Main orchestrator (format → count → hash)
   - `__init__.py`: Public API
   - **Comprehensive documentation**: README.md (461 lines) + LLD.md
   - **Features**: Token limit validation (8000 tokens), intelligent truncation, metadata preservation, batch processing

3. **Embedder Module - Complete Implementation** (src/embedder/)
   - Modular package structure with 5 focused files
   - `models.py`: EmbeddingResult dataclass
   - `api_client.py`: OpenAI API integration with error handling
   - `cache.py`: File-based and in-memory caching (deterministic lookups)
   - `embedder.py`: Main orchestrator (batch processing + rate limiting)
   - `__init__.py`: Public API
   - **Comprehensive documentation**: README.md (16K) + LLD.md (12K)
   - **Features**: Batch processing (efficiency), exponential backoff (rate limits), dual-layer caching, deterministic results validation

### What I Learned

**Parser Design Patterns:**
- AST parsing for Python (reliable, handles all syntax)
- Regex for JS/TS (simpler, no dependencies)
- Visitor pattern for tree traversal
- Graceful degradation (one bad file doesn't crash pipeline)
- Directory filtering prevents embedding noise (node_modules, __pycache__, .git)

**Chunker Design Insights:**
- Three-layer formatting optimizes embedding quality (location + semantics + code)
- Deterministic IDs enable change detection and incremental updates
- tiktoken vs character counting: subword tokenization matters
- Intelligent truncation preserves semantic units (header + docstring, truncate code)
- Content-based hashing: same input = same ID always

**Embedder Design Insights:**
- Determinism enables aggressive caching (identical vectors for same input)
- Batch processing reduces API overhead (~10x efficiency improvement)
- Exponential backoff handles rate limits gracefully
- Dual-layer caching (in-memory + file-based) balances performance and persistence
- API costs are significant - caching is not optimization, it's necessity

**Modular Architecture Benefits:**
- Each file has single responsibility (models, API, cache, orchestration)
- Easy to test independently
- Easy to extend (swap API clients, caching backends)
- Code is maintainable and understandable

### Challenges
- None - modular design made implementation clean and focused

### Wins
- ✅ Implemented 3 complete modules with comprehensive documentation
- ✅ Created modular package structure (not monolithic files)
- ✅ 1,600+ lines of documentation (README + LLD across 3 modules)
- ✅ Validated design through implementation
- ✅ Ready for next components (storage, retriever, generator)
- ✅ **43% progress** (3/7 components complete)

### Tomorrow's Plan
- Implement storage.py (Component 4) - ChromaDB management + persistence
- Implement retriever.py (Component 5) - Similarity search

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
