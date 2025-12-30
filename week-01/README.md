# Week 1: Foundations & First Ship

**Dates**: Dec 28, 2025 - Jan 3, 2026
**Status**: Not Started
**Time Commitment**: 40-50 hours (Holiday Week - Aggressive!)

## Goals
- ✅ Build mental model of AI engineering stack
- ✅ Understand embeddings and vector similarity
- ✅ Ship first semantic search tool
- ✅ Get comfortable with Claude Code

## Project: Semantic Code Search

Build a CLI tool that lets you semantically search through codebases.

### Example Usage
```bash
semantic-search "how does job matching work?"
# Returns relevant code snippets from the job-search-agent codebase
```

## Day-by-Day Plan

### Day 1-2: World View (16 hours)
**Saturday-Sunday, Dec 28-29, 2025**

**Reading** (12+ hours) - Follow this sequence:
⚠️ **Note**: Time estimates account for deep dives and rabbit holes. You learn by asking questions, not just skimming. Go deep.

1. [x] **Claude Code Quickstart** (30 min) - Learn your development tool first ✅ **DONE**
   - [Claude Code Quickstart Guide](https://code.claude.com/docs/en/quickstart.md)
   - [Common Workflows Reference](https://code.claude.com/docs/en/common-workflows.md)

2. [x] **Stack Overflow: Intuitive Intro to Embeddings** (5+ hours with deep dives) ✅ **DONE**
   - [An Intuitive Introduction to Text Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
   - **Why this**: Clear analogies, explains high-dimensional spaces intuitively
   - *Actual: Went deep on RNNs, LSTMs, Transformers - rabbit holes = understanding*

3. [ ] **Andrej Karpathy: Deep Dive into LLMs like ChatGPT** (3.5+ hours) 🔄 **IN PROGRESS** (~50% done)
   - STRATEGIC CHOICE: Chose depth over quick 1-hour intro
   - ✅ Checkpoint 1: Complete Pretraining Flow (tokenization → transformer → output)
   - ✅ Checkpoint 2: Post-Training Revolution (base model → helpful assistant)
   - 🔄 Currently watching: Hallucinations section and beyond
   - Still to watch: Tool use, knowledge, spelling, jagged intelligence, fine-tuning, RL, RLHF, future directions
   - All learnings documented in `notes/day2-deep-dive-notes.md` (updated as you watch)

4. [ ] **Jay Alammar's Illustrated Word2Vec** (30-45 min) - OPTIONAL
   - [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
   - Can revisit after completing Deep Dive if needed

5. [ ] **Weaviate: Vector Embeddings Explained** (30-45 min) - OPTIONAL
   - [Vector Embeddings Explained](https://weaviate.io/blog/vector-embeddings-explained)
   - Can revisit after completing Deep Dive if needed

6. [ ] **Pinecone: Vector Database** (45 min - 1 hour) - OPTIONAL
   - [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
   - Can revisit after completing Deep Dive if needed

**Exercise - Mental Model** 🔄 (Building through Deep Dive):
- [x] Checkpoint 1-2: Pretraining & Post-Training flow (documented)
- 🔄 Continuing: Add remaining sections as you watch
- [ ] Final: Complete mental model after full video
  ```
  Raw Internet Text → Tokenization (BPE)
           ↓
  Token Embeddings (768 dimensions)
           ↓
  12 Transformer Layers with Attention
           ↓
  Output Distribution (50k vocab scores)
           ↓
  Softmax → Next token prediction

  Then: Post-training on conversation data
  And: Reinforcement Learning for alignment (RLHF)
           ↓
  Base model → Helpful, aligned assistant
  ```
- [ ] Document complete understanding in `notes/day2-deep-dive-notes.md`
- [ ] Understand full spectrum: pretraining → fine-tuning → RL → RLHF
- [ ] Ready for Day 3 hands-on experiments

**Deliverable**:
- Mental model diagram
- Notes on key concepts (embeddings, vector similarity, semantic search, pre-trained models)
- Questions list for deeper research
- Comfort with Claude Code basics

---

### Day 3: Hands-On Experimentation with Core Concepts (8 hours)
**Monday, Dec 30, 2025**

**Goal**: Validate your strong foundation from Day 1-2 learning through practical experiments. You now understand the theory deeply - time to see it work in code.

**Setup** (1 hour):
- [ ] Initialize project: `semantic-code-search`
- [ ] Set up Python environment (pyenv, venv)
- [ ] Install dependencies (openai, chromadb, click, rich)
- [ ] Configure OpenAI API key

**Experiment 1: Embeddings Generation** (1.5 hours)
- [ ] Use OpenAI API to generate embeddings for sample texts
- [ ] Verify embedding dimensions (1536 for text-embedding-3-small)
- [ ] Test with: simple words (cat, dog), phrases (software engineering), code snippets
- [ ] Observe: Do similar semantic meanings have similar embeddings?
- [ ] Document in `notes/experiments.md`: Example embeddings and observations
- **Theory connection**: See the 768-dim vectors from Deep Dive become real 1536-dim vectors

**Experiment 2: Vector Similarity & Distance Metrics** (1.5 hours)
- [ ] Implement cosine similarity calculation from scratch
- [ ] Test with embedding pairs: (cat, dog), (cat, car), (function, method)
- [ ] Compare cosine vs Euclidean distance
- [ ] Verify: Does cosine distance match your intuition about semantic similarity?
- [ ] Document in `notes/experiments.md`: Distance calculations and comparison
- **Theory connection**: Validate the attention mechanism's softmax weighting in practice

**Experiment 3: ChromaDB Storage & Retrieval** (1.5 hours)
- [ ] Create ChromaDB collection locally
- [ ] Add 10 code snippets with embeddings (from a real codebase)
- [ ] Test similarity search: query "job matching algorithm"
- [ ] Verify returned snippets are semantically relevant
- [ ] Experiment with different numbers of results
- [ ] Document in `notes/experiments.md`: Query results and relevance assessment
- **Theory connection**: See post-training's "helpful assistant" concept in search results

**Experiment 4: Chunking Strategies** (1.5 hours)
- [ ] Extract 3 functions from a real codebase
- [ ] Try multiple chunking approaches:
  - Chunk by function (semantic unit)
  - Chunk by semantic blocks (5-10 lines)
  - Chunk by line count (20 lines per chunk)
- [ ] Generate embeddings for each approach
- [ ] Query same semantic question against each approach
- [ ] Compare: Which chunking strategy gives best results?
- [ ] Document in `notes/experiments.md`: Chunking comparison and trade-offs
- **Key insight**: This is where understanding from Deep Dive becomes practical wisdom

**Documentation & Synthesis** (1 hour):
- [ ] Document all findings in `notes/experiments.md`
- [ ] Create preliminary `notes/architecture.md` based on learnings
- [ ] Write down insights about each concept
- [ ] Identify potential edge cases for production build
- [ ] Note: How Day 1-2 theory manifests in practical code

---

### Day 4: Build Core (8 hours)
**Tuesday, Dec 31, 2025**

**Core Implementation**:
- [ ] Implement code file parser
- [ ] Build chunking logic (function-level, file-level)
- [ ] Create embedding pipeline
- [ ] Implement ChromaDB storage layer
- [ ] Build search function

**Testing**:
- [ ] Test with job-search-agent codebase
- [ ] Verify search results quality
- [ ] Iterate on chunking strategy

---

### Day 5: Polish & Ship (8 hours)
**Wednesday, Jan 1, 2026**

**CLI Development**:
- [ ] Build Click CLI interface
- [ ] Add Rich formatting for beautiful output
- [ ] Implement progress bars for indexing
- [ ] Add configuration file support

**Features**:
- [ ] Index command: `semantic-search index <path>`
- [ ] Search command: `semantic-search query "..."`
- [ ] List command: `semantic-search list`
- [ ] Stats command: `semantic-search stats`

**Documentation**:
- [ ] Write comprehensive README
- [ ] Add usage examples
- [ ] Create API documentation
- [ ] Add architecture diagram

---

### Day 6-7: Blog & Reflection (8 hours)
**Thursday-Friday, Jan 2-3, 2026**

**Blog Post**:
- [ ] Write "I Built Semantic Search for My Codebase in 3 Days"
- [ ] Include code snippets
- [ ] Explain key learnings
- [ ] Add demo GIFs/screenshots

**Polish**:
- [ ] Code cleanup
- [ ] Add tests
- [ ] Improve error handling
- [ ] Performance optimization

**Publish**:
- [ ] Push to GitHub
- [ ] Publish blog post
- [ ] Share on LinkedIn/Twitter
- [ ] Update portfolio

---

## Tech Stack

### Core
- **Python 3.9+**
- **OpenAI API** - text-embedding-3-small
- **ChromaDB** - Local vector database
- **Click** - CLI framework
- **Rich** - Terminal UI

### Development
- **pytest** - Testing
- **black** - Code formatting
- **mypy** - Type checking

## Key Learnings Expected

### Technical
- How embeddings represent semantic meaning
- Vector similarity search mechanics
- ChromaDB operations and indexing
- CLI design patterns

### Conceptual
- Trade-offs in chunking strategies
- When to use semantic vs. keyword search
- Cost optimization for embeddings
- Production considerations

## Success Metrics

- [ ] Tool successfully indexes job-search-agent codebase
- [ ] Search finds relevant code >80% of the time
- [ ] CLI is user-friendly and well-documented
- [ ] Code is clean, tested, and production-ready
- [ ] Blog post demonstrates deep understanding
- [ ] Shipped and shared on social media

## Resources

### Essential Reading (Day 1-2)
1. **Claude Code**
   - [Quickstart Guide](https://code.claude.com/docs/en/quickstart.md) - Start here!
   - [Common Workflows](https://code.claude.com/docs/en/common-workflows.md) - Ongoing reference
   - [Full Documentation](https://code.claude.com/docs/)

2. **Embeddings & Vector Search** (Conceptual)
   - [Stack Overflow: Intuitive Intro to Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/) - Best starting point
   - [Jay Alammar: Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Visual learning
   - [Weaviate: Vector Embeddings Explained](https://weaviate.io/blog/vector-embeddings-explained) - Semantic search context
   - [Pinecone: Vector Database](https://www.pinecone.io/learn/vector-database/) - Architecture deep dive

3. **Additional Conceptual Resources** (Optional)
   - [Google ML Crash Course: Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings) - Interactive, comprehensive
   - [Hugging Face: Getting Started With Embeddings](https://huggingface.co/blog/getting-started-with-embeddings) - Practical applications
   - [AWS: What is Embedding?](https://aws.amazon.com/what-is/embeddings-in-machine-learning/) - Business context

4. **Implementation Reference (Day 3-5)**
   - [ChromaDB Documentation](https://docs.trychroma.com/)
   - [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) - API reference
   - [Pinecone Search Guide](https://docs.pinecone.io/guides/search/search-overview) - Advanced patterns

### Videos
- [Andrej Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) - Must watch!
- [Vector Databases Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4) - Optional

### Code Examples
- Look at how GitHub Copilot indexes code
- Study existing semantic search implementations
- Review ChromaDB examples for Python CLI apps

## Daily Log

Use `notes/daily-log.md` to track:
- What you built today
- What you learned
- Challenges faced
- Tomorrow's plan

## Questions to Answer

By end of week, you should be able to explain:
1. What are embeddings and how do they work?
2. How does cosine similarity measure semantic relevance?
3. What are the trade-offs between different chunking strategies?
4. When should you use semantic search vs. keyword search?
5. How do you evaluate semantic search quality?

---

**Ready to start?** Begin with Day 1 reading and mental model building. Use Claude Code to help you build, but make sure you understand every line of code!

**Next Week**: Week 2 - Vector Databases Deep Dive
