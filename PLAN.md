# 8-Week AI Engineering Mastery Plan
*From Director to Shipping AI Engineer*

**Author**: Ankur
**Start Date**: December 28, 2025
**End Date**: February 22, 2026
**Story Arc**: "Building Production AI Systems: From Embeddings to Autonomous Agents"

---

## Your Learning Philosophy (Applied)
- **Week 1**: World view + foundations + quick wins
- **Weeks 2-8**: Progressive depth with weekly shipping milestones
- **Approach**: Build mental models first, then dive deep into implementation
- **Tools**: Claude Code for development, real-world projects for learning

---

## Week 1: RAG-Powered Code Q&A Assistant
**Dates**: Dec 28, 2025 - Jan 3, 2026
**Goal**: Ship a RAG-powered code assistant that answers questions (not just searches)
**Time Commitment**: 40-50 hours

### Learning Objectives
- ✅ Deep understanding of LLM mechanics (pretraining → post-training → RL)
- ✅ Embeddings, vector similarity, and semantic search fundamentals
- ✅ RAG architecture (retrieval → working memory → generation)
- ✅ Working memory vs knowledge (preventing hallucinations)
- ✅ Development workflow with Claude Code
- ✅ First production-ready RAG tool shipped

### Day-by-Day Breakdown

**Day 1-3: Deep Learning Foundation (24 hours)** ✅ COMPLETE
**Actual path taken** (depth over breadth):
1. ✅ Claude Code Quickstart (30 min)
2. ✅ Stack Overflow: Embeddings Deep Dive (5+ hours with RNN/LSTM/Transformer rabbit holes)
3. ✅ Andrej Karpathy's "Deep Dive into LLMs like ChatGPT" (3.5 hours - chose depth over 1.5hr intro)
   - Complete pretraining flow (tokenization → transformers → output)
   - Post-training revolution (base model → helpful assistant)
   - Hallucinations, tool use, working memory vs knowledge
   - Chain-of-thought and why tokens are thinking
   - Reinforcement learning (DeepSeek-R1, thinking models, RLHF)
   - Deep philosophical understanding (how numbers encode intelligence)

**Result**: 1,600+ lines of detailed notes capturing complete LLM understanding

**Day 4: Validate Theory + Start Building (8 hours)**

*Morning: Quick Validation (3 hours)*
- **Experiment 1**: Embeddings work (45 min) - validate semantic similarity
- **Experiment 2**: ChromaDB retrieval (1.5 hrs) - test retrieval quality, decide chunking
- **Experiment 3**: Basic RAG test (45 min) - retrieve → Claude → answer with citations

*Afternoon: Core Build (5 hours)*
- Set up project structure
- Implement code parser (extract functions/classes)
- Build chunking logic (by function - semantic units)
- Start embedding pipeline

**Day 5: Complete RAG Pipeline (8 hours)**
- **Indexing Pipeline** (3 hrs): Parse → Chunk → Embed → Store in ChromaDB
- **Retrieval Pipeline** (2 hrs): Query → Embed → Search → Retrieve top-k with context
- **Generation Pipeline** (2 hrs): Retrieved code → prompt → Claude → answer with citations
- **Testing** (1 hr): Test with job-search-agent codebase

**Day 6: CLI + Polish (6 hours)**
- Build Click CLI interface (`code-qa ask/explain/search/stats`)
- Rich formatting (syntax highlighting, tables)
- Configuration file support
- Error handling and caching

**Day 7: Documentation + Blog (6 hours)**
- Comprehensive README with architecture diagram
- Blog post: "I Learned How LLMs Work, Then Built Code Q&A"
- Connect theory → practice (working memory, hallucination prevention, CoT)
- Push to GitHub and share

### What You'll Build

```bash
code-qa ask "How does the job matching algorithm work?"

# Returns natural language answer with citations:
# "The job matching algorithm works in three stages:
#
# 1. Candidate Scoring (JobMatcher.calculate_score):
#    - Extracts skills from resume using NLP
#    - Generates compatibility score (0-100)
#
# 2. Ranking (JobMatcher.rank_candidates):
#    - Sorts candidates by score
#
# 3. Recommendation (JobMatcher.recommend):
#    - Returns top N matches with explanations
#
# Sources:
# - src/matcher/job_matcher.py:45-78
# - src/matcher/scoring.py:12-34"

code-qa explain <file.py>     # High-level explanation
code-qa search "auth flow"    # Fallback to semantic search
code-qa stats                 # Show indexed stats
```

**Tech Stack**:
- Python
- OpenAI embeddings API (text-embedding-3-small)
- ChromaDB (local, simple for Week 1)
- Claude Sonnet (best for code understanding)
- Click CLI + Rich formatting

**Key Learnings** (connects to deep dive):
- Chunking by function → semantic units (from CoT understanding)
- Working memory > knowledge → retrieval prevents hallucinations
- Tool use pattern → orchestrator retrieves code, LLM processes
- Prompt engineering → how to present code context
- Citations → source tracking prevents hallucinations

**Deliverable**:
- GitHub repo: `code-qa-assistant`
- Working CLI tool (installed via pip)
- Blog post: "I Learned How LLMs Work, Then Built Code Q&A"
- Indexed job-search-agent codebase (demo ready)

**Success Metrics**:
- Can answer code questions accurately (>80%)
- Provides proper citations (file:line)
- Working CLI tool you'll actually use
- Blog post connects deep theory → practice
- Foundation for Weeks 4-5 RAG deep dives

### Resources for Week 1
**Day 1-2 Reading (in order):**
1. [Claude Code Quickstart](https://code.claude.com/docs/en/quickstart.md)
2. [Stack Overflow: Intuitive Intro to Embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
3. [Jay Alammar: Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
4. [Weaviate: Vector Embeddings Explained](https://weaviate.io/blog/vector-embeddings-explained)
5. [Pinecone: Vector Database](https://www.pinecone.io/learn/vector-database/)
6. [Andrej Karpathy's Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g)

**Optional Deep Dives:**
- [Google ML Crash Course: Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings)
- [Hugging Face: Getting Started With Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)

**Day 3-5 Implementation:**
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings API Reference](https://platform.openai.com/docs/guides/embeddings)
- [Claude Code Common Workflows](https://code.claude.com/docs/en/common-workflows.md)

---

## Week 2: Vector Databases Deep Dive
**Dates**: Jan 4 - Jan 10, 2026
**Goal**: Production-grade vector operations + understanding trade-offs
**Time Commitment**: 20-25 hours

### Project: "Personal Knowledge Base with Semantic Memory"
Build a system that ingests your notes, docs, bookmarks and lets you query semantically.

**What You'll Build**:
```bash
knowledge add <file/url>           # Ingests and indexes
knowledge search "AI agent patterns"  # Semantic search
knowledge similar <doc-id>         # Find related content
knowledge chat "explain RAG"       # LLM + context retrieval
```

**Tech Stack**:
- **Vector DB**: Qdrant (better than ChromaDB for production)
- **Language**: Python or TypeScript (your choice - good time to decide)
- **Storage**: PostgreSQL + pgvector (learn hybrid search)
- **LLM**: Claude API (use what you know)

**Key Learnings**:
- Vector DB architecture (HNSW, IVF, quantization)
- Metadata filtering + hybrid search
- Indexing strategies for different content types
- Performance optimization (batch operations, caching)

**Deliverable**:
- Production-ready knowledge base
- Comparison blog: "ChromaDB vs Qdrant vs pgvector: What I Learned"
- 1000+ personal documents indexed

**Success Metrics**:
- Can explain vector index algorithms (HNSW, IVF)
- System handles 10K+ documents efficiently
- Blog post demonstrates deep understanding

---

## Week 3: Production Vector Search
**Dates**: Jan 11 - Jan 17, 2026
**Goal**: Scale, performance, and real-world patterns
**Time Commitment**: 20-25 hours

### Project: "Multi-Tenant Semantic Search SaaS"
Build the backend for a hypothetical SaaS that provides semantic search as a service.

**What You'll Build**:
- REST API with tenant isolation
- Async indexing pipeline
- Usage tracking and rate limiting
- Admin dashboard for monitoring

**Tech Stack**:
- **Framework**: FastAPI (Python) or Express + TypeScript
- **Vector DB**: Pinecone (learn managed services)
- **Queue**: Redis + Bull/Celery for async jobs
- **Monitoring**: Prometheus + Grafana basics

**Key Learnings**:
- Multi-tenancy in vector databases
- Async processing patterns
- API design for AI services
- Cost optimization (embedding caching, incremental updates)
- Security (API keys, data isolation)

**Deliverable**:
- Deployable API service
- Load testing results (100+ concurrent users)
- Technical deep-dive: "Architecting Multi-Tenant Vector Search"

**Success Metrics**:
- API handles 100+ concurrent requests
- Can explain multi-tenancy trade-offs
- Code demonstrates production-ready patterns

---

## Week 4: RAG Systems - Foundations
**Dates**: Jan 18 - Jan 24, 2026
**Goal**: Build production RAG from scratch
**Time Commitment**: 20-25 hours

### Project: "Technical Documentation Assistant"
Build a RAG system for any technical documentation (choose: Kubernetes, AWS, React, etc.)

**What You'll Build**:
```bash
docs-bot ask "How do I configure horizontal pod autoscaling?"
# Returns: Answer + source citations + confidence score
```

**Tech Stack**:
- **Framework**: Build from scratch first (understand before abstracting)
- **Vector DB**: Reuse Week 2/3 learnings
- **LLM**: Claude Sonnet (best for technical content)
- **Eval**: RAGAS or custom eval framework

**Key Learnings**:
- Chunking strategies (semantic, sliding window, recursive)
- Retrieval methods (similarity, MMR, reranking)
- Prompt engineering for RAG
- Citation and source tracking
- Evaluation metrics (answer relevance, faithfulness, context recall)

**Deliverable**:
- Working RAG system with >85% accuracy
- Evaluation framework and metrics
- Blog: "I Built a RAG System from Scratch: What LangChain Hides"

**Success Metrics**:
- System achieves >85% accuracy on test questions
- Can explain RAG architecture trade-offs
- Evaluation framework is reusable

**Key Innovation**:
Build your own orchestration layer - understand what frameworks abstract away.

---

## Week 5: Advanced RAG + Fine-Tuning
**Dates**: Jan 25 - Jan 31, 2026
**Goal**: Production-grade RAG with advanced techniques + fine-tuning mastery
**Time Commitment**: 20-25 hours

### Project: "Adaptive RAG System + Fine-Tuned Models"
Enhance Week 4's project with advanced patterns AND learn when to fine-tune vs RAG.

**Part 1: Advanced RAG Patterns** (12 hours)
1. **Query Routing**: Route questions to different retrieval strategies
2. **Reranking**: Cohere/Cross-encoder reranking
3. **Hypothetical Document Embeddings (HyDE)**: Generate hypothetical answers, embed those
4. **Agentic RAG**: Agent decides when to retrieve vs. when to answer directly
5. **Memory**: Conversation history and user preferences

**Part 2: Fine-Tuning** (8 hours)
1. **Fine-Tuning Basics**:
   - When to fine-tune vs RAG (decision framework)
   - Dataset preparation (format, quality, size)
   - OpenAI fine-tuning API workflow

2. **Hands-On Fine-Tuning**:
   - Prepare dataset for specific domain (technical docs, code, etc.)
   - Fine-tune GPT-3.5 or GPT-4 via OpenAI API
   - Evaluate fine-tuned model vs base model
   - Compare fine-tuned model vs RAG approach

3. **Cost/Performance Analysis**:
   - Training costs vs inference costs
   - When RAG is cheaper (most cases)
   - When fine-tuning wins (consistent domain, style)
   - Hybrid approach (fine-tune + RAG)

**Tech Stack**:
- **Reranker**: Cohere Rerank API
- **Agent Framework**: Build simple agent loop (don't use frameworks yet)
- **Memory**: Redis for session state
- **Fine-tuning**: OpenAI fine-tuning API
- **Evaluation**: Custom eval framework + RAGAS
- **Observability**: LangSmith or custom tracing

**Key Learnings**:
- Query understanding and classification
- Multi-stage retrieval pipelines
- Agent reasoning patterns
- **Fine-tuning dataset preparation**
- **Fine-tuning vs RAG trade-offs**
- **When to use hybrid approaches**
- Observability in non-deterministic systems
- A/B testing RAG configurations

**Deliverable**:
- Production RAG with 3+ retrieval strategies
- Fine-tuned model for specific domain
- Comparative analysis: Base vs RAG vs Fine-tuned vs Hybrid
- Decision framework: "When to RAG, When to Fine-tune"
- Blog: "Advanced RAG + Fine-Tuning: The Complete Picture"

**Success Metrics**:
- Can implement HyDE, query routing, reranking
- Can fine-tune models via OpenAI API
- Can articulate fine-tuning vs RAG trade-offs with data
- System shows measurable improvement over basic RAG
- Blog demonstrates mastery of both approaches

---

## Week 6: MCP Fundamentals
**Dates**: Feb 1 - Feb 7, 2026
**Goal**: Master Model Context Protocol + build your first MCP server
**Time Commitment**: 20-25 hours

### Project: "MCP Server Suite for Development"
Build 3-4 MCP servers that actually solve your problems.

**What You'll Build**:
1. **Code Context MCP**: Semantic search from Week 1 as MCP server
2. **Knowledge Base MCP**: Week 2 project as MCP server
3. **GitHub MCP**: Enhanced GitHub integration with semantic PR search
4. **Jira/Linear MCP**: Project management with AI summarization

**Tech Stack**:
- **Language**: TypeScript (MCP has better TS support)
- **MCP SDK**: @modelcontextprotocol/sdk
- **Integration**: Claude Desktop + Claude Code

**Key Learnings**:
- MCP protocol architecture
- Tool/resource/prompt design patterns
- Error handling in MCP servers
- Testing MCP servers
- Security and authentication

**Deliverable**:
- 3-4 production MCP servers
- Published to npm/PyPI
- Blog: "Building MCP Servers: A Practical Guide"
- Video demo of your MCP servers in action

**Success Metrics**:
- MCP servers work reliably in Claude Desktop
- Can explain MCP protocol design
- Servers are useful in daily workflow

---

## Week 7: Autonomous Agents
**Dates**: Feb 8 - Feb 14, 2026
**Goal**: Build agents that actually work
**Time Commitment**: 20-25 hours

### Project: "Research Agent with Memory and Tools"
Build an agent that conducts research, uses tools, and learns over time.

**What You'll Build**:
```bash
research-agent investigate "How do I optimize vector search for 10M+ documents?"
# Agent:
# 1. Searches web and documentation
# 2. Reads papers and benchmarks
# 3. Summarizes findings
# 4. Proposes implementation plan
# 5. Remembers this research for future queries
```

**Tech Stack**:
- **Agent Framework**: Build your own (ReAct pattern)
- **Tools**: Web search (Tavily), documentation RAG, code executor
- **Memory**: Vector + graph database (Neo4j for relationships)
- **Orchestration**: Custom event loop with Claude

**Key Learnings**:
- ReAct (Reasoning + Acting) pattern
- Tool calling and execution
- Multi-step reasoning
- Memory architectures (short-term, long-term, episodic)
- Agent evaluation and safety

**Deliverable**:
- Working autonomous research agent
- Comparison with LangGraph/CrewAI
- Technical write-up: "Building Reliable Agents: Lessons Learned"

**Success Metrics**:
- Agent completes multi-step research tasks
- Can explain ReAct pattern and implement from scratch
- Blog demonstrates understanding of agent architectures

---

## Week 8: Integration & Portfolio Polish
**Dates**: Feb 15 - Feb 22, 2026
**Goal**: Tie everything together + create compelling portfolio
**Time Commitment**: 30 hours

### Project: "Personal AI Engineering Platform"
Integrate all previous projects into a cohesive platform.

**What You'll Build**:
- **Unified Dashboard**: Next.js app showing all your AI tools
- **MCP Integration**: All tools accessible via MCP
- **Agent Orchestrator**: Agents can use all previous tools
- **Deployment**: Production deployment on Vercel/Railway/Fly

**Architecture**:
```
Frontend (Next.js)
    ↓
API Gateway (FastAPI/tRPC)
    ↓
├── Semantic Search Service (Week 1)
├── Knowledge Base (Week 2)
├── RAG System (Week 4-5)
├── MCP Servers (Week 6)
└── Research Agent (Week 7)
```

**Key Activities**:
1. **Integration**: Connect all services
2. **Testing**: End-to-end tests
3. **Documentation**: Comprehensive README, architecture diagrams
4. **Deployment**: Full production deployment
5. **Content Creation**:
   - Portfolio website showcasing the journey
   - "8 Weeks of AI Engineering" blog series (8 posts)
   - Video walkthrough of the platform
   - GitHub repos with excellent READMEs

**Deliverable**:
- Live platform (public demo)
- Complete blog series
- Portfolio website
- GitHub profile that tells the story
- LinkedIn posts throughout the journey

**Success Metrics**:
- All services deployed and working
- Portfolio demonstrates complete journey
- Content is shareable and impressive

---

## Success Metrics & Portfolio Story

### Technical Competencies Gained
- ✅ Deep understanding of LLM mechanics (pretraining → post-training → RL)
- ✅ Vector embeddings and similarity search
- ✅ Production vector database operations
- ✅ RAG system design and implementation (basic → advanced)
- ✅ **Fine-tuning models (when to fine-tune vs RAG)**
- ✅ MCP server development and protocol mastery
- ✅ Autonomous agent architecture (ReAct pattern)
- ✅ Full-stack AI application development
- ✅ System design for non-deterministic systems
- ✅ Prompt engineering and chain-of-thought
- ✅ Hallucination prevention strategies

### Portfolio Arc
**"From Engineering Leader to AI Practitioner: 8 Weeks of Building"**

**Week 1**: "I Learned How LLMs Work, Then Built Code Q&A"
**Week 2**: "Building a Knowledge Base: Vector DBs in Production"
**Week 3**: "What I Learned Scaling Semantic Search to 100K+ Documents"
**Week 4**: "RAG from Scratch: Lessons from Building Without Frameworks"
**Week 5**: "Advanced RAG + Fine-Tuning: The Complete Picture"
**Week 6**: "MCP Servers: Extending Claude's Capabilities"
**Week 7**: "Building Autonomous Agents That Actually Work"
**Week 8**: "Shipping a Personal AI Platform: The Complete Journey"

### Recommended Content Strategy
- **Weekly blog post**: Technical deep-dive (publish every Sunday)
- **Daily tweets/LinkedIn**: Progress updates, learnings, challenges
- **GitHub**: 8 public repos with excellent documentation
- **Video**: 1 demo video per major project
- **Final presentation**: "AI Engineering: A Director's Journey Back to Code"

---

## Tools & Resources

### Essential Reading (Start in Week 1)
- "Patterns for Building LLM-based Systems & Products" - Eugene Yan
- "What We Learned from a Year of Building with LLMs" - O'Reilly
- "LLM Patterns" - Simon Willison's blog
- Anthropic's "Claude Prompt Engineering" guide
- "Building LLM Applications for Production" - Chip Huyen

### Recommended Stack
- **Languages**: Python (AI/ML), TypeScript (MCP, tooling)
- **Vector DBs**: ChromaDB → Qdrant → Pinecone (progression)
- **LLMs**: Claude API (you're using Claude Code anyway)
- **Frameworks**: Build from scratch first, then LangChain/LlamaIndex for comparison
- **Deployment**: Railway (easiest), Fly.io (more control), or Vercel (frontend)
- **Monitoring**: LangSmith, Prometheus, Grafana

### Development Workflow with Claude Code
- Use Claude Code for scaffolding, debugging, and learning
- Ask Claude to explain concepts before implementing
- Have Claude review your code for best practices
- Use Claude to generate tests and documentation
- But: Write the core logic yourself to learn deeply

---

## Weekly Time Commitment

| Week | Hours | Focus |
|------|-------|-------|
| Week 1 (Holiday) | 40-50 | Foundations + First Ship |
| Week 2 | 20-25 | Vector Databases |
| Week 3 | 20-25 | Production Patterns |
| Week 4 | 20-25 | RAG Foundations |
| Week 5 | 20-25 | Advanced RAG |
| Week 6 | 20-25 | MCP Servers |
| Week 7 | 20-25 | Autonomous Agents |
| Week 8 (Final) | 30 | Integration & Polish |
| **Total** | **~200 hours** | **Over 8 weeks** |

---

## Accountability & Tracking

### Daily
- [ ] Log progress in week-XX/notes/daily-log.md
- [ ] Commit code to GitHub
- [ ] Post update on LinkedIn/Twitter

### Weekly
- [ ] Complete weekly project
- [ ] Write blog post
- [ ] Code review with Claude Code
- [ ] Update this plan with learnings

### Bi-weekly
- [ ] Architecture review
- [ ] Portfolio check-in
- [ ] Adjust plan if needed

### End of Program
- [ ] All 8 projects deployed
- [ ] All 8 blog posts published
- [ ] Portfolio website live
- [ ] Platform demo video complete

---

## Getting Started

### Week 1 Kickoff (Today!)

1. **Set up workspace** ✅
   ```bash
   cd /Users/ankur/D/Playground/ai-engineering-journey
   ```

2. **Read this plan thoroughly**
   - Understand the arc
   - Note questions
   - Get excited!

3. **Complete Deep Learning** ✅ DONE
   - Karpathy Deep Dive complete (1,600+ lines of notes)
   - Complete LLM understanding achieved
   - Ready for implementation

4. **Start building RAG-powered Q&A**
   - Begin code-qa-assistant project
   - Validate theory through experiments
   - Use Claude Code actively
   - Ship by end of week!

### Ready?
Let's build something amazing. Start with Week 1, Day 1 - and remember: the goal is to ship, learn, and build a portfolio that tells your story as an AI engineer.

**Your journey starts now!** 🚀

---

## Notes & Reflections
(Add your thoughts, learnings, and adjustments here as you progress)

