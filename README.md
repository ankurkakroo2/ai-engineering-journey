# AI Engineering Journey

## Quick Links
- 📋 [**Master Plan**](PLAN.md) - Complete 8-week curriculum
- 📊 [**Progress Tracker**](PROGRESS.md) - Track your journey
- 📚 [**Resources**](resources/README.md) - Curated learning materials

## Journey Overview

```
Week 1: Foundations & Semantic Search          [████░░░░] 45%
Week 2: Vector Databases Deep Dive              [________] 0%
Week 3: Production Vector Search                [________] 0%
Week 4: RAG Systems - Foundations               [________] 0%
Week 5: Advanced RAG Patterns                   [________] 0%
Week 6: MCP Fundamentals                        [________] 0%
Week 7: Autonomous Agents                       [________] 0%
Week 8: Integration & Portfolio                 [________] 0%
```

## Projects You'll Build

1. **Semantic Code Search** - Week 1
2. **Personal Knowledge Base** - Week 2
3. **Multi-Tenant Search SaaS** - Week 3
4. **Technical Documentation Assistant** - Week 4
5. **Adaptive RAG System** - Week 5
6. **MCP Server Suite** - Week 6
7. **Research Agent** - Week 7
8. **Personal AI Platform** - Week 8

## Folder Structure

```
ai-engineering-journey/
├── PLAN.md                    # Master plan
├── PROGRESS.md                # Progress tracking
├── README.md                  # This file
├── week-01/                   # Week 1: Semantic Search
│   ├── project/              # Code and deliverables
│   ├── notes/                # Daily logs and learnings
│   └── resources/            # Week-specific resources
├── week-02/                   # Week 2: Vector Databases
├── week-03/                   # Week 3: Production Patterns
├── week-04/                   # Week 4: RAG Foundations
├── week-05/                   # Week 5: Advanced RAG
├── week-06/                   # Week 6: MCP Servers
├── week-07/                   # Week 7: Autonomous Agents
├── week-08/                   # Week 8: Integration
├── portfolio/                 # Portfolio artifacts
│   ├── blog-posts/           # Weekly blog posts
│   ├── videos/               # Demo videos
│   └── demos/                # Live demos
└── resources/                 # Global resources
    ├── reading/              # Reading materials
    ├── tools/                # Tool configurations
    └── templates/            # Code templates
```

## Current Week: Week 1 - In Progress (45% Complete)

### ✅ Completed (Days 1-4)
**Learning Phase (30+ hours)**
- Day 1: Embeddings fundamentals - RNNs, LSTMs, Transformers (5+ hours)
- Day 2-3: Deep dive into LLM architecture, training, and alignment (8+ hours)
- Day 3: Finalized mental models and learning documentation

**Planning Phase (In Progress)**
- Day 4: Created Experiment 1 with 7 validation tests
  - SPEC.md and README.md with clear methodology
  - 7 test files (pre-training, semantic clustering, dimensionality, distance metrics, relationships, chunking, working memory)
  - day4readingnotes.md with detailed learning objectives for each test
  - Infrastructure: run_all.py, results.md template, requirements.txt

### 🔄 In Progress (Day 5-7)
- Day 5: Run 7 validation tests and document findings
- Day 6: Build rag-code-qa project with validated architecture
- Day 7: Finalize, polish, and ship

### Learning Documentation
- [day1-reading-notes.md](week-01/notes/day1-reading-notes.md) - Embeddings fundamentals
- [Date2-3-Deep-Dive-Notes.md](week-01/notes/Date2-3-Deep-Dive-Notes.md) - LLM architecture & training
- [day4readingnotes.md](week-01/notes/day4readingnotes.md) - Experiment learning objectives
- [daily-log.md](week-01/notes/daily-log.md) - Day-by-day progress

### Architecture Decisions (Informed by Experiments)
- **Embedding Model**: text-embedding-3-small (1536 dimensions)
- **Distance Metric**: Cosine similarity (optimal for embeddings)
- **Chunking Strategy**: By function (semantic units > fixed-size)
- **Multi-language Support**: Python, JavaScript, TypeScript
- **Caching**: Safe and necessary (embeddings are pre-computed)

### Next: Run Experiments
```bash
cd week-01/project/experiments/01_embeddings
python run_all.py
```

## Tech Stack

### Languages
- **Python** - AI/ML, data processing
- **TypeScript** - MCP servers, tooling

### Vector Databases
- ChromaDB (Week 1)
- Qdrant (Week 2)
- Pinecone (Week 3)
- pgvector (Week 2)

### LLMs & APIs
- Claude API (primary)
- OpenAI (embeddings, GPT-4)
- Cohere (reranking)

### Frameworks
- Click (CLI)
- FastAPI (APIs)
- Next.js (Frontend)
- LangChain (comparative study)

## Daily Workflow

1. **Morning**: Read and plan (30-60 min)
2. **Code**: Build and ship (2-4 hours)
3. **Document**: Write notes (15-30 min)
4. **Share**: Post on LinkedIn/Twitter (5 min)

## Content Strategy

- **Daily**: Social media updates
- **Weekly**: Technical blog post
- **Bi-weekly**: Video demo
- **End**: Portfolio website

## Success Criteria

- ✅ 8 projects shipped and deployed
- ✅ 8 blog posts published
- ✅ Portfolio website live
- ✅ Deep understanding of AI engineering

## Let's Build! 🚀

Start with [PLAN.md](PLAN.md) and begin your journey.
