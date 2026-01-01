# Experiment 1: Embeddings Validation

## Overview

This experiment validates 7 core embeddings concepts from Day 1-3 learning. Each test file corresponds to one concept from the learning notes.

## Prerequisites

1. OpenAI API key (for text-embedding-3-small)
2. Python 3.9+
3. Environment variables set up

## Setup

```bash
# Navigate to experiment folder
cd week-01/project/experiments/01_embeddings

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../../rag-code-qa/config/.env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Running Tests

### Run All Tests (Recommended)
```bash
python run_all.py
```

This runs all 7 tests in sequence and outputs results to stdout and results.md.

### Run Individual Tests
```bash
python test_01_pretraining.py
python test_02_semantic_clustering.py
python test_03_dimensionality.py
python test_04_distance_metrics.py
python test_05_relationships.py
python test_06_chunking.py
python test_07_working_memory.py
```

## Test Files

| File | Validates | Learning Reference |
|------|-----------|-------------------|
| test_01_pretraining.py | Embeddings are pre-computed lookups | Day 1, lines 20-33 |
| test_02_semantic_clustering.py | Similar code clusters together | Day 1, lines 36-65 |
| test_03_dimensionality.py | 1536 dims vs 384 dims trade-offs | Day 1, lines 69-87 |
| test_04_distance_metrics.py | Cosine vs Euclidean for code | Day 1, lines 12-15 |
| test_05_relationships.py | Language-specific clustering | Day 1, lines 227-234 |
| test_06_chunking.py | Function-level vs fixed-size chunks | Day 2-3, lines 666-668, 735-736 |
| test_07_working_memory.py | Retrieval quality for RAG | Day 2-3, lines 691-704, 611-615 |

## Results

Results are documented in `results.md` with:
- Test observations
- Theory validation
- Connections to RAG-Powered Code Q&A project
- Synthesis and actionable decisions

## Next Steps

After running experiments:
1. Review results.md
2. Extract decisions for rag-code-qa/ (chunking strategy, distance metric, top-k)
3. Move to Experiment 2: ChromaDB Retrieval
