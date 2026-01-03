# Quick Start Guide

## Running the Pipeline Test

### Option 1: Using the Helper Script (Recommended)

```bash
cd week-01/project/rag-code-qa
./run_test.sh
```

This script will:
- Check for Python 3.12 venv
- Install dependencies automatically
- Run the full pipeline test
- Show beautiful terminal visualization

### Option 2: Manual Execution

```bash
cd week-01/project/rag-code-qa

# Activate virtual environment
source venv/bin/activate

# Verify Python version (should be 3.12.x)
python --version

# Run the test
python src/test.py

# Deactivate when done
deactivate
```

## Important Notes

### Python Version Requirements

- **Required:** Python 3.12.x (recommended)
- **Supported:** Python 3.9 - 3.12
- **Not Supported:** Python 3.14+ (ChromaDB dependency issues)

### Why Python 3.12?

ChromaDB requires `onnxruntime` which doesn't support Python 3.14 yet. The venv is configured to use Python 3.12.8.

### Common Issues

**Issue:** `python3` still uses Python 3.14
- **Solution:** After activating venv, use `python` (not `python3`)

**Issue:** ChromaDB import errors
- **Solution:** Recreate venv with Python 3.12:
  ```bash
  rm -rf venv
  python3.12 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**Issue:** Missing OpenAI API key
- **Solution:** The embedder stage will use cache. To generate new embeddings:
  1. Create `.env` file (copy from `.env.example`)
  2. Add your OpenAI API key: `OPENAI_API_KEY=sk-...`

## What You'll See

The test runs all 4 pipeline stages:

1. **Parser** → Extracts functions from source code
2. **Chunker** → Transforms into embeddable units
3. **Embedder** → Generates 1536-dim vectors (uses cache)
4. **Storage** → Persists to ChromaDB with cosine similarity

Each stage shows:
- Input/output details
- Performance metrics
- Sample data visualization
- Statistics

## Next Steps

After seeing the pipeline in action:
1. Explore the modular code in `src/`
2. Read module READMEs (parser, chunker, embedder, storage)
3. Review LLD diagrams for detailed architecture
4. Build the Retriever module (next component)
