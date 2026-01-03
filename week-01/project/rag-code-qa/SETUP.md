# Environment Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your key from: https://platform.openai.com/api-keys
```

Your `.env` file should look like:

```bash
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Verify Setup

```bash
# Run the pipeline test
python3 src/test.py

# Or use the helper script
./run_test.sh
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model to use |
| `OPENAI_MAX_RETRIES` | No | `3` | Max retries for API calls |
| `OPENAI_TIMEOUT` | No | `30` | API timeout in seconds |

## Security Notes

✅ **`.env` is gitignored** - Your API keys will never be committed  
✅ **`.env.example` is safe to commit** - Contains no secrets  
✅ **Never commit API keys** - Always use environment variables

## Troubleshooting

### "OPENAI_API_KEY not set"

Make sure:
1. You created the `.env` file (not just `.env.example`)
2. The file contains `OPENAI_API_KEY=sk-...`
3. You're running from the project root directory

### "Module not found: dotenv"

```bash
pip install python-dotenv
```

### API Key Not Loading

```bash
# Test if .env is being loaded
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

## Alternative: Export in Shell

Instead of using `.env`, you can export directly:

```bash
export OPENAI_API_KEY=sk-your-key-here
python3 src/test.py
```

But using `.env` is recommended for persistence across sessions.
