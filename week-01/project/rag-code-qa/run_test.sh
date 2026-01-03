#!/bin/bash
# Helper script to run the pipeline test with proper environment
# Usage: ./run_test.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         RAG Pipeline Test Runner (Python 3.12)            ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if venv exists and has correct Python version
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating with Python 3.12...${NC}"
    if command -v python3.12 &> /dev/null; then
        python3.12 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created with Python 3.12${NC}"
    else
        echo -e "${RED}✗ Python 3.12 not found. Please install Python 3.12 first.${NC}"
        echo -e "${YELLOW}  ChromaDB requires Python 3.9-3.12 (not 3.14)${NC}"
        exit 1
    fi
fi

# Activate venv
source venv/bin/activate

# Check Python version in venv
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}• Using Python ${PYTHON_VERSION} from venv${NC}"

if [[ ! "$PYTHON_VERSION" =~ ^3\.12 ]]; then
    echo -e "${YELLOW}⚠ Warning: venv is using Python ${PYTHON_VERSION}, but 3.12 is recommended${NC}"
    echo -e "${YELLOW}  To fix: rm -rf venv && python3.12 -m venv venv${NC}"
fi

# Check if requirements.txt exists and install
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}• Checking dependencies...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found, installing core packages...${NC}"
    pip install -q tiktoken numpy openai python-dotenv chromadb pydantic-settings
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                  Running Pipeline Test                     ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Run the test (use 'python' not 'python3' inside venv)
python src/test.py

# Deactivate venv
deactivate

echo ""
echo -e "${GREEN}✓ Test completed. Virtual environment deactivated.${NC}"
