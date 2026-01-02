"""
Parser Module - Component 1 of RAG-Code-QA Pipeline

LEARNING CONTEXT:
This is the first component in your 8-week AI Engineering Journey (Week 1).
The parser extracts functions from source code files, preparing them for
the RAG (Retrieval-Augmented Generation) pipeline.

WHY THIS MODULE EXISTS:
Without parsed functions, we can't create semantic chunks, generate embeddings,
or answer questions about code. The parser is the foundation that transforms
raw source code into structured data (ParsedFunction objects) that downstream
components can work with.

WHAT IT DOES:
1. Reads Python/JavaScript/TypeScript files
2. Extracts function definitions with metadata (name, code, docstring, location)
3. Handles edge cases (nested functions, class methods, async functions)
4. Provides graceful error handling (one bad file doesn't stop the pipeline)

KEY LEARNING OBJECTIVES:
- AST (Abstract Syntax Tree) parsing for Python
- Regex-based parsing for JavaScript/TypeScript
- Why metadata (line numbers, file paths) matters for RAG
- Graceful degradation in production data pipelines

ARCHITECTURE DECISIONS:
- Python: AST-based (reliable, handles all valid syntax)
- JS/TS: Regex-based (simpler, good enough for standard patterns)
- Metadata: Absolute paths + line numbers for citations
- Error handling: Return empty list, log warning, continue processing

USAGE:
    from parser import parse_file, parse_directory, ParsedFunction

    # Parse single file
    functions = parse_file("src/auth.py")

    # Parse directory
    all_functions = parse_directory("./src", languages=["python"])

    # Access metadata
    for func in functions:
        print(f"{func.name} at {func.file_path}:{func.start_line}")
"""

from .models import ParsedFunction
from .python_parser import parse_python_file
from .javascript_parser import parse_javascript_file, parse_typescript_file
from .dispatcher import parse_file
from .directory_walker import parse_directory

__all__ = [
    'ParsedFunction',
    'parse_python_file',
    'parse_javascript_file',
    'parse_typescript_file',
    'parse_file',
    'parse_directory',
]

__version__ = '1.0.0'
