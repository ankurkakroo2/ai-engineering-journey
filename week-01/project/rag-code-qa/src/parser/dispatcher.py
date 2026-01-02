"""
File dispatcher - routes files to appropriate language-specific parsers.

LEARNING CONTEXT:
The dispatcher is a simple but important pattern in multi-language systems.
It examines the file extension and routes to the appropriate parser.

KEY LEARNING POINT:
This is the Strategy Pattern in action: we have multiple parsing strategies
(Python AST, JS/TS regex) and select the right one based on file type.

WHY THIS MATTERS FOR RAG:
By supporting multiple languages, we can index polyglot codebases (e.g.,
a project with Python backend and TypeScript frontend). The embeddings
will capture semantic similarity across languages.

File: dispatcher.py
Function: parse_file
"""

import logging
from pathlib import Path
from typing import List

from .models import ParsedFunction
from .python_parser import parse_python_file
from .javascript_parser import parse_javascript_file, parse_typescript_file

logger = logging.getLogger(__name__)


def parse_file(file_path: str) -> List[ParsedFunction]:
    """
    Parse a single file based on its extension.

    This is the main entry point for parsing individual files. It examines
    the file extension and dispatches to the appropriate language-specific
    parser.

    LEARNING POINT - Extension Mapping:
    .py  → Python parser (AST-based)
    .js  → JavaScript parser (regex-based)
    .ts  → TypeScript parser (regex-based)
    .tsx → TypeScript parser (regex-based, JSX ignored)

    LEARNING POINT - Unsupported Files:
    For unsupported extensions, we return an empty list (not an error).
    This allows the directory walker to try all files without crashing.

    File: dispatcher.py
    Function: parse_file

    Args:
        file_path: Path to the source file

    Returns:
        List of ParsedFunction objects, empty list if unsupported or parsing fails

    Example:
        functions = parse_file("src/auth.py")
        # Returns list of ParsedFunction objects from auth.py
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.py':
        return parse_python_file(file_path)
    elif suffix == '.js':
        return parse_javascript_file(file_path)
    elif suffix in ['.ts', '.tsx']:
        return parse_typescript_file(file_path)
    else:
        logger.debug(f"Unsupported file type: {suffix} for {path.name}")
        return []
