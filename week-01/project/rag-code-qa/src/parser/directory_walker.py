"""
Directory walker - recursively traverses directories to find and parse files.

LEARNING CONTEXT:
The directory walker is the entry point for indexing entire codebases.
It handles recursive traversal, filtering, and aggregation of results.

KEY LEARNING POINTS:
1. Recursive traversal using pathlib.rglob()
2. Smart filtering to skip dependencies and build artifacts
3. Language-based filtering (only parse requested languages)
4. Aggregation of results from multiple files

WHY SKIP CERTAIN DIRECTORIES:
- node_modules/: JavaScript dependencies (thousands of files)
- __pycache__/: Python bytecode cache (not source code)
- .git/: Git metadata (not code)
- venv/, env/: Python virtual environments (dependencies)
- dist/, build/: Build artifacts (generated, not source)

These directories contain code we don't want to index because:
1. They're not part of the actual project
2. They would pollute the vector database with irrelevant code
3. They would slow down indexing significantly

File: directory_walker.py
Function: parse_directory
"""

import logging
from pathlib import Path
from typing import List, Optional

from .models import ParsedFunction
from .dispatcher import parse_file

logger = logging.getLogger(__name__)

# Directories to skip during traversal
SKIP_DIRECTORIES = {
    'node_modules',      # JavaScript/TypeScript dependencies
    '__pycache__',       # Python bytecode cache
    '.git',              # Git metadata
    'venv',              # Python virtual environment
    'env',               # Python virtual environment
    '.venv',             # Python virtual environment
    'dist',              # Build output
    'build',             # Build output
    '.pytest_cache',     # Pytest cache
    '.mypy_cache',       # MyPy type checker cache
    'coverage',          # Coverage reports
    '.tox',              # Tox testing environments
    '.next',             # Next.js build cache
    '.nuxt',             # Nuxt.js build cache
    'target',            # Rust/Java build output
    'vendor',            # Go/PHP dependencies
}

# Map language names to file extensions
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js"],
    "typescript": [".ts", ".tsx"],
}


def parse_directory(
    dir_path: str,
    languages: Optional[List[str]] = None
) -> List[ParsedFunction]:
    """
    Recursively parse all supported files in a directory.

    This is the main entry point for indexing entire codebases. It walks
    the directory tree, filters files by language, skips irrelevant directories,
    and aggregates all parsed functions.

    LEARNING POINT - Recursive Traversal:
    We use pathlib.Path.rglob('*') which recursively finds all files.
    This is simpler and more Pythonic than os.walk().

    LEARNING POINT - Filtering Strategy:
    We filter at multiple levels:
    1. Skip entire directories (node_modules, etc.)
    2. Skip non-files (directories, symlinks)
    3. Skip files with wrong extensions

    This multi-level filtering is more efficient than parsing everything
    and filtering afterwards.

    LEARNING POINT - Aggregation:
    We collect results from all files into a single list. This gives us
    a complete view of the codebase for downstream processing.

    File: directory_walker.py
    Function: parse_directory

    Args:
        dir_path: Path to the directory to parse
        languages: List of languages to parse (default: ["python", "javascript", "typescript"])

    Returns:
        List of all ParsedFunction objects found in the directory

    Example:
        # Parse only Python files
        functions = parse_directory("./src", languages=["python"])

        # Parse all supported languages
        functions = parse_directory("./src")
    """
    if languages is None:
        languages = ["python", "javascript", "typescript"]

    # Build list of extensions to search for
    extensions = []
    for lang in languages:
        if lang in LANGUAGE_EXTENSIONS:
            extensions.extend(LANGUAGE_EXTENSIONS[lang])

    all_functions = []
    dir_path_obj = Path(dir_path)

    # Validate directory exists
    if not dir_path_obj.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    if not dir_path_obj.is_dir():
        logger.warning(f"Path is not a directory: {dir_path}")
        return []

    # Walk the directory tree
    file_count = 0
    for file_path in dir_path_obj.rglob('*'):
        # Skip if it's not a file
        if not file_path.is_file():
            continue

        # Skip if in a skip directory
        if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRECTORIES):
            continue

        # Skip if extension not in our list
        if file_path.suffix.lower() not in extensions:
            continue

        # Parse the file
        file_count += 1
        functions = parse_file(str(file_path))
        all_functions.extend(functions)

    logger.info(
        f"Parsed {len(all_functions)} total functions from "
        f"{file_count} files in {dir_path}"
    )
    return all_functions


def should_skip_directory(dir_name: str) -> bool:
    """
    Check if a directory should be skipped during traversal.

    File: directory_walker.py
    Function: should_skip_directory

    Args:
        dir_name: Name of the directory

    Returns:
        True if directory should be skipped, False otherwise
    """
    return dir_name in SKIP_DIRECTORIES
