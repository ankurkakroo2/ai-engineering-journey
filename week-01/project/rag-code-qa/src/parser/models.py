"""
Data models for the parser module.

This file defines the core data structure (ParsedFunction) that represents
a parsed function with all its metadata.

LEARNING POINT:
The ParsedFunction dataclass is designed specifically for RAG retrieval.
Each field serves a purpose in the downstream pipeline:
- name: For display and filtering
- code: The actual content to embed
- docstring: Additional semantic context
- file_path: For citations in generated answers
- start_line/end_line: For precise code location references
- language: For language-specific handling
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParsedFunction:
    """
    Represents a parsed function with its metadata.

    This is the fundamental data structure that flows through the RAG pipeline:
    Parser → Chunker → Embedder → Storage → Retriever → Generator

    Attributes:
        name: Function/method name (e.g., "MyClass.method_name" for methods)
        code: Full source code including signature and body
        docstring: Extracted docstring if present (None otherwise)
        file_path: Absolute path to source file (for citations)
        start_line: Line number where function starts (1-indexed)
        end_line: Line number where function ends (1-indexed)
        language: Source language ("python" | "javascript" | "typescript")

    Example:
        ParsedFunction(
            name="authenticate_user",
            code="def authenticate_user(username, password):\\n    ...",
            docstring="Validates user credentials against database.",
            file_path="/path/to/auth.py",
            start_line=45,
            end_line=67,
            language="python"
        )
    """
    name: str
    code: str
    docstring: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    language: str

    def __repr__(self) -> str:
        """Pretty representation for debugging."""
        return (f"ParsedFunction(name='{self.name}', "
                f"file='{Path(self.file_path).name}', "
                f"lines={self.start_line}-{self.end_line}, "
                f"language='{self.language}')")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({Path(self.file_path).name}:{self.start_line}-{self.end_line})"

    @property
    def location(self) -> str:
        """Returns a citation-friendly location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def has_docstring(self) -> bool:
        """Check if function has a docstring."""
        return self.docstring is not None and len(self.docstring.strip()) > 0
