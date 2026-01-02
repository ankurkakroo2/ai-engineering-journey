"""
Python parser using AST (Abstract Syntax Tree).

LEARNING CONTEXT:
This module demonstrates AST-based parsing, which is the gold standard for
parsing programming languages. Python's built-in `ast` module uses the same
parser that runs your code, so if Python can execute it, we can parse it.

KEY LEARNING POINTS:
1. AST parsing is deterministic and handles ALL valid Python syntax
2. We can extract docstrings directly using ast.get_docstring()
3. Line numbers are available as node attributes (lineno, end_lineno)
4. ast.get_source_segment() gives us the exact source code
5. NodeVisitor pattern allows us to walk the AST tree systematically

WHY AST > REGEX FOR PYTHON:
- Handles complex syntax (decorators, multi-line signatures, nested functions)
- Understands Python's grammar completely
- No false positives/negatives from pattern matching
- Extracts docstrings reliably

TRADE-OFF:
AST parsing requires the language's parser (built-in for Python).
For languages without a Python AST library (like JS/TS), we use regex.
"""

import ast
import logging
from pathlib import Path
from typing import List

from .models import ParsedFunction

logger = logging.getLogger(__name__)


class FunctionVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts function definitions.

    The NodeVisitor pattern is a standard way to traverse AST trees.
    We override visit_* methods for the node types we care about.

    LEARNING POINT:
    The class_stack tracks nested classes so we can generate names like
    "MyClass.method_name" for class methods. This is important for RAG
    because it provides context about where a function lives.

    File: python_parser.py
    Methods: __init__, visit_ClassDef, visit_FunctionDef,
             visit_AsyncFunctionDef, _process_function
    """

    def __init__(self, source: str, file_path: str):
        """
        Initialize the visitor.

        Args:
            source: The source code (needed for ast.get_source_segment)
            file_path: Path to the file being parsed (for metadata)
        """
        self.source = source
        self.file_path = file_path
        self.functions: List[ParsedFunction] = []
        self.class_stack: List[str] = []  # Track nested classes

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        Visit a class definition.

        We push the class name onto the stack, visit its children (methods),
        then pop the class name. This allows us to track nesting depth.

        File: python_parser.py
        Method: FunctionVisitor.visit_ClassDef
        """
        self.class_stack.append(node.name)
        self.generic_visit(node)  # Visit children (methods)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit a regular function definition.

        File: python_parser.py
        Method: FunctionVisitor.visit_FunctionDef
        """
        self._process_function(node)
        self.generic_visit(node)  # Visit nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        Visit an async function definition.

        File: python_parser.py
        Method: FunctionVisitor.visit_AsyncFunctionDef
        """
        self._process_function(node, is_async=True)
        self.generic_visit(node)  # Visit nested functions

    def _process_function(self, node, is_async: bool = False) -> None:
        """
        Extract metadata from a function node and create ParsedFunction.

        LEARNING POINT:
        This is where we extract all the metadata needed for RAG:
        - Function name (with class prefix if it's a method)
        - Docstring (using ast.get_docstring)
        - Line numbers (from node.lineno and node.end_lineno)
        - Source code (using ast.get_source_segment)

        File: python_parser.py
        Method: FunctionVisitor._process_function

        Args:
            node: AST node (FunctionDef or AsyncFunctionDef)
            is_async: Whether this is an async function
        """
        # Build function name with class prefix if inside a class
        if self.class_stack:
            func_name = f"{'.'.join(self.class_stack)}.{node.name}"
        else:
            func_name = node.name

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Get line numbers
        start_line = node.lineno
        end_line = node.end_lineno if node.end_lineno else start_line

        # Extract source code
        try:
            code = ast.get_source_segment(self.source, node)
        except AttributeError:
            # Fallback for Python < 3.8
            lines = self.source.split('\n')
            code = '\n'.join(lines[start_line-1:end_line])

        if code:
            self.functions.append(ParsedFunction(
                name=func_name,
                code=code,
                docstring=docstring,
                file_path=str(Path(self.file_path).resolve()),
                start_line=start_line,
                end_line=end_line,
                language="python"
            ))


def parse_python_file(file_path: str) -> List[ParsedFunction]:
    """
    Parse a Python file and extract all function definitions.

    This is the main entry point for Python parsing. It reads the file,
    parses it into an AST, walks the tree to find functions, and returns
    a list of ParsedFunction objects.

    LEARNING POINT - Error Handling:
    We use graceful degradation: if a file can't be parsed, we log a warning
    and return an empty list. This ensures one bad file doesn't stop the
    entire indexing process.

    File: python_parser.py
    Function: parse_python_file

    Args:
        file_path: Path to the Python file

    Returns:
        List of ParsedFunction objects, empty list if parsing fails

    Raises:
        No exceptions are raised; errors are logged and empty list returned
    """
    try:
        # Read the source file
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse into AST
        tree = ast.parse(source, filename=file_path)

        # Visit all nodes and extract functions
        visitor = FunctionVisitor(source, file_path)
        visitor.visit(tree)

        logger.info(f"Parsed {len(visitor.functions)} functions from {Path(file_path).name}")
        return visitor.functions

    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return []
    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return []
    except PermissionError:
        logger.warning(f"Permission denied: {file_path}")
        return []
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return []
