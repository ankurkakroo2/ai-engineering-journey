"""
JavaScript/TypeScript parser using regex patterns.

LEARNING CONTEXT:
This module demonstrates regex-based parsing as an alternative to AST parsing.
Unlike Python (which has a built-in AST module), JavaScript/TypeScript would
require external libraries like tree-sitter for full AST parsing.

KEY LEARNING POINTS:
1. Regex parsing is simpler but less robust than AST parsing
2. Works well for standard patterns but may miss edge cases
3. No external dependencies required
4. Good enough for learning and prototyping

WHY REGEX FOR JS/TS:
- Simpler to implement (no external dependencies)
- Faster to prototype
- Handles common patterns well
- Can upgrade to tree-sitter later if needed

TRADE-OFFS:
- May miss complex nested functions
- Doesn't understand language grammar fully
- Brace counting can be fooled by strings/comments
- Less reliable than AST parsing

PATTERNS WE MATCH:
1. Function declarations: function myFunc() {}
2. Arrow functions: const myFunc = () => {}
3. Async variants: async function myFunc() {}
4. Export variants: export function myFunc() {}
"""

import re
import logging
from pathlib import Path
from typing import List

from .models import ParsedFunction

logger = logging.getLogger(__name__)


def parse_javascript_file(file_path: str) -> List[ParsedFunction]:
    """
    Parse a JavaScript file and extract function definitions.

    File: javascript_parser.py
    Function: parse_javascript_file

    Args:
        file_path: Path to the JavaScript file

    Returns:
        List of ParsedFunction objects, empty list if parsing fails
    """
    return _parse_js_ts_file(file_path, language="javascript")


def parse_typescript_file(file_path: str) -> List[ParsedFunction]:
    """
    Parse a TypeScript file and extract function definitions.

    TypeScript syntax is similar to JavaScript, so we can reuse the same
    regex patterns. Type annotations are ignored by our patterns.

    File: javascript_parser.py
    Function: parse_typescript_file

    Args:
        file_path: Path to the TypeScript file

    Returns:
        List of ParsedFunction objects, empty list if parsing fails
    """
    return _parse_js_ts_file(file_path, language="typescript")


def _parse_js_ts_file(file_path: str, language: str) -> List[ParsedFunction]:
    """
    Internal function to parse JavaScript/TypeScript files using regex.

    LEARNING POINT - Regex Patterns:
    We define three main patterns to match different function styles:
    1. Function declarations: function myFunc() { ... }
    2. Arrow functions: const myFunc = () => { ... }
    3. Method definitions: myFunc() { ... } (in classes)

    LEARNING POINT - Brace Counting:
    To find where a function ends, we count opening and closing braces.
    When the count returns to 0, we've found the end of the function.
    This is a simple heuristic that works for most code.

    LEARNING POINT - JSDoc Extraction:
    We look backwards from the function to find /** ... */ comments,
    which we treat as docstrings (similar to Python docstrings).

    File: javascript_parser.py
    Function: _parse_js_ts_file

    Args:
        file_path: Path to the file
        language: "javascript" or "typescript"

    Returns:
        List of ParsedFunction objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        functions = []
        lines = source.split('\n')

        # Regex patterns for different function types
        # Pattern 1: function declarations (including async, export)
        func_decl_pattern = r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\('

        # Pattern 2: arrow functions (const/let/var)
        arrow_pattern = r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'

        i = 0
        while i < len(lines):
            line = lines[i]

            # Try to match function declaration
            match = re.search(func_decl_pattern, line.strip())
            if not match:
                # Try arrow function
                match = re.search(arrow_pattern, line.strip())

            if match:
                func_name = match.group(1)
                start_line = i + 1  # Line numbers are 1-indexed

                # Find the end of the function by counting braces
                code_lines = [line]
                brace_count = line.count('{') - line.count('}')
                j = i + 1

                while j < len(lines) and brace_count > 0:
                    code_lines.append(lines[j])
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1

                end_line = j
                code = '\n'.join(code_lines)

                # Try to extract JSDoc comment as docstring
                docstring = _extract_jsdoc(lines, i)

                functions.append(ParsedFunction(
                    name=func_name,
                    code=code,
                    docstring=docstring,
                    file_path=str(Path(file_path).resolve()),
                    start_line=start_line,
                    end_line=end_line,
                    language=language
                ))

                # Move to the end of this function
                i = j
            else:
                i += 1

        logger.info(f"Parsed {len(functions)} functions from {Path(file_path).name}")
        return functions

    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return []
    except PermissionError:
        logger.warning(f"Permission denied: {file_path}")
        return []
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return []


def _extract_jsdoc(lines: List[str], func_line_idx: int) -> str | None:
    """
    Extract JSDoc comment above a function.

    JSDoc comments look like:
    /**
     * Function description
     * @param {string} name - Parameter description
     */

    We look backwards from the function line to find the closing */,
    then continue backwards to find the opening /**.

    File: javascript_parser.py
    Function: _extract_jsdoc

    Args:
        lines: All lines in the file
        func_line_idx: Index of the line where function starts (0-indexed)

    Returns:
        JSDoc comment string, or None if no JSDoc found
    """
    if func_line_idx == 0:
        return None

    # Check if previous line ends with */
    if not lines[func_line_idx - 1].strip().endswith('*/'):
        return None

    # Look backwards for /** comment
    doc_lines = []
    k = func_line_idx - 1

    while k >= 0 and not lines[k].strip().startswith('/**'):
        doc_lines.insert(0, lines[k].strip())
        k -= 1

    if k >= 0 and lines[k].strip().startswith('/**'):
        doc_lines.insert(0, lines[k].strip())
        return '\n'.join(doc_lines)

    return None
