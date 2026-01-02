"""
Content Formatter - Transforms ParsedFunction into Embeddable Text

LEARNING CONTEXT:
This module handles Transformation 1 in the chunking pipeline: converting
structured ParsedFunction objects into formatted text strings that optimize
embedding quality.

PURPOSE:
The embedding model only accepts text strings, not structured objects.
This formatter serializes ParsedFunction into a three-layer format:
1. Location header (file path + line numbers for spatial context)
2. Docstring (natural language description for semantic matching)
3. Code implementation (actual function body)

WHY THIS FORMAT:
- Layer 1 helps queries like "functions in models.py" or "code around line 70"
- Layer 2 matches user queries (they ask in English, not code syntax)
- Layer 3 captures technical details and implementation patterns

The embedding model learns to map this format to vectors where semantic
similarity = geometric proximity in high-dimensional space.

DEPENDENCIES:
- Input: ParsedFunction from parser module
- Output: Formatted string ready for token counting and embedding
"""

from typing import Optional
from ..parser.models import ParsedFunction


def format_function_for_embedding(func: ParsedFunction) -> str:
    parts = []

    parts.append(f"# {func.file_path}:{func.start_line}-{func.end_line}")

    if func.docstring:
        parts.append(f'"""{func.docstring}"""')

    parts.append(func.code)

    return "\n\n".join(parts)


def truncate_content(content: str, max_tokens: int, current_tokens: int,
                     token_counter_func) -> tuple[str, bool]:
    if current_tokens <= max_tokens:
        return content, False

    lines = content.split("\n")

    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            header_end = i + 1
        else:
            break

    docstring_end = header_end
    in_docstring = False
    for i in range(header_end, len(lines)):
        if '"""' in lines[i]:
            if not in_docstring:
                in_docstring = True
            else:
                docstring_end = i + 1
                break

    header = "\n".join(lines[:header_end])
    docstring = "\n".join(lines[header_end:docstring_end]) if docstring_end > header_end else ""
    code_lines = lines[docstring_end:]

    preserved = header
    if docstring:
        preserved += "\n" + docstring

    preserved_tokens = token_counter_func(preserved)
    available_tokens = max_tokens - preserved_tokens - 50

    truncated_code = []
    for line in code_lines:
        test_content = preserved + "\n" + "\n".join(truncated_code + [line])
        if token_counter_func(test_content) <= max_tokens:
            truncated_code.append(line)
        else:
            break

    if truncated_code:
        final_content = preserved + "\n" + "\n".join(truncated_code)
    else:
        final_content = preserved

    final_content += "\n\n# ... (truncated)"

    return final_content, True
