"""
Deterministic ID Generator - Content-Based Hashing for Chunks

LEARNING CONTEXT:
This module handles Transformation 3 in the chunking pipeline: assigning
unique, deterministic identifiers to chunks based on their content.

PURPOSE:
Each chunk needs a unique ID for database storage. Using SHA-256 hashing
provides three critical properties:
1. Deterministic: Same content always produces same ID
2. Content-based: Different content produces different ID (avalanche effect)
3. Collision-resistant: Probability of collision ≈ 0

WHY THIS ENABLES INCREMENTAL UPDATES:
When a file is modified and re-indexed:
- Unchanged functions → same content → same ID → cache hit (skip embedding)
- Modified functions → different content → different ID → new embedding needed
- Deleted functions → old IDs not in new set → remove from database

This eliminates the need to track "which chunk came from which function" -
the content itself determines identity.

HASH FUNCTION PROPERTIES:
- Input: Any string (formatted chunk content)
- Output: 64-character hex string (256 bits)
- One character change in input → completely different output
- No reverse engineering (one-way function)

DEPENDENCIES:
- Input: Formatted content string from formatter
- Output: Deterministic ID string for Chunk dataclass
"""

import hashlib


def generate_chunk_id(content: str, file_path: str, start_line: int) -> str:
    hash_input = f"{file_path}:{start_line}:{content}"

    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def generate_id_from_metadata(file_path: str, function_name: str,
                               start_line: int, end_line: int) -> str:
    metadata_string = f"{file_path}:{function_name}:{start_line}:{end_line}"

    return hashlib.sha256(metadata_string.encode('utf-8')).hexdigest()
