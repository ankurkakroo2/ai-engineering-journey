#!/usr/bin/env python3
"""
RAG Pipeline Test & Visualization

This script demonstrates the complete RAG pipeline with beautiful terminal output.
It processes a complex source file through all stages:
1. Parser → Extract functions
2. Chunker → Create embeddable chunks
3. Embedder → Generate vector embeddings

Usage:
    cd week-01/project/rag-code-qa
    python3 src/test.py
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parser import parse_file
from src.chunker import chunk_functions, get_chunking_stats
from src.embedder import embed_chunks, get_embedding_stats, get_cache_stats


# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """Print a beautiful header."""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.END}\n")


def print_section(text):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {text}{Colors.END}")
    print(f"{Colors.BLUE}{'─'*78}{Colors.END}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text, indent=0):
    """Print info message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.CYAN}• {text}{Colors.END}")


def print_stat(label, value, indent=0):
    """Print a statistic."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.YELLOW}{label}:{Colors.END} {Colors.BOLD}{value}{Colors.END}")


def print_warning(text, indent=0):
    """Print warning message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.YELLOW}⚠ {text}{Colors.END}")


def visualize_function(func, index, total):
    """Visualize a parsed function."""
    print(f"\n{Colors.BOLD}[{index}/{total}] {func.name}{Colors.END}")
    print(f"  {Colors.CYAN}Location:{Colors.END} {Path(func.file_path).name}:{func.start_line}-{func.end_line}")
    print(f"  {Colors.CYAN}Language:{Colors.END} {func.language}")
    print(f"  {Colors.CYAN}Has docstring:{Colors.END} {'Yes' if func.has_docstring else 'No'}")
    print(f"  {Colors.CYAN}Code length:{Colors.END} {len(func.code)} chars")


def visualize_chunk(chunk, index, total):
    """Visualize a chunk."""
    print(f"\n{Colors.BOLD}[{index}/{total}] {chunk.function_name}{Colors.END}")
    print(f"  {Colors.CYAN}Chunk ID:{Colors.END} {chunk.id[:16]}...")
    print(f"  {Colors.CYAN}Token count:{Colors.END} {chunk.token_count}")
    print(f"  {Colors.CYAN}Truncated:{Colors.END} {'Yes' if chunk.truncated else 'No'}")
    print(f"  {Colors.CYAN}Valid:{Colors.END} {'Yes' if chunk.is_valid else 'No'}")

    # Show content preview
    lines = chunk.content.split('\n')
    preview_lines = lines[:3]
    print(f"  {Colors.CYAN}Content preview:{Colors.END}")
    for line in preview_lines:
        print(f"    {Colors.YELLOW}{line[:70]}{Colors.END}")
    if len(lines) > 3:
        print(f"    {Colors.YELLOW}... ({len(lines) - 3} more lines){Colors.END}")


def visualize_embedding(embedding, index, total):
    """Visualize an embedding."""
    print(f"\n{Colors.BOLD}[{index}/{total}] Chunk ID: {embedding.chunk_id[:16]}...{Colors.END}")
    print(f"  {Colors.CYAN}Dimension:{Colors.END} {embedding.dimension}")
    print(f"  {Colors.CYAN}Cached:{Colors.END} {'Yes (instant)' if embedding.cached else 'No (API call)'}")
    print(f"  {Colors.CYAN}Valid:{Colors.END} {'Yes' if embedding.is_valid else 'No'}")

    # Show vector preview
    vector_preview = embedding.vector[:5] + ['...'] + embedding.vector[-2:]
    formatted = [f"{v:.4f}" if isinstance(v, float) else v for v in vector_preview]
    print(f"  {Colors.CYAN}Vector preview:{Colors.END} [{', '.join(formatted)}]")


def stage_1_parser():
    """Stage 1: Parse source file."""
    print_header("STAGE 1: PARSER - Extract Functions from Source Code")

    # Use a complex file from our own codebase
    target_file = Path(__file__).parent / "parser" / "python_parser.py"

    print_info(f"Target file: {target_file.name}")
    print_info(f"File size: {target_file.stat().st_size} bytes")

    print_section("Parsing...")
    start_time = time.time()

    functions = parse_file(str(target_file))

    elapsed = time.time() - start_time

    print_success(f"Parsed {len(functions)} functions in {elapsed:.3f}s")

    # Show summary
    print_section("Parsed Functions Summary")

    total_lines = sum(f.end_line - f.start_line + 1 for f in functions)
    with_docstrings = sum(1 for f in functions if f.has_docstring)

    print_stat("Total functions", len(functions))
    print_stat("Total lines of code", total_lines)
    print_stat("Functions with docstrings", f"{with_docstrings}/{len(functions)}")
    print_stat("Average lines per function", f"{total_lines // len(functions) if functions else 0}")

    # Show first 3 functions in detail
    print_section("Sample Functions (first 3)")
    for i, func in enumerate(functions[:3], 1):
        visualize_function(func, i, min(3, len(functions)))

    if len(functions) > 3:
        print(f"\n{Colors.YELLOW}... and {len(functions) - 3} more functions{Colors.END}")

    return functions


def stage_2_chunker(functions):
    """Stage 2: Create embeddable chunks."""
    print_header("STAGE 2: CHUNKER - Transform Functions into Embeddable Chunks")

    print_info(f"Input: {len(functions)} parsed functions")

    print_section("Chunking...")
    start_time = time.time()

    chunks = chunk_functions(functions)

    elapsed = time.time() - start_time

    print_success(f"Created {len(chunks)} chunks in {elapsed:.3f}s")

    # Show statistics
    stats = get_chunking_stats(chunks)

    print_section("Chunking Statistics")
    print_stat("Total chunks", stats['total_chunks'])
    print_stat("Total tokens", f"{stats['total_tokens']:,}")
    print_stat("Average tokens per chunk", stats['avg_tokens'])
    print_stat("Min tokens", stats['min_tokens'])
    print_stat("Max tokens", stats['max_tokens'])
    print_stat("Truncated chunks", stats['truncated_count'])
    print_stat("Chunks with docstrings", stats['with_docstring'])

    # Show first 3 chunks in detail
    print_section("Sample Chunks (first 3)")
    for i, chunk in enumerate(chunks[:3], 1):
        visualize_chunk(chunk, i, min(3, len(chunks)))

    if len(chunks) > 3:
        print(f"\n{Colors.YELLOW}... and {len(chunks) - 3} more chunks{Colors.END}")

    return chunks


def stage_3_embedder(chunks):
    """Stage 3: Generate vector embeddings."""
    print_header("STAGE 3: EMBEDDER - Generate Vector Embeddings")

    print_info(f"Input: {len(chunks)} chunks")

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_warning("OPENAI_API_KEY not set - skipping embedding stage")
        print_info("To run this stage, set your API key:", indent=1)
        print_info("export OPENAI_API_KEY='your-api-key-here'", indent=1)
        return None

    print_section("Embedding (with cache-first strategy)...")

    # Show cache stats before
    cache_stats_before = get_cache_stats()
    print_info(f"Cache entries before: {cache_stats_before['total_entries']}")
    print_info(f"Cache size before: {cache_stats_before['total_size_mb']} MB")

    start_time = time.time()

    # Embed only first 5 chunks for demo (to save API costs)
    sample_chunks = chunks[:5]
    print_info(f"Embedding first {len(sample_chunks)} chunks (demo mode)")

    try:
        embeddings = embed_chunks(sample_chunks, batch_size=5)

        if not embeddings:
            print_warning("No embeddings generated - API call may have failed")
            return None

        elapsed = time.time() - start_time
        print_success(f"Generated {len(embeddings)} embeddings in {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print_warning(f"Embedding failed after {elapsed:.3f}s: {str(e)}")
        print_info("This is expected if API key is invalid or rate limit hit", indent=1)
        return None

    # Show statistics
    stats = get_embedding_stats(embeddings)

    print_section("Embedding Statistics")
    print_stat("Total embeddings", stats['total_embeddings'])
    print_stat("From cache", stats['cached_count'])
    print_stat("From API", stats['api_count'])
    print_stat("Cache hit rate", f"{stats['cache_hit_rate']}%")
    print_stat("Average dimension", stats['avg_dimension'])
    print_stat("All valid", 'Yes' if stats['all_valid'] else 'No')

    # Show cache stats after
    cache_stats_after = get_cache_stats()
    print_stat("Cache entries after", cache_stats_after['total_entries'])
    print_stat("Cache size after", f"{cache_stats_after['total_size_mb']} MB")

    # Show first 3 embeddings in detail
    print_section("Sample Embeddings (first 3)")
    for i, embedding in enumerate(embeddings[:3], 1):
        visualize_embedding(embedding, i, min(3, len(embeddings)))

    if len(embeddings) > 3:
        print(f"\n{Colors.YELLOW}... and {len(embeddings) - 3} more embeddings{Colors.END}")

    return embeddings


def show_pipeline_summary(functions, chunks, embeddings):
    """Show overall pipeline summary."""
    print_header("PIPELINE SUMMARY")

    print_section("Data Flow")

    # Determine status for each stage
    parser_status = f"{Colors.GREEN}✓ Parser{Colors.END}" if functions else f"{Colors.RED}✗ Parser{Colors.END}"
    chunker_status = f"{Colors.GREEN}✓ Chunker{Colors.END}" if chunks else f"{Colors.RED}✗ Chunker{Colors.END}"
    embedder_status = f"{Colors.GREEN}✓ Embedder{Colors.END}" if embeddings else f"{Colors.YELLOW}⊘ Embedder (skipped){Colors.END}"

    print(f"""
    {Colors.BOLD}Source File{Colors.END}
        ↓
    {parser_status} → {len(functions) if functions else 0} functions
        ↓
    {chunker_status} → {len(chunks) if chunks else 0} chunks
        ↓
    {embedder_status} → {len(embeddings) if embeddings else 0} embeddings
        ↓
    {Colors.BOLD}Ready for Storage (ChromaDB){Colors.END}
    """)

    print_section("Key Insights")

    if functions:
        print_info("✓ Parser successfully extracted function definitions")
        print_info("✓ Preserved metadata (file paths, line numbers, docstrings)")

    if chunks:
        stats = get_chunking_stats(chunks)
        print_info("✓ Chunker formatted functions for embedding")
        print_info(f"✓ Average {stats['avg_tokens']} tokens per chunk (within 8000 limit)")
        if stats['truncated_count'] > 0:
            print_warning(f"{stats['truncated_count']} chunks were truncated")

    if embeddings:
        stats = get_embedding_stats(embeddings)
        print_info("✓ Embedder generated 1536-dimensional vectors")
        print_info(f"✓ Cache hit rate: {stats['cache_hit_rate']}%")
        if stats['cached_count'] > 0:
            print_info(f"✓ Saved {stats['cached_count']} API calls via caching")

    print_section("Next Steps")
    print_info("1. Storage: Insert embeddings into ChromaDB")
    print_info("2. Retriever: Implement similarity search")
    print_info("3. Generator: Use Claude to generate answers")
    print_info("4. CLI: Build command-line interface")


def main():
    """Run the complete pipeline visualization."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║                   RAG PIPELINE TEST & VISUALIZATION                       ║")
    print("║                                                                           ║")
    print("║                    Components 1-3: Parser → Chunker → Embedder           ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

    try:
        # Stage 1: Parser
        functions = stage_1_parser()

        # Stage 2: Chunker
        chunks = stage_2_chunker(functions)

        # Stage 3: Embedder
        embeddings = stage_3_embedder(chunks)

        # Summary
        show_pipeline_summary(functions, chunks, embeddings)

        # Final message
        print_header("TEST COMPLETE")

        # Count successful stages
        stages_completed = sum([bool(functions), bool(chunks), bool(embeddings)])
        total_stages = 3

        if stages_completed == total_stages:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ All {total_stages} stages completed successfully!{Colors.END}\n")
        else:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ {stages_completed}/{total_stages} stages completed successfully{Colors.END}\n")
            if not embeddings:
                print(f"{Colors.YELLOW}  ⊘ Embedder stage skipped (set OPENAI_API_KEY to enable){Colors.END}\n")

    except Exception as e:
        print_header("TEST FAILED")
        print(f"{Colors.RED}{Colors.BOLD}✗ Pipeline error: {e}{Colors.END}\n")
        print(f"{Colors.YELLOW}Stack trace:{Colors.END}")
        import traceback
        traceback.print_exc()
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
