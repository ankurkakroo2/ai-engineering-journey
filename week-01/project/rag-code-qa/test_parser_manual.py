#!/usr/bin/env python3
"""
Manual test script for parser.py

This script tests the parser by parsing the rag-code-qa codebase itself.
It's a great way to validate that our parser works on real code.

Usage:
    python test_parser_manual.py
"""

import sys
from pathlib import Path

# Add src to path so we can import parser
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parser import parse_file, parse_directory, ParsedFunction


def test_single_file():
    """Test parsing a single Python file (models.py from parser module)."""
    print("\n" + "="*70)
    print("TEST 1: Parsing parser/models.py")
    print("="*70)

    models_path = Path(__file__).parent / "src" / "parser" / "models.py"
    functions = parse_file(str(models_path))

    print(f"\nFound {len(functions)} functions:\n")
    for func in functions:
        print(f"  • {func.name}")
        print(f"    Lines: {func.start_line}-{func.end_line}")
        print(f"    Has docstring: {'Yes' if func.docstring else 'No'}")
        print()

    return functions


def test_directory():
    """Test parsing the parser module directory."""
    print("\n" + "="*70)
    print("TEST 2: Parsing entire src/parser/ directory")
    print("="*70)

    parser_module_path = Path(__file__).parent / "src" / "parser"
    functions = parse_directory(str(parser_module_path), languages=["python"])

    print(f"\nFound {len(functions)} total functions\n")

    # Group by file
    by_file = {}
    for func in functions:
        file_name = Path(func.file_path).name
        if file_name not in by_file:
            by_file[file_name] = []
        by_file[file_name].append(func)

    for file_name, funcs in by_file.items():
        print(f"\n{file_name}: {len(funcs)} functions")
        for func in funcs:
            print(f"  • {func.name} (lines {func.start_line}-{func.end_line})")

    return functions


def test_function_details():
    """Test detailed extraction of a specific function."""
    print("\n" + "="*70)
    print("TEST 3: Detailed function inspection")
    print("="*70)

    python_parser_path = Path(__file__).parent / "src" / "parser" / "python_parser.py"
    functions = parse_file(str(python_parser_path))

    # Find the parse_python_file function
    target_func = None
    for func in functions:
        if func.name == "parse_python_file":
            target_func = func
            break

    if target_func:
        print(f"\nFunction: {target_func.name}")
        print(f"File: {Path(target_func.file_path).name}")
        print(f"Lines: {target_func.start_line}-{target_func.end_line}")
        print(f"Language: {target_func.language}")
        print(f"\nDocstring:")
        if target_func.docstring:
            print(target_func.docstring[:200] + "..." if len(target_func.docstring) > 200 else target_func.docstring)
        else:
            print("  (none)")
        print(f"\nCode preview (first 300 chars):")
        print(target_func.code[:300] + "..." if len(target_func.code) > 300 else target_func.code)
    else:
        print("Could not find target function")

    return target_func


def test_edge_cases():
    """Test edge cases like nested functions and class methods."""
    print("\n" + "="*70)
    print("TEST 4: Edge cases (nested functions, class methods)")
    print("="*70)

    # Create a temporary test file with edge cases
    test_code = '''
class MyClass:
    """A test class."""

    def method_one(self):
        """A class method."""
        pass

    async def async_method(self):
        """An async method."""
        pass

def outer_function():
    """Outer function."""
    def inner_function():
        """Nested function."""
        pass
    return inner_function

async def async_function():
    """An async function."""
    pass
'''

    test_file = Path(__file__).parent / "test_edge_cases_temp.py"
    test_file.write_text(test_code)

    try:
        functions = parse_file(str(test_file))
        print(f"\nFound {len(functions)} functions in test file:\n")
        for func in functions:
            print(f"  • {func.name} (lines {func.start_line}-{func.end_line})")
            if func.docstring:
                print(f"    Docstring: {func.docstring.strip()}")

        # Check for expected functions
        expected = ["MyClass.method_one", "MyClass.async_method", "outer_function", "inner_function", "async_function"]
        found_names = [f.name for f in functions]

        print(f"\nExpected functions: {expected}")
        print(f"Found functions: {found_names}")

        missing = set(expected) - set(found_names)
        if missing:
            print(f"\n⚠️  Missing: {missing}")
        else:
            print(f"\n✓ All expected functions found!")

        return functions
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PARSER MANUAL TEST SUITE")
    print("="*70)
    print("\nThis script tests the modular parser package by parsing real Python code.")
    print("We'll parse the parser module itself as validation.")

    try:
        # Run tests
        test_single_file()
        test_directory()
        test_function_details()
        test_edge_cases()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        print("\n✓ Parser is working! Review the output above to verify:")
        print("  1. Functions are extracted with correct names")
        print("  2. Line numbers are accurate")
        print("  3. Docstrings are preserved")
        print("  4. Nested functions and class methods are handled")
        print("\nNext step: Implement chunker.py to convert these ParsedFunction")
        print("objects into chunks suitable for embedding.")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
