from setuptools import setup, find_packages

setup(
    name="code-qa",
    version="0.1.0",
    description="RAG-powered code Q&A assistant for semantic codebase search and question answering",
    author="Ankur",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "anthropic>=0.8.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tree-sitter>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "code-qa=src.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
