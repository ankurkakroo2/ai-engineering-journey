# Chunker Module - Low-Level Design

This document contains a comprehensive flow diagram illustrating the complete chunking pipeline with exact file and method references.

---

## Complete Chunking Pipeline

```mermaid
graph TB
    subgraph Input [Input Layer]
        ParsedFunc[ParsedFunction Object<br/>from parser module]
    end

    subgraph Formatting [Content Formatting - formatter.py]
        ExtractMeta[Extract metadata<br/>file_path, start_line, end_line]
        BuildHeader["Build header<br/># path:start-end"]
        CheckDoc{Has<br/>docstring?}
        AddDoc["Add docstring layer<br/>triple quotes"]
        SkipDoc[Skip docstring]
        ExtractCode[Extract code body]
        CombineLayers["Combine layers<br/>header + docstring + code"]
        FormattedText[Formatted Text String]
    end

    subgraph TokenCounting [Token Counting - token_counter.py]
        GetEncoding[get_encoding]
        CheckCache{encoding<br/>cached?}
        LoadTiktoken["Load tiktoken<br/>cl100k_base"]
        CacheEncoding[Cache encoding]
        UseCache[Use cached encoding]
        EncodeText[encoding.encode]
        CountLen["len(tokens)"]
        TokenCount[Token Count]
    end

    subgraph Validation [Size Validation]
        CheckLimit{token_count<br/>> 8000?}
    end

    subgraph Truncation [Truncation Strategy - formatter.py]
        ParseLines[Split into lines]
        FindHeader[Find header section]
        FindDocstring[Find docstring section]
        SplitSections[Split: header, docstring, code]
        PreserveImportant["Preserve header + docstring"]
        CountPreserved[Count preserved tokens]
        CalcAvailable["Calculate available tokens<br/>8000 - preserved - 50"]
        AddLines[Add code lines iteratively]
        CheckFit{Fits in<br/>limit?}
        AddLine[Add line]
        StopTruncate[Stop adding]
        MarkTruncated["Append truncation marker<br/># ... (truncated)"]
        TruncatedText[Truncated Content]
    end

    subgraph Hashing [ID Generation - hasher.py]
        CombineForHash["Combine for hash<br/>file_path:start_line:content"]
        EncodeUTF8[Encode to UTF-8 bytes]
        SHA256["hashlib.sha256"]
        Hexdigest[Convert to hex string]
        ChunkID[Deterministic Chunk ID<br/>64-char hex]
    end

    subgraph ChunkCreation [Chunk Object Creation - models.py]
        BuildMetadata["Build metadata dict<br/>name, file_path, lines, language, docstring"]
        CreateChunk["Create Chunk object<br/>id, content, metadata, token_count, truncated"]
        ValidateChunk[chunk.is_valid property]
        CheckTokensValid{token_count<br/>> 0?}
        CheckContentValid{content.strip<br/>not empty?}
    end

    subgraph Output [Output Layer]
        ReturnChunk[Return Chunk Object]
        ReturnNull[Return None]
    end

    subgraph BatchProcessing [Batch Processing - chunker.py]
        BatchInput["List[ParsedFunction]"]
        InitList[Initialize empty chunks list]
        LoopFuncs[For each function]
        CallChunkFunc[Call chunk_function]
        CheckValidChunk{chunk not<br/>None?}
        AppendChunk[Append to list]
        LogWarning["logger.warning<br/>invalid chunk"]
        MoreFuncs{More<br/>functions?}
        CalcStats[Calculate statistics]
        CountTruncated[Count truncated chunks]
        LogSuccess["logger.info<br/>success count"]
        CheckTruncWarning{truncated > 0?}
        LogTruncWarning["logger.warning<br/>truncation count"]
        BatchOutput["Return List[Chunk]"]
    end

    ParsedFunc --> ExtractMeta
    ExtractMeta --> BuildHeader
    ExtractMeta --> CheckDoc
    CheckDoc -->|Yes| AddDoc
    CheckDoc -->|No| SkipDoc
    BuildHeader --> CombineLayers
    AddDoc --> CombineLayers
    SkipDoc --> CombineLayers
    ExtractMeta --> ExtractCode
    ExtractCode --> CombineLayers
    CombineLayers --> FormattedText

    FormattedText --> GetEncoding
    GetEncoding --> CheckCache
    CheckCache -->|No| LoadTiktoken
    LoadTiktoken --> CacheEncoding
    CacheEncoding --> UseCache
    CheckCache -->|Yes| UseCache
    UseCache --> EncodeText
    EncodeText --> CountLen
    CountLen --> TokenCount

    TokenCount --> CheckLimit

    CheckLimit -->|Yes| ParseLines
    ParseLines --> FindHeader
    FindHeader --> FindDocstring
    FindDocstring --> SplitSections
    SplitSections --> PreserveImportant
    PreserveImportant --> CountPreserved
    CountPreserved --> CalcAvailable
    CalcAvailable --> AddLines
    AddLines --> CheckFit
    CheckFit -->|Yes| AddLine
    AddLine --> AddLines
    CheckFit -->|No| StopTruncate
    StopTruncate --> MarkTruncated
    MarkTruncated --> TruncatedText
    TruncatedText --> CombineForHash

    CheckLimit -->|No| CombineForHash
    FormattedText --> CombineForHash

    CombineForHash --> EncodeUTF8
    EncodeUTF8 --> SHA256
    SHA256 --> Hexdigest
    Hexdigest --> ChunkID

    ChunkID --> BuildMetadata
    FormattedText --> BuildMetadata
    TruncatedText --> BuildMetadata
    TokenCount --> BuildMetadata
    BuildMetadata --> CreateChunk
    CreateChunk --> ValidateChunk
    ValidateChunk --> CheckTokensValid
    CheckTokensValid -->|No| ReturnNull
    CheckTokensValid -->|Yes| CheckContentValid
    CheckContentValid -->|No| ReturnNull
    CheckContentValid -->|Yes| ReturnChunk

    BatchInput --> InitList
    InitList --> LoopFuncs
    LoopFuncs --> CallChunkFunc
    CallChunkFunc --> CheckValidChunk
    CheckValidChunk -->|Yes| AppendChunk
    CheckValidChunk -->|No| LogWarning
    AppendChunk --> MoreFuncs
    LogWarning --> MoreFuncs
    MoreFuncs -->|Yes| LoopFuncs
    MoreFuncs -->|No| CalcStats
    CalcStats --> CountTruncated
    CountTruncated --> LogSuccess
    LogSuccess --> CheckTruncWarning
    CheckTruncWarning -->|Yes| LogTruncWarning
    CheckTruncWarning -->|No| BatchOutput
    LogTruncWarning --> BatchOutput
```

---

## Key Components Reference

### File: `formatter.py`
- **format_function_for_embedding()**: Creates three-layer format (header + docstring + code)
- **truncate_content()**: Intelligently truncates content while preserving semantic information

### File: `token_counter.py`
- **get_encoding()**: Loads and caches tiktoken encoding (cl100k_base)
- **count_tokens()**: Accurately counts tokens using OpenAI's tokenizer

### File: `hasher.py`
- **generate_chunk_id()**: Creates deterministic SHA-256 hash from content + metadata

### File: `models.py`
- **Chunk dataclass**: Core data structure with id, content, metadata, token_count, truncated
- **is_valid property**: Validates chunk has positive token count and non-empty content

### File: `chunker.py`
- **chunk_function()**: Main orchestrator for single function transformation
- **chunk_functions()**: Batch processor with graceful degradation and statistics

---

## Design Principles

### 1. Three-Layer Content Format
```python
# /path/to/file.py:70-73          <- Location (spatial context)
"""Returns location string"""       <- Docstring (semantic meaning)
@property                           <- Code (implementation)
def location(self) -> str:
    return f"{self.file_path}:{self.start_line}"
```

### 2. Intelligent Truncation Strategy
When content exceeds 8000 tokens:
1. Preserve header (location context)
2. Preserve docstring (semantic meaning)
3. Truncate code from end (implementation details less critical)
4. Mark as truncated for monitoring

### 3. Deterministic ID Generation
```python
hash_input = f"{file_path}:{start_line}:{content}"
chunk_id = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
```

**Properties:**
- Same content → same ID (deterministic)
- Different content → different ID (avalanche effect)
- Collision probability ≈ 0 (SHA-256)

### 4. Token Counting with Caching
- tiktoken encoding loaded once and cached
- Reduces overhead for batch processing (~100ms saved per chunk)
- Accurate subword tokenization matching OpenAI's API

### 5. Graceful Degradation
- Invalid chunks filtered out (don't crash pipeline)
- Warnings logged for debugging
- Statistics tracked for quality monitoring

---

## Incremental Update Strategy

When a file is modified:

```python
# Conceptual logic (implemented in storage/embedder modules)
old_ids = set(db.get_all_ids_for_file(file_path))
new_ids = set(chunk.id for chunk in new_chunks)

deleted = old_ids - new_ids  # In old but not new → remove from DB
new = new_ids - old_ids      # In new but not old → insert to DB
unchanged = new_ids & old_ids # In both → skip (cache hit)
```

**Efficiency:**
- Unchanged functions: Same ID → skip embedding (cache hit)
- Modified functions: Different ID → re-embed (content changed)
- Deleted functions: ID not in new set → remove from database
