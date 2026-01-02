# Parser Module - Low-Level Design (LLD)

**Purpose**: Detailed technical diagram showing data flow, function calls, and implementation details

---

## Complete System Flow

This diagram shows the complete flow from user code to parsed functions, with file and method references for every component.

```mermaid
graph TB
    subgraph UserCode [User Code Entry Points]
        API1["parse_file(path)<br/>dispatcher.py"]
        API2["parse_directory(path, languages)<br/>directory_walker.py"]
    end

    subgraph DirectoryWalker [Directory Walker Module]
        DW_Start["parse_directory()<br/>directory_walker.py:65"]
        DW_Validate["Validate directory exists<br/>directory_walker.py:93-99"]
        DW_Walk["Path.rglob('*')<br/>directory_walker.py:102"]
        DW_Filter1["Skip if not file<br/>directory_walker.py:104"]
        DW_Filter2["Skip if in SKIP_DIRECTORIES<br/>directory_walker.py:107"]
        DW_Filter3["Skip if wrong extension<br/>directory_walker.py:111"]
        DW_Parse["Call parse_file()<br/>directory_walker.py:115"]
        DW_Aggregate["Aggregate results<br/>directory_walker.py:116"]
        DW_Return["Return all functions<br/>directory_walker.py:122"]
    end

    subgraph Dispatcher [File Dispatcher Module]
        DISP_Start["parse_file()<br/>dispatcher.py:31"]
        DISP_Ext["Get file extension<br/>dispatcher.py:54-55"]
        DISP_Route{"Route by extension<br/>dispatcher.py:57-64"}
    end

    subgraph PythonParser [Python Parser Module AST]
        PY_Start["parse_python_file()<br/>python_parser.py:133"]
        PY_Read["Read source file<br/>python_parser.py:147-149"]
        PY_Parse["ast.parse(source)<br/>python_parser.py:152"]
        PY_Visit["FunctionVisitor.visit(tree)<br/>python_parser.py:156"]

        subgraph Visitor [FunctionVisitor Class]
            V_Init["__init__(source, file_path)<br/>python_parser.py:49"]
            V_ClassDef["visit_ClassDef(node)<br/>python_parser.py:64"]
            V_FuncDef["visit_FunctionDef(node)<br/>python_parser.py:77"]
            V_AsyncDef["visit_AsyncFunctionDef(node)<br/>python_parser.py:87"]
            V_Process["_process_function(node)<br/>python_parser.py:97"]

            V_Name["Build function name<br/>python_parser.py:109-112"]
            V_Doc["ast.get_docstring(node)<br/>python_parser.py:115"]
            V_Lines["Extract line numbers<br/>python_parser.py:118-119"]
            V_Code["ast.get_source_segment()<br/>python_parser.py:123"]
            V_Create["Create ParsedFunction<br/>python_parser.py:130"]
        end

        PY_Return["Return visitor.functions<br/>python_parser.py:158"]
        PY_Error["Exception handling<br/>python_parser.py:160-170"]
    end

    subgraph JSParser [JavaScript Parser Module Regex]
        JS_Start["parse_javascript_file()<br/>javascript_parser.py:37"]
        TS_Start["parse_typescript_file()<br/>javascript_parser.py:50"]
        JSTS_Parse["_parse_js_ts_file()<br/>javascript_parser.py:63"]

        JSTS_Read["Read source file<br/>javascript_parser.py:99-100"]
        JSTS_Split["Split into lines<br/>javascript_parser.py:103"]
        JSTS_Patterns["Define regex patterns<br/>javascript_parser.py:106-111"]

        subgraph RegexLoop [Line-by-Line Processing]
            JSTS_Loop["while i < len(lines)<br/>javascript_parser.py:113"]
            JSTS_Match1["Match function declaration<br/>javascript_parser.py:117"]
            JSTS_Match2["Match arrow function<br/>javascript_parser.py:119"]
            JSTS_Found{"Match found?<br/>javascript_parser.py:121"}
            JSTS_Name["Extract function name<br/>javascript_parser.py:122"]
            JSTS_Braces["Count braces to find end<br/>javascript_parser.py:126-133"]
            JSTS_JSDoc["Extract JSDoc comment<br/>javascript_parser.py:138"]
            JSTS_Create["Create ParsedFunction<br/>javascript_parser.py:140"]
            JSTS_Next["Move to next function<br/>javascript_parser.py:151"]
        end

        JS_Return["Return functions list<br/>javascript_parser.py:156"]
        JS_Error["Exception handling<br/>javascript_parser.py:158-165"]
    end

    subgraph Models [Data Models]
        PF_Class["ParsedFunction dataclass<br/>models.py:23"]
        PF_Fields["name, code, docstring<br/>file_path, start_line<br/>end_line, language<br/>models.py:45-51"]
        PF_Repr["__repr__() method<br/>models.py:53"]
        PF_Str["__str__() method<br/>models.py:57"]
        PF_Location["location property<br/>models.py:61"]
        PF_HasDoc["has_docstring property<br/>models.py:66"]
    end

    %% User entry points
    API1 --> DISP_Start
    API2 --> DW_Start

    %% Directory walker flow
    DW_Start --> DW_Validate
    DW_Validate --> DW_Walk
    DW_Walk --> DW_Filter1
    DW_Filter1 --> DW_Filter2
    DW_Filter2 --> DW_Filter3
    DW_Filter3 --> DW_Parse
    DW_Parse --> DISP_Start
    DISP_Start --> DW_Aggregate
    DW_Aggregate --> DW_Return

    %% Dispatcher routing
    DISP_Start --> DISP_Ext
    DISP_Ext --> DISP_Route
    DISP_Route -->|".py"| PY_Start
    DISP_Route -->|".js"| JS_Start
    DISP_Route -->|".ts/.tsx"| TS_Start

    %% Python parser flow
    PY_Start --> PY_Read
    PY_Read --> PY_Parse
    PY_Parse --> PY_Visit
    PY_Visit --> V_Init
    V_Init --> V_ClassDef
    V_Init --> V_FuncDef
    V_Init --> V_AsyncDef
    V_ClassDef --> V_Process
    V_FuncDef --> V_Process
    V_AsyncDef --> V_Process
    V_Process --> V_Name
    V_Name --> V_Doc
    V_Doc --> V_Lines
    V_Lines --> V_Code
    V_Code --> V_Create
    V_Create --> PF_Class
    PF_Class --> PY_Return
    PY_Return --> DISP_Start
    PY_Parse -.->|error| PY_Error
    PY_Error --> DISP_Start

    %% JavaScript parser flow
    JS_Start --> JSTS_Parse
    TS_Start --> JSTS_Parse
    JSTS_Parse --> JSTS_Read
    JSTS_Read --> JSTS_Split
    JSTS_Split --> JSTS_Patterns
    JSTS_Patterns --> JSTS_Loop
    JSTS_Loop --> JSTS_Match1
    JSTS_Match1 --> JSTS_Match2
    JSTS_Match2 --> JSTS_Found
    JSTS_Found -->|yes| JSTS_Name
    JSTS_Found -->|no| JSTS_Loop
    JSTS_Name --> JSTS_Braces
    JSTS_Braces --> JSTS_JSDoc
    JSTS_JSDoc --> JSTS_Create
    JSTS_Create --> PF_Class
    JSTS_Create --> JSTS_Next
    JSTS_Next --> JSTS_Loop
    JSTS_Loop -->|done| JS_Return
    JS_Return --> DISP_Start
    JSTS_Read -.->|error| JS_Error
    JS_Error --> DISP_Start

    %% ParsedFunction model
    PF_Class --> PF_Fields
    PF_Fields --> PF_Repr
    PF_Fields --> PF_Str
    PF_Fields --> PF_Location
    PF_Fields --> PF_HasDoc

    classDef entryPoint fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef parser fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px

    class API1,API2 entryPoint
    class PY_Start,JS_Start,TS_Start,JSTS_Parse parser
    class PF_Class,PF_Fields model
    class DISP_Route,JSTS_Found decision
    class PY_Error,JS_Error error
```

---

## Python Parser Detailed Flow

This diagram focuses on the Python AST parsing implementation.

```mermaid
sequenceDiagram
    participant User
    participant ParsePython as parse_python_file<br/>python_parser.py
    participant FileSystem as File System
    participant AST as ast.parse()
    participant Visitor as FunctionVisitor<br/>python_parser.py
    participant Model as ParsedFunction<br/>models.py

    User->>ParsePython: parse_python_file(file_path)

    rect rgb(240, 248, 255)
        Note over ParsePython,FileSystem: File Reading (line 147-149)
        ParsePython->>FileSystem: open(file_path, 'r')
        FileSystem-->>ParsePython: source code string
    end

    rect rgb(255, 250, 240)
        Note over ParsePython,AST: AST Parsing (line 152)
        ParsePython->>AST: ast.parse(source, filename)
        AST-->>ParsePython: AST tree
    end

    rect rgb(240, 255, 240)
        Note over ParsePython,Visitor: Tree Walking (line 155-156)
        ParsePython->>Visitor: __init__(source, file_path)
        Visitor-->>ParsePython: visitor instance
        ParsePython->>Visitor: visit(tree)

        loop For each node in tree
            alt Node is ClassDef
                Visitor->>Visitor: visit_ClassDef(node)
                Note over Visitor: Push class name to stack<br/>(line 66)
                Visitor->>Visitor: generic_visit(node)
                Note over Visitor: Pop class name from stack<br/>(line 70)
            else Node is FunctionDef
                Visitor->>Visitor: visit_FunctionDef(node)
                Visitor->>Visitor: _process_function(node)

                rect rgb(255, 245, 245)
                    Note over Visitor: Function Processing (line 97-138)
                    Note over Visitor: 1. Build name with class prefix<br/>(line 109-112)
                    Note over Visitor: 2. Extract docstring<br/>(line 115)
                    Note over Visitor: 3. Get line numbers<br/>(line 118-119)
                    Note over Visitor: 4. Extract source code<br/>(line 123-127)
                end

                Visitor->>Model: ParsedFunction(...)
                Model-->>Visitor: function object
                Note over Visitor: Append to functions list<br/>(line 130-138)

                Visitor->>Visitor: generic_visit(node)
                Note over Visitor: Visit nested functions
            else Node is AsyncFunctionDef
                Visitor->>Visitor: visit_AsyncFunctionDef(node)
                Visitor->>Visitor: _process_function(node, is_async=True)
            end
        end

        Visitor-->>ParsePython: visitor.functions list
    end

    ParsePython-->>User: List[ParsedFunction]
```

---

## JavaScript Parser Detailed Flow

This diagram focuses on the regex-based JavaScript/TypeScript parsing.

```mermaid
sequenceDiagram
    participant User
    participant ParseJS as parse_javascript_file<br/>javascript_parser.py
    participant Internal as _parse_js_ts_file<br/>javascript_parser.py
    participant FileSystem as File System
    participant Regex as re.search()
    participant JSDoc as _extract_jsdoc<br/>javascript_parser.py
    participant Model as ParsedFunction<br/>models.py

    User->>ParseJS: parse_javascript_file(file_path)
    ParseJS->>Internal: _parse_js_ts_file(file_path, "javascript")

    rect rgb(240, 248, 255)
        Note over Internal,FileSystem: File Reading (line 99-100)
        Internal->>FileSystem: open(file_path, 'r')
        FileSystem-->>Internal: source code string
        Note over Internal: Split into lines (line 103)
    end

    rect rgb(255, 250, 240)
        Note over Internal: Define regex patterns (line 106-111)
        Note over Internal: func_decl_pattern: function myFunc()<br/>arrow_pattern: const myFunc = () =>
    end

    rect rgb(240, 255, 240)
        Note over Internal: Line-by-line processing (line 113-153)
        loop For each line (i = 0 to len(lines))
            Internal->>Regex: search(func_decl_pattern, line)
            Regex-->>Internal: match or None

            alt No match
                Internal->>Regex: search(arrow_pattern, line)
                Regex-->>Internal: match or None
            end

            alt Match found
                Note over Internal: Extract function name<br/>(line 122)
                Note over Internal: start_line = i + 1<br/>(line 123)

                rect rgb(255, 245, 245)
                    Note over Internal: Find function end (line 126-133)
                    Note over Internal: Count braces:<br/>brace_count = '{' - '}'
                    loop While brace_count > 0
                        Note over Internal: Read next line<br/>Update brace_count
                    end
                    Note over Internal: end_line = j
                end

                Internal->>JSDoc: _extract_jsdoc(lines, i)

                rect rgb(250, 250, 255)
                    Note over JSDoc: JSDoc Extraction (line 173-201)
                    Note over JSDoc: Look backwards for /**<br/>Find matching */
                    alt JSDoc found
                        JSDoc-->>Internal: docstring
                    else No JSDoc
                        JSDoc-->>Internal: None
                    end
                end

                Internal->>Model: ParsedFunction(name, code, docstring, ...)
                Model-->>Internal: function object
                Note over Internal: Append to functions list<br/>(line 140-149)
                Note over Internal: i = j (skip to end of function)<br/>(line 151)
            else No match
                Note over Internal: i += 1 (next line)<br/>(line 153)
            end
        end
    end

    Internal-->>ParseJS: List[ParsedFunction]
    ParseJS-->>User: List[ParsedFunction]
```

---

## Directory Walker Detailed Flow

This diagram shows how directory traversal and filtering works.

```mermaid
flowchart TB
    Start["parse_directory(dir_path, languages)<br/>directory_walker.py:65"]

    subgraph Initialization [Initialization Phase]
        Init1["Set default languages if None<br/>directory_walker.py:94"]
        Init2["Build extension list from languages<br/>directory_walker.py:97-101"]
        Init3["Define SKIP_DIRECTORIES set<br/>directory_walker.py:32-48"]
    end

    subgraph Validation [Validation Phase]
        Val1{"Directory exists?<br/>directory_walker.py:107"}
        Val2{"Is directory?<br/>directory_walker.py:111"}
        ValError["Log warning, return []<br/>directory_walker.py:108 or 112"]
    end

    subgraph Traversal [Traversal Phase]
        Walk["for file_path in dir_path.rglob('*')<br/>directory_walker.py:116"]

        Filter1{"Is file?<br/>directory_walker.py:118"}
        Filter2{"In skip directory?<br/>directory_walker.py:121"}
        Filter3{"Extension matches?<br/>directory_walker.py:125"}

        Parse["functions = parse_file(file_path)<br/>directory_walker.py:129"]
        Aggregate["all_functions.extend(functions)<br/>directory_walker.py:130"]

        Skip["Continue to next file"]
    end

    subgraph Results [Results Phase]
        Log["Log total functions and files<br/>directory_walker.py:133-136"]
        Return["return all_functions<br/>directory_walker.py:137"]
    end

    Start --> Init1
    Init1 --> Init2
    Init2 --> Init3
    Init3 --> Val1

    Val1 -->|No| ValError
    Val1 -->|Yes| Val2
    Val2 -->|No| ValError
    Val2 -->|Yes| Walk

    Walk --> Filter1
    Filter1 -->|No| Skip
    Filter1 -->|Yes| Filter2
    Filter2 -->|Yes| Skip
    Filter2 -->|No| Filter3
    Filter3 -->|No| Skip
    Filter3 -->|Yes| Parse
    Parse --> Aggregate
    Aggregate --> Walk

    Skip --> Walk
    Walk -->|Done| Log
    Log --> Return

    classDef init fill:#e3f2fd,stroke:#1976d2
    classDef validate fill:#fff3e0,stroke:#f57c00
    classDef process fill:#e8f5e9,stroke:#388e3c
    classDef result fill:#f3e5f5,stroke:#7b1fa2

    class Init1,Init2,Init3 init
    class Val1,Val2,ValError validate
    class Walk,Filter1,Filter2,Filter3,Parse,Aggregate,Skip process
    class Log,Return result
```

---

## Data Model Structure

This diagram shows the ParsedFunction dataclass and its properties.

```mermaid
classDiagram
    class ParsedFunction {
        +str name
        +str code
        +Optional~str~ docstring
        +str file_path
        +int start_line
        +int end_line
        +str language

        +__repr__() str
        +__str__() str
        +location() str
        +has_docstring() bool
    }

    note for ParsedFunction "models.py:23-70<br/><br/>Core data structure for parsed functions.<br/>Flows through entire RAG pipeline:<br/>Parser → Chunker → Embedder →<br/>Storage → Retriever → Generator"

    class PythonParser {
        +parse_python_file(file_path) List~ParsedFunction~
    }

    class FunctionVisitor {
        -str source
        -str file_path
        -List~ParsedFunction~ functions
        -List~str~ class_stack

        +visit_ClassDef(node)
        +visit_FunctionDef(node)
        +visit_AsyncFunctionDef(node)
        -_process_function(node, is_async)
    }

    class JavaScriptParser {
        +parse_javascript_file(file_path) List~ParsedFunction~
        +parse_typescript_file(file_path) List~ParsedFunction~
        -_parse_js_ts_file(file_path, language) List~ParsedFunction~
        -_extract_jsdoc(lines, func_line_idx) str|None
    }

    class Dispatcher {
        +parse_file(file_path) List~ParsedFunction~
    }

    class DirectoryWalker {
        +parse_directory(dir_path, languages) List~ParsedFunction~
        +should_skip_directory(dir_name) bool
    }

    PythonParser --> ParsedFunction : creates
    FunctionVisitor --> ParsedFunction : creates
    JavaScriptParser --> ParsedFunction : creates
    Dispatcher --> PythonParser : routes to
    Dispatcher --> JavaScriptParser : routes to
    DirectoryWalker --> Dispatcher : calls
    PythonParser --> FunctionVisitor : uses
```

---

## Error Handling Flow

This diagram shows how errors are handled gracefully throughout the module.

```mermaid
flowchart TB
    subgraph ParseFile [parse_file dispatcher.py]
        PF_Start["parse_file(file_path)"]
        PF_Route{"File extension?"}
        PF_Unsupported["Log debug: unsupported<br/>Return []"]
    end

    subgraph PythonParse [parse_python_file python_parser.py]
        PY_Try["try: parse Python"]
        PY_FileNotFound["except FileNotFoundError<br/>Log warning, return []"]
        PY_Syntax["except SyntaxError<br/>Log warning, return []"]
        PY_Permission["except PermissionError<br/>Log warning, return []"]
        PY_Generic["except Exception<br/>Log warning, return []"]
        PY_Success["return functions"]
    end

    subgraph JSParse [_parse_js_ts_file javascript_parser.py]
        JS_Try["try: parse JS/TS"]
        JS_FileNotFound["except FileNotFoundError<br/>Log warning, return []"]
        JS_Permission["except PermissionError<br/>Log warning, return []"]
        JS_Generic["except Exception<br/>Log warning, return []"]
        JS_Success["return functions"]
    end

    subgraph DirWalk [parse_directory directory_walker.py]
        DW_Check1{"Directory exists?"}
        DW_Check2{"Is directory?"}
        DW_Warn["Log warning, return []"]
        DW_Continue["Continue processing"]
    end

    PF_Start --> PF_Route
    PF_Route -->|".py"| PY_Try
    PF_Route -->|".js/.ts"| JS_Try
    PF_Route -->|"other"| PF_Unsupported

    PY_Try -->|FileNotFoundError| PY_FileNotFound
    PY_Try -->|SyntaxError| PY_Syntax
    PY_Try -->|PermissionError| PY_Permission
    PY_Try -->|Exception| PY_Generic
    PY_Try -->|Success| PY_Success

    JS_Try -->|FileNotFoundError| JS_FileNotFound
    JS_Try -->|PermissionError| JS_Permission
    JS_Try -->|Exception| JS_Generic
    JS_Try -->|Success| JS_Success

    DW_Check1 -->|No| DW_Warn
    DW_Check1 -->|Yes| DW_Check2
    DW_Check2 -->|No| DW_Warn
    DW_Check2 -->|Yes| DW_Continue

    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef success fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef check fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    class PF_Unsupported,PY_FileNotFound,PY_Syntax,PY_Permission,PY_Generic,JS_FileNotFound,JS_Permission,JS_Generic,DW_Warn error
    class PY_Success,JS_Success,DW_Continue success
    class PF_Route,DW_Check1,DW_Check2 check
```

---

## Key Design Patterns

### 1. Visitor Pattern (Python Parser)

**File**: `python_parser.py`
**Classes**: `FunctionVisitor` extends `ast.NodeVisitor`

The Visitor pattern allows us to walk the AST tree and process specific node types without modifying the AST structure.

### 2. Strategy Pattern (Dispatcher)

**File**: `dispatcher.py`
**Function**: `parse_file()`

The Strategy pattern selects the appropriate parsing algorithm (Python AST vs JS/TS regex) based on file extension.

### 3. Template Method Pattern (Error Handling)

**Files**: All parser files
**Pattern**: `try-except-return-empty-list`

Each parser follows the same error handling template: try to parse, catch specific exceptions, log warnings, return empty list.

### 4. Builder Pattern (ParsedFunction Creation)

**File**: `models.py`
**Class**: `ParsedFunction` (dataclass)

The dataclass provides a clean builder-like interface for creating function objects with all required metadata.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `parse_file()` | O(n) | n = file size in lines |
| `parse_directory()` | O(m × n) | m = files, n = avg file size |
| AST parsing | O(n) | Linear in source size |
| Regex parsing | O(n²) | Worst case with nested braces |
| Directory traversal | O(m) | m = total files in tree |

### Space Complexity

| Structure | Complexity | Notes |
|-----------|------------|-------|
| ParsedFunction | O(k) | k = function code size |
| Functions list | O(f × k) | f = functions, k = avg size |
| AST tree | O(n) | n = source size |
| Visitor stack | O(d) | d = nesting depth |

### Bottlenecks

1. **File I/O**: Reading large files from disk
2. **Regex matching**: Nested loops in JS/TS parser
3. **Memory**: Storing full source code for each function

### Optimizations

1. **Caching**: Could cache parsed results by file hash
2. **Streaming**: Could process files in chunks
3. **Parallel**: Could parse multiple files concurrently
4. **Tree-sitter**: Could upgrade JS/TS to O(n) parsing

---

## Summary

This LLD provides:

1. **Complete data flow** from user code to parsed functions
2. **File and method references** for every component
3. **Sequence diagrams** showing interaction patterns
4. **Error handling flows** demonstrating graceful degradation
5. **Design patterns** used throughout the module
6. **Performance characteristics** and optimization opportunities

Use this diagram to:
- Understand how components interact
- Trace execution paths for debugging
- Identify where to make changes
- Learn design patterns in practice
- Optimize performance bottlenecks

---

**Last Updated**: January 2, 2026
**Diagrams**: 6 Mermaid diagrams with full file/method references
**Purpose**: Technical reference for parser module implementation
