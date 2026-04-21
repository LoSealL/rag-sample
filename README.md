# sec-rag

Local RAG system for code and documents with multi-project indexing, code desensitization, and security scanning.

Index Python/C/C++/SystemC/Verilog codebases plus Markdown/Word docs across multiple projects, then query with natural language — only the most relevant chunks go to the LLM, cutting token costs. Sensitive information is automatically detected and blocked.

## Quick start

```bash
# Install
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your localhost model gateway config

# Index a single project
sec-rag-index --path ./src --languages python,c,cpp

# Index multiple projects at once (each gets its own summary)
sec-rag-index --path ./frontend --path ./backend --languages python,cpp,verilog

# Ask questions
sec-rag-query "how does authentication work?"

# View stats
sec-rag-stats
```

## Features

### Multi-Project Indexing
- Index multiple project directories in one pass: `--path proj1 --path proj2 --path proj3`
- Each project gets a dedicated summary used as a retrieval entry point
- Two-stage retrieval: project summary → function/module level
- Metadata filtering by project, language, document kind

### Code Desensitization & Security
- **Automatic desensitization** of C/C++ and SystemC code:
  - Extracts function declaration only (removes implementation)
  - Generates LLM-based behavioral summary (inputs, outputs, state transitions)
  - Creates a 9-line sanitized document combining metadata + declaration + summary
- **Sensitive info detection** (post-desensitization):
  - Blocks passwords, email addresses, IP addresses, phone numbers, URLs with credentials
  - Hard-fail policy: raises `SensitiveInfoError` if any sensitive pattern detected
  - Applies to all indexed documents including project summaries

### Language Support
- **Python**: AST-based chunking (functions, classes)
- **C/C++**: tree-sitter AST parsing, desensitized → LLM summary
- **SystemC/TLM**: Auto-detected from includes (`#include <systemc.h>`, `SC_MODULE`), uses C++ parser with special tagging
- **Verilog/SystemVerilog**: Regex-based module/port/always block extraction
- **Documents**: Markdown, Word (via `unstructured.io`)

### Localhost Model Flexibility
- **LLM**: Any OpenAI-compatible or Anthropic-compatible API (auto-detected)
  - Default: `http://localhost:8000` (qwen3.5-8b-instruct-awq)
- **Embeddings**: Any compatible embedding API
  - Default: `http://localhost:8000` (qwen3-vl-embedding-8b)
- Override with `--api-base-url`, `--llm-provider`, `--llm-model`

## Installation dependencies

System packages required:
```bash
sudo apt install libmagic-dev poppler-utils
```

Python dependencies (installed by `pip install -e .`):
- `chromadb` (vector store)
- `tree-sitter`, `tree-sitter-python`, `tree-sitter-c`, `tree-sitter-cpp` (AST parsing)
- `loguru` (logging)
- `click` (CLI)
- `dotenv` (environment config)
- `tiktoken` (token counting)

## Architecture

```
Multi-Project Indexing
│
├─ Project 1
│  ├─ Python files → CodeChunker → annotate + LLM summary
│  ├─ C++ files → CodeChunker → sanitize + LLM summary → detect sensitive info
│  ├─ Verilog files → VerilogChunker → annotate + LLM summary
│  └─ Docs → DocChunker → annotate
│  └─ Generate project summary → PROJECT_INDEX (two-stage retrieval entry)
│
├─ Project 2
│  └─ (same pipeline per project)
│
└─ Query
   ├─ Embed query (localhost embedding API)
   ├─ PROJECT_INDEX retrieval (top-k projects)
   ├─ CODE_INDEX + DOC_INDEX retrieval (filtered by project_id)
   ├─ Assemble chunks + metadata
   └─ Send to localhost LLM API (Qwen3.5-8B or compatible)

Indices
────────────────────
code_index:     function/class/module chunks with project/language metadata
doc_index:      markdown/word document chunks
project_index:  project-level summaries for first-hop retrieval
```

## Configuration

### `.env` file
```bash
# Localhost model gateway
LOCAL_LLM_BASE_URL=http://localhost:8000
LLM_PROVIDER=auto  # or "openai" / "anthropic"
LLM_MODEL=Qwen3.5-8B
EMBEDDING_MODEL=Qwen3-VL-Embedding-8B

# Index storage
RAG_INDEX_DIR=.rag_index
```

### CLI options

```bash
sec-rag-index \
  --path ./project-a \
  --path ./project-b \
  --languages python,c,cpp,verilog,doc \
  --batch-size 10 \
  --api-base-url http://localhost:8000 \
  --llm-provider auto \
  --llm-model Qwen3.5-8B \
  --verbose
```

| Option | Default | Description |
|--------|---------|-------------|
| `--path` | (required) | Project directory to index. Repeatable for multiple projects. |
| `--languages` | `python,c,doc` | Comma-separated: `python`, `c`, `cpp`, `systemc` (auto-detect), `verilog`, `systemverilog`, `doc` |
| `--batch-size` | `10` | Files to process per batch |
| `--api-base-url` | `http://localhost:8000` | Localhost model gateway URL |
| `--llm-provider` | `auto` | `auto`, `openai`, or `anthropic` |
| `--llm-model` | `Qwen3.5-8B` | LLM model name |
| `--verbose` | (off) | Debug logging |

## Desensitization Flow (C/C++/SystemC)

1. **Extract declaration**: Remove function body, keep signature only
   ```c
   // Before
   int authenticate_user(const char *user, const char *pass) {
       // 50 lines of logic...
   }

   // After
   int authenticate_user(const char *user, const char *pass);
   ```

2. **Summarize behavior** (LLM prompt):
   > "Convert this implementation into natural language. Do not copy code, strings, IPs, credentials."
   ```
   authenticate_user: Validates user identity against stored credentials, returns 1 if
   match, 0 otherwise. Uses constant-time comparison to prevent timing attacks.
   ```

3. **Build sanitized document**:
   ```
   project: my-app
   project_root: /path/to/my-app
   file: src/auth.c
   language: c
   symbol_type: function
   symbol_name: authenticate_user
   line_range: 42-68
   declaration: int authenticate_user(const char *user, const char *pass);
   summary: Validates user identity against stored credentials...
   ```

4. **Check sensitive info** (post-desensitization):
   - Raises `SensitiveInfoError` if any pattern matches:
     - Passwords: `password=***`, `SECRET=...`
     - Emails: `user@domain.com`
     - IPs: `192.168.1.1`
     - Phone: `+1 (555) 123-4567`
     - URL credentials: `ssh://user:pass@host`

## Sensitive Info Detection

Patterns checked (regex-based):
- `password`, `passwd`, `pwd`, `secret`, `token`, `api_key`
- Email addresses (RFC-simplified)
- IP addresses (IPv4)
- Phone numbers (international format)
- URLs with embedded credentials

**Policy**: Hard-fail on detection. Indexing stops immediately with error message.

## Project Structure

```
src/sec_rag/
├── __init__.py
├── cli.py                          # CLI entry points
├── index_store.py                  # ChromaDB persistence
├── query_pipeline.py               # End-to-end RAG query
├── project_processing.py           # Project context, desensitization, security
├── chunkers/
│   ├── __init__.py
│   ├── code_chunker.py             # Python/C/C++/SystemC (tree-sitter)
│   ├── doc_chunker.py              # Markdown/Word
│   └── verilog_chunker.py          # Verilog/SystemVerilog (regex)
tests/
├── conftest.py
├── test_code_chunker.py
├── test_doc_chunker.py
├── test_index_store.py
├── test_query_pipeline.py
```

## Testing

```bash
# Run all tests
pytest -q

# Run specific test class
pytest tests/test_code_chunker.py::TestPythonFunctionExtraction

# With verbose output
pytest -v
```

## Example Usage

### Index two projects
```bash
sec-rag-index \
  --path ./firmware/arm \
  --path ./firmware/dsp \
  --languages c,cpp,verilog \
  --verbose
```

This will:
1. Discover `.c`, `.h`, `.cpp`, `.hpp`, `.v`, `.sv` files
2. Parse C/C++ for functions/classes, Verilog for modules/blocks
3. For C/C++ files: sanitize → LLM summary → check sensitive info
4. For Verilog: annotate with metadata
5. Generate project summary for each project
6. Store in ChromaDB indices with project_id metadata

### Query
```bash
sec-rag-query "how does the DMA controller initialize?"
```

This will:
1. Embed query
2. Find top-3 projects most relevant (from `project_index`)
3. For each project: retrieve top-5 code chunks + top-3 doc chunks
4. Rank by similarity score
5. Assemble final prompt with top-10 unique chunks
6. Send to LLM

## Performance Notes

- **First run**: Model download + ChromaDB setup (~5-10 min depending on model size)
- **Indexing**: ~100-200 files/min (depends on file size, LLM latency for summaries)
- **Query**: ~1-2s (local embedding + ChromaDB retrieval) + LLM generation time
- **Token savings**: ~60-80% reduction vs. whole-file retrieval (desensitized summaries are 100-150 tokens vs. 2000+ for full functions)

## Troubleshooting

**Error: ModuleNotFoundError: No module named 'chromadb'**
```bash
pip install -e ".[dev]"
```

**Error: Failed to initialize local LLM**
- Ensure your localhost model gateway is running
- Check `--api-base-url` points to correct endpoint
- Verify endpoint is OpenAI-compatible or Anthropic-compatible

**Error: SensitiveInfoError during indexing**
- Review the flagged document section
- If false positive, submit an issue
- Sensitive patterns are hard-fail by design

## License

MIT
