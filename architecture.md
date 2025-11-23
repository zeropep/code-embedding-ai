# AI Code Embedding Pipeline - Architecture Design

## System Overview
```
[Source Code Repository]
        ↓
[Code Parser & Chunker]
        ↓
[Sensitive Data Detector & Masker]
        ↓
[Embedding Generator (Jina)]
        ↓
[Vector Database (ChromaDB)]
        ↓
[REST API / CLI Interface]
```

## Core Components

### 1. Code Parser & Chunker (`code_parser/`)
**Purpose**: Extract and chunk source code into manageable pieces
- **Input**: Spring Boot + Thymeleaf source files
- **Output**: Code chunks (200-500 tokens)
- **Technology**: tree-sitter for Java/Kotlin, custom parser for Thymeleaf
- **Functions**:
  - `parse_java_files()` - Extract classes, methods, functions
  - `parse_thymeleaf_templates()` - Parse HTML templates
  - `chunk_code()` - Split large code blocks
  - `validate_chunk_size()` - Ensure token limits

### 2. Sensitive Data Handler (`security/`)
**Purpose**: Detect and mask sensitive information
- **Input**: Raw code chunks
- **Output**: Sanitized code chunks with sensitivity metadata
- **Technology**: truffleHog integration
- **Functions**:
  - `scan_for_secrets()` - Detect secrets, keys, tokens
  - `mask_sensitive_data()` - Replace with placeholders
  - `preserve_syntax()` - Maintain code structure
  - `tag_sensitivity_level()` - Classify sensitivity (LOW/MED/HIGH)

### 3. Embedding Generator (`embeddings/`)
**Purpose**: Generate vector embeddings for code chunks
- **Input**: Sanitized code chunks
- **Output**: Vector embeddings (1024-dimensional)
- **Technology**: jina-embeddings-v2-base-code API
- **Functions**:
  - `generate_embedding()` - Create vectors
  - `batch_process()` - Handle multiple chunks
  - `retry_failed_embeddings()` - Error recovery
  - `validate_embedding_quality()` - Check vector quality

### 4. Vector Database Manager (`database/`)
**Purpose**: Store and manage embeddings with metadata
- **Input**: Embeddings + metadata
- **Output**: Indexed vector storage
- **Technology**: ChromaDB client
- **Functions**:
  - `store_embedding()` - Save vectors with metadata
  - `update_embedding()` - Modify existing entries
  - `delete_embedding()` - Remove outdated vectors
  - `query_similar()` - Search similar code chunks

### 5. Incremental Update Manager (`updates/`)
**Purpose**: Detect changes and update only modified code
- **Input**: Git repository state
- **Output**: List of changed files
- **Technology**: GitPython, file system monitoring
- **Functions**:
  - `detect_git_changes()` - Find modified files
  - `calculate_file_hash()` - Track file changes
  - `schedule_updates()` - Queue re-processing
  - `cleanup_deleted_files()` - Remove obsolete entries

### 6. API & CLI Interface (`api/`)
**Purpose**: Provide external access to the pipeline
- **Input**: HTTP requests / CLI commands
- **Output**: JSON responses / CLI output
- **Technology**: FastAPI, Click CLI
- **Functions**:
  - `trigger_full_scan()` - Process entire codebase
  - `trigger_incremental_update()` - Process changes only
  - `query_embeddings()` - Search vector database
  - `get_pipeline_status()` - Monitor system health

## Data Flow

### 1. Initial Processing
```
Source Code → Parser → Chunker → Security Scanner → Masker → Embedding Generator → Vector DB
```

### 2. Incremental Updates
```
Git Changes → Change Detector → Modified Files → (repeat processing for changed chunks only)
```

### 3. Query Processing
```
Query Request → API → Vector DB → Similar Chunks → Metadata → Response
```

## Metadata Schema

### Code Chunk Metadata
```json
{
  "chunk_id": "unique_identifier",
  "file_path": "src/main/java/com/example/Controller.java",
  "function_name": "getUserById",
  "class_name": "UserController",
  "layer_type": "Controller|Service|Repository|Entity",
  "line_start": 45,
  "line_end": 67,
  "chunk_tokens": 234,
  "sensitivity_level": "LOW|MEDIUM|HIGH",
  "masked_secrets": ["API_KEY", "PASSWORD"],
  "last_updated": "2024-01-15T10:30:00Z",
  "file_hash": "sha256_hash",
  "embedding_vector": [0.1, 0.2, ...]
}
```

## Configuration

### Environment Variables
- `JINA_API_KEY` - API key for embedding service
- `CHROMADB_HOST` - Vector database connection
- `REPO_PATH` - Source code repository path
- `LOG_LEVEL` - Logging verbosity
- `CHUNK_SIZE_LIMIT` - Maximum tokens per chunk

### Configuration Files
- `config/pipeline.yaml` - Pipeline settings
- `config/security.yaml` - Sensitivity detection rules
- `config/parser.yaml` - Language-specific parsing rules

## Scalability Considerations
- **Parallel Processing**: Multi-threaded chunk processing
- **Batch Operations**: Group API calls to reduce latency
- **Caching**: Store intermediate results for faster re-processing
- **Queue Management**: Handle large repositories with job queues