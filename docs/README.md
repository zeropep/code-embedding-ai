# Code Embedding AI Pipeline

A comprehensive AI-powered code analysis pipeline that processes Spring Boot + Thymeleaf codebases, generates semantic embeddings, and enables intelligent code search with security features.

## Features

- **Multi-language Code Parsing**: Support for Java, Kotlin, HTML/Thymeleaf, and Python with AST-based analysis
- **Intelligent Chunking**: Semantic code splitting with configurable token limits and overlap
- **Security Scanning**: Automated detection and masking of secrets, credentials, and sensitive data
- **Vector Embeddings**: High-quality code embeddings using jina-code-embeddings-1.5b
- **Vector Storage**: ChromaDB integration with efficient similarity search
- **Incremental Updates**: Git diff-based monitoring for processing only changed files
- **REST API**: Full-featured web API for integration with external systems
- **CLI Interface**: Command-line tools for batch processing and management
- **Monitoring**: Comprehensive logging, metrics, and health monitoring

## Quick Start

### Prerequisites

- Python 3.9+
- Git repository with Java/Kotlin/HTML code
- Jina AI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd code-embedding-ai

# Install dependencies
uv sync

# Set up environment variables
export JINA_API_KEY="your-jina-api-key"
export CHROMADB_PERSIST_DIR="/path/to/vector/storage"
```

### Basic Usage

#### CLI Interface

```bash
# Process a repository
uv run python -m src.cli process /path/to/spring-boot-repo

# Search for code
uv run python -m src.cli search "user authentication logic"

# Start monitoring for changes
uv run python -m src.cli monitor /path/to/repo

# Start web server
uv run python -m src.cli server start --port 8000
```

#### Python API

```python
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.code_parser.models import ParserConfig
from src.security.models import SecurityConfig
from src.embeddings.models import EmbeddingConfig

# Configure pipeline
parser_config = ParserConfig(
    min_tokens=50,
    max_tokens=500,
    excluded_dirs=[".venv", "node_modules", "__pycache__"]  # Exclude directories
)
security_config = SecurityConfig(enabled=True)
embedding_config = EmbeddingConfig(api_key="your-api-key")

# Create pipeline
pipeline = EmbeddingPipeline(
    parser_config=parser_config,
    security_config=security_config,
    embedding_config=embedding_config,
    chunk_batch_size=100  # Process 100 chunks at a time
)

# Process repository
result = await pipeline.process_repository("/path/to/repo")
```

## Architecture

The pipeline consists of several key components:

### 1. Code Parser (`src/code_parser/`)
- **Language Support**: Java, Kotlin, HTML/Thymeleaf, Python
- **AST Analysis**: Method/class extraction, layer detection
- **Chunking Strategy**: Semantic splitting with configurable parameters
- **Python Frameworks**: Django, Flask, FastAPI detection and layer classification

### 2. Security Scanner (`src/security/`)
- **Secret Detection**: Passwords, API keys, tokens, database URLs
- **Content Masking**: Preserve syntax while hiding sensitive data
- **Pattern Matching**: Configurable regex patterns and whitelists

### 3. Embedding Service (`src/embeddings/`)
- **Jina AI Integration**: jina-code-embeddings-1.5b model
- **Batch Processing**: Optimized API calls with retry logic
- **Caching**: Optional embedding caching for performance

### 4. Vector Database (`src/database/`)
- **ChromaDB Backend**: Persistent vector storage
- **Metadata Indexing**: Rich metadata for filtering and search
- **Similarity Search**: Cosine similarity with configurable thresholds

### 5. Update System (`src/updates/`)
- **Git Monitoring**: Automatic detection of file changes
- **Incremental Processing**: Process only modified files
- **File Watching**: Real-time monitoring option

### 6. API & CLI (`src/api/`, `src/cli.py`)
- **REST Endpoints**: Process, search, monitor, stats
- **CLI Commands**: Batch processing and management
- **Authentication**: Optional API key authentication

### 7. Monitoring (`src/monitoring/`)
- **Structured Logging**: JSON-formatted logs with context
- **Metrics Collection**: Performance and usage statistics
- **Health Checks**: Service availability monitoring
- **Alerting**: Configurable alerts for errors and performance

## Configuration

### Environment Variables

```bash
# Required
JINA_API_KEY=your-jina-api-key

# Optional
CHROMADB_COLLECTION_NAME=code_embeddings
CHROMADB_PERSIST_DIR=/data/vector_db
LOG_LEVEL=INFO
CHUNK_MIN_TOKENS=50
CHUNK_MAX_TOKENS=500
SECURITY_ENABLED=true
API_PORT=8000
```

### Configuration Files

Create `config.yaml` for detailed configuration:

```yaml
parser:
  min_tokens: 50
  max_tokens: 500
  overlap_tokens: 50
  supported_extensions: [".java", ".kt", ".html", ".py"]
  excluded_dirs:
    - ".venv"
    - "venv"
    - "node_modules"
    - "__pycache__"
    - ".pytest_cache"
    - "chroma_db"
    - "dist"
    - "build"
    - ".git"
    - ".idea"
    - ".vscode"

security:
  enabled: true
  preserve_syntax: true
  sensitivity_threshold: 0.7
  whitelist_patterns: ["test_", "example_"]

embedding:
  model_name: "jina-code-embeddings-1.5b"
  batch_size: 32
  timeout: 30
  enable_caching: true

database:
  collection_name: "code_embeddings"
  persistent: true
  max_batch_size: 100

monitoring:
  enable_metrics: true
  enable_alerting: true
  log_level: "INFO"
```

### Parser Configuration Details

#### Excluded Directories

By default, the parser automatically excludes common directories that should not be embedded:

**Development Environments:**
- `.venv`, `venv`, `.env` - Python virtual environments
- `node_modules`, `bower_components` - JavaScript dependencies

**Version Control:**
- `.git`, `.svn`, `.hg` - VCS directories

**Python Caches:**
- `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`

**Build Artifacts:**
- `dist`, `build`, `target`, `out`

**IDE Settings:**
- `.idea`, `.vscode`, `.vs`

**Test/Coverage:**
- `coverage`, `.coverage`, `htmlcov`, `.tox`, `.nox`

**Databases:**
- `chroma_db`, `chromadb` - Vector database storage

**Other:**
- `logs`, `log` - Log files
- `.DS_Store`, `Thumbs.db` - OS files

You can customize this list using the `excluded_dirs` parameter:

```python
from src.code_parser.models import ParserConfig

# Add custom exclusions
parser_config = ParserConfig(
    excluded_dirs=[
        ".venv", "node_modules",  # Default exclusions
        "generated", "migrations",  # Custom exclusions
        "vendor", "third_party"
    ]
)
```

#### Batch Size Configuration

Control how many code chunks are processed at once:

```python
from src.embeddings.embedding_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    chunk_batch_size=100  # Process 100 chunks per batch (recommended for GPU)
)

# For CPU processing, use smaller batches:
# chunk_batch_size=50

# For high-memory GPU (24GB+):
# chunk_batch_size=200
```

**Batch Size Guidelines:**
- **CPU**: 10-50 chunks (slower, memory safe)
- **GPU (12GB)**: 100-150 chunks (recommended)
- **GPU (24GB+)**: 200-500 chunks (high performance)

## API Reference

### Process Repository
```http
POST /api/v1/process
{
    "repository_path": "/path/to/repo",
    "include_security_scan": true,
    "force_reprocess": false
}
```

### Search Code
```http
POST /api/v1/search
{
    "query": "user authentication logic",
    "limit": 10,
    "similarity_threshold": 0.7,
    "language_filter": "java",
    "layer_filter": "service"
}
```

### Health Check
```http
GET /health
```

### Get Statistics
```http
GET /api/v1/stats
```

## Security Features

### Secret Detection
The pipeline automatically detects and masks:
- Passwords and credentials
- API keys and tokens
- Database URLs with embedded credentials
- Private keys and certificates
- OAuth tokens and secrets

### Content Masking
- Preserves code syntax and structure
- Replaces sensitive values with typed placeholders
- Maintains code functionality for analysis
- Configurable sensitivity levels

### Security Reports
```python
# Generate security report
scanner = SecurityScanner(config)
report = scanner.generate_security_report(chunks)

print(f"Found {report['scan_summary']['total_secrets_found']} secrets")
print(f"High-risk files: {len(report['high_risk_files'])}")
```

## Performance Optimization

### Batch Processing
- Configurable batch sizes for API calls
- Parallel processing for multiple files
- Memory-efficient chunk processing

### Caching
- Optional embedding caching
- File hash-based change detection
- Persistent vector storage

### Incremental Updates
- Git diff-based change detection
- Process only modified files
- Efficient database updates

## Monitoring and Observability

### Metrics
- Processing times and throughput
- API response times and error rates
- Resource usage (CPU, memory)
- Cache hit/miss ratios

### Logging
- Structured JSON logs
- Request/response tracing
- Security event logging
- Performance metrics

### Health Checks
- Service availability monitoring
- Database connectivity
- API endpoint status
- Resource utilization

### Alerting
- Error rate thresholds
- Performance degradation
- Resource exhaustion
- Security events

## Development

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py --unit
python run_tests.py --integration

# Run with coverage
python run_tests.py --coverage --html
```

### Code Quality
```bash
# Type checking
mypy src/

# Code style
flake8 src/ tests/

# Security scan
bandit -r src/
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY requirements.txt .
RUN uv sync

COPY src/ src/
COPY config.yaml .

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Configuration
- Use environment-specific configuration files
- Set up proper logging and monitoring
- Configure security settings and API keys
- Set up backup and recovery procedures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

[License information]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test examples