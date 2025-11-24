# AI Code Embedding Pipeline

## 한국어 가이드는 ./docs/README_ko.md 를 참조하세요.

A Python-based pipeline for processing Spring Boot + Thymeleaf codebases to generate vector embeddings for semantic code search and analysis.

## Features

- **Code Parsing**: Extract and chunk Java, Kotlin, and Thymeleaf code
- **Security Scanning**: Detect and mask sensitive information
- **Vector Embeddings**: Generate embeddings using jina-code-embeddings-1.5b
- **Vector Storage**: Store embeddings in ChromaDB with metadata
- **Incremental Updates**: Process only changed files using Git diff
- **REST API**: Query embeddings via HTTP endpoints
- **CLI Interface**: Command-line tools for pipeline operations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd code-embedding-pipeline

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 2. Configuration

Edit `.env` file:
```env
JINA_API_KEY=your_jina_api_key_here
REPO_PATH=./target_repo
CHROMADB_HOST=localhost
```

### 3. Usage

#### Process a repository
```bash
python src/main.py process --repo-path /path/to/spring-boot-repo
```

#### Update embeddings (incremental)
```bash
python src/main.py update --repo-path /path/to/spring-boot-repo
```

#### Search embeddings
```bash
python src/main.py search --query "user authentication"
```

#### Start API server
```bash
python src/main.py serve
```

## Architecture

```
Source Code → Parser → Security Scanner → Embedding Generator → Vector Database → API
```

See [architecture.md](architecture.md) for detailed system design.

## Development

### Setup Development Environment
```bash
pip install -e .[dev]
```

### Run Tests
```bash
pytest
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Configuration

- `config/pipeline.yaml`: Main pipeline configuration
- `config/security.yaml`: Security scanning rules
- `.env`: Environment variables and API keys

## API Endpoints

- `POST /embeddings/process`: Process repository
- `POST /embeddings/update`: Incremental update
- `GET /embeddings/search`: Search similar code
- `GET /health`: Health check

## License

MIT License