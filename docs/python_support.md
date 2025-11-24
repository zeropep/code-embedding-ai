# Python Support Guide

This document describes the Python language support features in the Code Embedding AI Pipeline.

## Overview

The pipeline now supports Python code analysis with specialized features for:
- **Django** (models, views, serializers, admin, signals)
- **Flask** (routes, blueprints, resources)
- **FastAPI** (endpoints, schemas, dependencies)

## Supported File Types

- `.py` - Python source files
- `.pyw` - Python Windows script files

## Features

### 1. AST-Based Parsing

Python files are parsed using Python's built-in `ast` module, providing accurate extraction of:

- **Classes** - Class definitions with inheritance information
- **Functions/Methods** - Function signatures with decorators
- **Type Hints** - Full type annotation support
- **Docstrings** - Automatic extraction of documentation

### 2. Framework Detection

The parser automatically detects Python frameworks based on:

- Import statements (django, flask, fastapi, pydantic)
- File naming patterns (models.py, views.py, serializers.py)
- Project structure (manage.py, app.py, main.py)

### 3. Layer Classification

Code is automatically classified into architectural layers:

| Layer | Django | Flask | FastAPI |
|-------|--------|-------|---------|
| Entity/Model | models.Model | db.Model | BaseModel |
| View/Controller | View, APIView | Resource, route | get/post endpoints |
| Serializer | Serializer | - | BaseModel (schemas) |
| Form | Form, ModelForm | FlaskForm | - |
| Admin | ModelAdmin | - | - |
| Middleware | MiddlewareMixin | before_request | middleware |
| Signal | receiver | - | on_event |
| Task | task, shared_task | - | - |

### 4. Security Scanning

Python-specific secret detection patterns include:

#### Django
- `SECRET_KEY` in settings.py
- Database passwords in DATABASES config
- Django REST Framework tokens

#### Flask
- `app.secret_key` assignments
- Flask-JWT secrets
- Session configuration

#### FastAPI
- OAuth2 secrets
- JWT configuration
- Pydantic Settings

#### Cloud Providers
- AWS Access Keys and Secret Keys
- Azure Storage Keys and Client Secrets
- GCP API Keys and Service Account Keys

#### Common Patterns
- GitHub tokens (ghp_, gho_, etc.)
- Stripe API keys (sk_live_, sk_test_)
- Slack tokens (xox-)
- Database URLs (postgresql://, mysql://)

## Usage

### CLI

```bash
# Process a Python project
python src/cli.py process /path/to/python-project

# Search in Python code
python src/cli.py search "user authentication" --language python

# Process with framework-specific optimization
python src/cli.py process /path/to/django-project --framework django
```

### Python API

```python
from src.code_parser.parser_factory import ParserFactory
from src.code_parser.models import ParserConfig

# Configure parser
config = ParserConfig(
    min_tokens=50,
    max_tokens=500,
    supported_extensions=['.py']
)

# Create factory
factory = ParserFactory(config)

# Check if Python project
if factory.is_python_project(project_path):
    # Process Python files
    parsed_files = factory.parse_directory(project_path)
```

### Framework-Specific Processing

```python
from src.code_parser.python_parser import PythonParser
from src.code_parser.python_framework_detector import PythonFrameworkDetector

# Detect framework
detector = PythonFrameworkDetector()
framework = detector.detect_framework_from_project(project_path)

# Parse with framework context
parser = PythonParser(config)
result = parser.parse_file(file_path)
```

## Configuration

### Parser Configuration

```yaml
parser:
  min_tokens: 50
  max_tokens: 500
  overlap_tokens: 20
  supported_extensions:
    - ".py"
    - ".pyw"
  extract_methods: true
  extract_classes: true
```

### Security Configuration

```yaml
security:
  enabled: true
  scan_python: true
  python_patterns:
    - django_secret_key
    - flask_secret_key
    - aws_credentials
    - database_urls
```

## Best Practices

### 1. Project Structure

For best results, organize your Python project with clear separation:

```
project/
├── app/
│   ├── models.py      # Will be tagged as ENTITY
│   ├── views.py       # Will be tagged as VIEW
│   ├── serializers.py # Will be tagged as SERIALIZER
│   └── admin.py       # Will be tagged as ADMIN
├── tests/
│   └── test_*.py      # Will be tagged as TEST
└── manage.py
```

### 2. Type Hints

Adding type hints improves metadata extraction:

```python
def get_user(user_id: int) -> Optional[User]:
    """Get user by ID."""
    return User.objects.filter(id=user_id).first()
```

### 3. Docstrings

Include docstrings for better semantic understanding:

```python
class UserService:
    """Service for user-related operations.

    Handles user authentication, registration, and profile management.
    """

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user with credentials."""
        ...
```

## Excluded Directories

The following are automatically excluded from processing:

- Virtual environments: `venv/`, `.venv/`, `env/`
- Cache directories: `__pycache__/`, `.pytest_cache/`
- Build artifacts: `dist/`, `*.egg-info/`
- IDE settings: `.idea/`, `.vscode/`

## Troubleshooting

### 1. Syntax Errors

If a file has syntax errors, the parser falls back to regex-based chunking:

```
Warning: Syntax error in /path/to/file.py, using fallback parser
```

### 2. Missing Framework Detection

If the framework isn't detected, ensure your imports are at the top of the file:

```python
# Good - will be detected
from django.db import models

# Bad - might not be detected
def get_models():
    from django.db import models  # Dynamic import
```

### 3. Performance Issues

For large projects, consider:

```bash
# Limit depth
python src/cli.py process /path/to/project --max-depth 3

# Exclude tests
python src/cli.py process /path/to/project --exclude "*/tests/*"
```

## API Reference

See [API Documentation](api_documentation.md) for complete REST API reference.

## Examples

### Django Project

```python
# models.py - Will be detected as ENTITY layer
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
```

### Flask Application

```python
# routes.py - Will be detected as CONTROLLER layer
from flask import Blueprint, jsonify

api = Blueprint('api', __name__)

@api.route('/users')
def get_users():
    return jsonify([])
```

### FastAPI Application

```python
# main.py - Will be detected as CONTROLLER layer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str

@app.get("/users/")
async def get_users():
    return []
```
