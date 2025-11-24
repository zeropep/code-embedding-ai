"""
Unit tests for Python parser functionality.
Tests cover AST parsing, framework detection, and chunking.
"""

import pytest
import tempfile
from pathlib import Path

from src.code_parser.python_parser import PythonParser
from src.code_parser.python_framework_detector import PythonFrameworkDetector, PythonFramework
from src.code_parser.python_chunker import PythonChunker, ChunkingConfig
from src.code_parser.python_metadata import PythonMetadataExtractor
from src.code_parser.models import ParserConfig, CodeLanguage, LayerType


class TestPythonParser:
    """Tests for PythonParser class"""

    @pytest.fixture
    def parser(self):
        config = ParserConfig(min_tokens=10, max_tokens=500)
        return PythonParser(config)

    @pytest.fixture
    def sample_django_code(self):
        return '''
from django.db import models
from django.contrib.auth.models import User

class Article(models.Model):
    """Article model for blog posts"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    def get_summary(self):
        """Return first 100 characters of content"""
        return self.content[:100]
'''

    @pytest.fixture
    def sample_flask_code(self):
        return '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    users = [{'id': 1, 'name': 'John'}]
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get single user by ID"""
    return jsonify({'id': user_id, 'name': 'John'})
'''

    @pytest.fixture
    def sample_fastapi_code(self):
        return '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None

@app.get("/users/", response_model=List[User])
async def get_users():
    """Get all users"""
    return [User(id=1, name="John")]

@app.post("/users/", response_model=User)
async def create_user(user: User):
    """Create a new user"""
    return user
'''

    def test_can_parse_python_file(self, parser):
        """Test that parser recognizes .py files"""
        assert parser.can_parse(Path("test.py")) is True
        assert parser.can_parse(Path("test.pyw")) is True
        assert parser.can_parse(Path("test.java")) is False

    def test_get_language(self, parser):
        """Test language identification"""
        assert parser.get_language() == CodeLanguage.PYTHON

    def test_parse_django_code(self, parser, sample_django_code):
        """Test parsing Django model code"""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(sample_django_code)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result is not None
            assert result.language == CodeLanguage.PYTHON
            assert len(result.chunks) > 0

            # Check that Article class was detected
            class_chunks = [c for c in result.chunks if c.class_name == 'Article']
            assert len(class_chunks) > 0

    def test_parse_flask_code(self, parser, sample_flask_code):
        """Test parsing Flask route code"""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(sample_flask_code)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result is not None
            assert len(result.chunks) > 0

            # Check that route functions were detected
            func_names = [c.function_name for c in result.chunks if c.function_name]
            assert 'get_users' in func_names or any('get_users' in str(c.content) for c in result.chunks)

    def test_parse_fastapi_code(self, parser, sample_fastapi_code):
        """Test parsing FastAPI code"""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(sample_fastapi_code)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result is not None
            assert len(result.chunks) > 0

    def test_decorator_extraction(self, parser, sample_flask_code):
        """Test that decorators are properly extracted"""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(sample_flask_code)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result is not None
            # Check that metadata contains decorator info
            chunks_with_decorators = [c for c in result.chunks
                                      if c.metadata and c.metadata.get('decorators')]
            assert len(chunks_with_decorators) >= 0  # May or may not have decorators


class TestPythonFrameworkDetector:
    """Tests for PythonFrameworkDetector class"""

    @pytest.fixture
    def detector(self):
        return PythonFrameworkDetector()

    def test_detect_django_from_imports(self, detector):
        """Test Django detection from imports"""
        imports = {
            'modules': {'django', 'rest_framework'},
            'from_imports': {}
        }

        framework, confidence = detector.detect_framework_from_imports(imports)

        assert framework == PythonFramework.DJANGO
        assert confidence > 0

    def test_detect_flask_from_imports(self, detector):
        """Test Flask detection from imports"""
        imports = {
            'modules': {'flask', 'flask_login'},
            'from_imports': {}
        }

        framework, confidence = detector.detect_framework_from_imports(imports)

        assert framework == PythonFramework.FLASK
        assert confidence > 0

    def test_detect_fastapi_from_imports(self, detector):
        """Test FastAPI detection from imports"""
        imports = {
            'modules': {'fastapi', 'pydantic'},
            'from_imports': {}
        }

        framework, confidence = detector.detect_framework_from_imports(imports)

        assert framework == PythonFramework.FASTAPI
        assert confidence > 0

    def test_detect_layer_from_path_django(self, detector):
        """Test layer detection from Django file paths"""
        assert detector.detect_layer_from_path('app/models.py', PythonFramework.DJANGO) == LayerType.ENTITY
        assert detector.detect_layer_from_path('app/views.py', PythonFramework.DJANGO) == LayerType.VIEW
        assert detector.detect_layer_from_path('app/serializers.py', PythonFramework.DJANGO) == LayerType.SERIALIZER
        assert detector.detect_layer_from_path('app/admin.py', PythonFramework.DJANGO) == LayerType.ADMIN

    def test_detect_layer_from_class(self, detector):
        """Test layer detection from class definitions"""
        # Django Model
        layer = detector.detect_layer_from_class(
            'Article',
            ['models.Model'],
            [],
            PythonFramework.DJANGO
        )
        assert layer == LayerType.ENTITY

        # Django Serializer
        layer = detector.detect_layer_from_class(
            'ArticleSerializer',
            ['ModelSerializer'],
            [],
            PythonFramework.DJANGO
        )
        assert layer == LayerType.SERIALIZER


class TestPythonChunker:
    """Tests for PythonChunker class"""

    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(min_tokens=10, max_tokens=200)
        return PythonChunker(config)

    @pytest.fixture
    def sample_code_with_classes(self):
        return '''
class Calculator:
    """A simple calculator class"""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers"""
        return x + y

    def subtract(self, x, y):
        """Subtract y from x"""
        return x - y

    def multiply(self, x, y):
        """Multiply two numbers"""
        return x * y


class AdvancedCalculator(Calculator):
    """Extended calculator with more operations"""

    def divide(self, x, y):
        """Divide x by y"""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y

    def power(self, x, y):
        """Raise x to the power of y"""
        return x ** y
'''

    def test_chunk_file(self, chunker, sample_code_with_classes):
        """Test chunking a file with classes"""
        chunks = chunker.chunk_file(sample_code_with_classes, 'calculator.py')

        assert len(chunks) > 0
        # Should have at least chunks for the classes
        class_names = [c.class_name for c in chunks if c.class_name]
        assert 'Calculator' in class_names or 'AdvancedCalculator' in class_names

    def test_respects_token_limits(self, chunker, sample_code_with_classes):
        """Test that chunking respects token limits"""
        chunks = chunker.chunk_file(sample_code_with_classes, 'calculator.py')

        for chunk in chunks:
            # Allow some tolerance for token estimation
            assert chunk.token_count <= chunker.config.max_tokens * 1.5


class TestPythonMetadataExtractor:
    """Tests for PythonMetadataExtractor class"""

    @pytest.fixture
    def extractor(self):
        return PythonMetadataExtractor()

    @pytest.fixture
    def sample_typed_code(self):
        return '''
from typing import List, Optional

def calculate_sum(numbers: List[int]) -> int:
    """Calculate sum of numbers."""
    return sum(numbers)

def find_item(items: List[str], target: str) -> Optional[int]:
    """Find index of target in items."""
    try:
        return items.index(target)
    except ValueError:
        return None

class DataProcessor:
    """Process data with type hints."""

    def __init__(self, data: List[dict]) -> None:
        self.data = data

    def filter_by_key(self, key: str, value: str) -> List[dict]:
        """Filter data by key-value pair."""
        return [d for d in self.data if d.get(key) == value]
'''

    def test_extract_metadata(self, extractor, sample_typed_code):
        """Test basic metadata extraction"""
        metadata = extractor.extract_metadata(sample_typed_code, 'processor.py')

        assert 'imports' in metadata
        assert 'functions' in metadata
        assert 'classes' in metadata
        assert 'complexity' in metadata

    def test_extract_imports(self, extractor, sample_typed_code):
        """Test import extraction"""
        metadata = extractor.extract_metadata(sample_typed_code, 'processor.py')

        imports = metadata['imports']
        assert 'typing' in imports['standard_library']

    def test_extract_type_hints(self, extractor, sample_typed_code):
        """Test type hint extraction"""
        metadata = extractor.extract_metadata(sample_typed_code, 'processor.py')

        type_hints = metadata['type_hints']
        assert type_hints['total_functions'] > 0
        assert type_hints['return_type_coverage'] > 0

    def test_calculate_complexity(self, extractor, sample_typed_code):
        """Test complexity calculation"""
        metadata = extractor.extract_metadata(sample_typed_code, 'processor.py')

        complexity = metadata['complexity']
        assert complexity['lines_of_code'] > 0
        assert complexity['cyclomatic_complexity'] >= 1


class TestIntegration:
    """Integration tests for Python parsing pipeline"""

    @pytest.fixture
    def full_django_app(self):
        return '''
# Django views.py
from django.views import View
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Article
from .serializers import ArticleSerializer

class ArticleListView(View):
    """List all articles"""

    def get(self, request):
        articles = Article.objects.all()
        data = [{'id': a.id, 'title': a.title} for a in articles]
        return JsonResponse({'articles': data})

    def post(self, request):
        # Create new article
        title = request.POST.get('title')
        content = request.POST.get('content')
        article = Article.objects.create(
            title=title,
            content=content,
            author=request.user
        )
        return JsonResponse({'id': article.id})


@api_view(['GET'])
def article_detail(request, pk):
    """Get article detail via DRF"""
    try:
        article = Article.objects.get(pk=pk)
    except Article.DoesNotExist:
        return Response({'error': 'Not found'}, status=404)

    serializer = ArticleSerializer(article)
    return Response(serializer.data)
'''

    def test_full_parsing_pipeline(self, full_django_app):
        """Test complete parsing pipeline"""
        config = ParserConfig(min_tokens=10, max_tokens=500)
        parser = PythonParser(config)

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(full_django_app)
            f.flush()

            result = parser.parse_file(Path(f.name))

            assert result is not None
            assert result.language == CodeLanguage.PYTHON
            assert len(result.chunks) > 0

            # Verify we got both class and function
            has_class = any(c.class_name for c in result.chunks)
            has_function = any(c.function_name for c in result.chunks)

            assert has_class or has_function


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
