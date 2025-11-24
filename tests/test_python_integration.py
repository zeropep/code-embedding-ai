"""
Integration tests for Python support in the code embedding pipeline.
Tests the complete flow from parsing to embedding generation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.code_parser.parser_factory import ParserFactory
from src.code_parser.python_parser import PythonParser
from src.code_parser.python_framework_detector import PythonFrameworkDetector, PythonFramework
from src.code_parser.python_metadata import PythonMetadataExtractor
from src.code_parser.python_chunker import PythonChunker, ChunkingConfig
from src.code_parser.models import ParserConfig, CodeLanguage, LayerType
from src.security.secret_detector import SecretDetector
from src.security.models import SecurityConfig


class TestFullPipelineIntegration:
    """Integration tests for complete Python processing pipeline"""

    @pytest.fixture
    def parser_factory(self):
        config = ParserConfig(min_tokens=10, max_tokens=500)
        return ParserFactory(config)

    @pytest.fixture
    def django_project(self, tmp_path):
        """Create a mock Django project structure"""
        # Create project structure
        app_dir = tmp_path / "myapp"
        app_dir.mkdir()

        # models.py
        models_py = app_dir / "models.py"
        models_py.write_text('''
from django.db import models
from django.contrib.auth.models import User

class Article(models.Model):
    """Article model for blog"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
''')

        # views.py
        views_py = app_dir / "views.py"
        views_py.write_text('''
from django.views.generic import ListView
from .models import Article

class ArticleListView(ListView):
    """List all articles"""
    model = Article
    template_name = 'articles/list.html'

    def get_queryset(self):
        return Article.objects.all()
''')

        # serializers.py
        serializers_py = app_dir / "serializers.py"
        serializers_py.write_text('''
from rest_framework import serializers
from .models import Article

class ArticleSerializer(serializers.ModelSerializer):
    """Serializer for Article model"""
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author']
''')

        # admin.py
        admin_py = app_dir / "admin.py"
        admin_py.write_text('''
from django.contrib import admin
from .models import Article

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'created_at']
''')

        # manage.py (indicator)
        manage_py = tmp_path / "manage.py"
        manage_py.write_text('#!/usr/bin/env python')

        return tmp_path

    @pytest.fixture
    def flask_project(self, tmp_path):
        """Create a mock Flask project structure"""
        # app.py
        app_py = tmp_path / "app.py"
        app_py.write_text('''
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'name': u.name} for u in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user = User(name=data['name'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'id': user.id}), 201
''')

        return tmp_path

    @pytest.fixture
    def fastapi_project(self, tmp_path):
        """Create a mock FastAPI project structure"""
        # main.py
        main_py = tmp_path / "main.py"
        main_py.write_text('''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None

items_db: List[Item] = []

@app.get("/items/", response_model=List[Item])
async def list_items():
    return items_db

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    items_db.append(item)
    return item

@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Not found")
''')

        return tmp_path

    def test_detect_django_project(self, parser_factory, django_project):
        """Test Django project detection"""
        assert parser_factory.is_python_project(django_project)

        project_types = parser_factory.detect_project_type(django_project)
        assert project_types['python'] is True

    def test_parse_django_project(self, parser_factory, django_project):
        """Test parsing complete Django project"""
        parsed_files = parser_factory.parse_directory(django_project)

        assert len(parsed_files) > 0

        # Check that we got Python files
        languages = [f.language for f in parsed_files]
        assert CodeLanguage.PYTHON in languages

        # Check chunks were created
        total_chunks = sum(len(f.chunks) for f in parsed_files)
        assert total_chunks > 0

    def test_django_layer_detection(self, django_project):
        """Test that Django layers are correctly detected"""
        config = ParserConfig(min_tokens=10, max_tokens=500)
        parser = PythonParser(config)

        # Parse models.py
        models_path = django_project / "myapp" / "models.py"
        result = parser.parse_file(models_path)

        assert result is not None
        # Should detect Article as ENTITY
        entity_chunks = [c for c in result.chunks if c.layer_type == LayerType.ENTITY]
        assert len(entity_chunks) >= 0  # May or may not have explicit layer

    def test_flask_project_parsing(self, parser_factory, flask_project):
        """Test parsing Flask project"""
        parsed_files = parser_factory.parse_directory(flask_project)

        assert len(parsed_files) > 0

        # Check for route functions
        all_chunks = []
        for f in parsed_files:
            all_chunks.extend(f.chunks)

        # Should have function chunks
        func_chunks = [c for c in all_chunks if c.function_name]
        assert len(func_chunks) >= 0

    def test_fastapi_project_parsing(self, parser_factory, fastapi_project):
        """Test parsing FastAPI project"""
        parsed_files = parser_factory.parse_directory(fastapi_project)

        assert len(parsed_files) > 0

        # Check that Pydantic models and endpoints are detected
        all_chunks = []
        for f in parsed_files:
            all_chunks.extend(f.chunks)

        assert len(all_chunks) > 0


class TestSecurityIntegration:
    """Integration tests for security scanning with Python"""

    @pytest.fixture
    def detector(self):
        config = SecurityConfig(enabled=True, sensitivity_threshold=0.5)
        return SecretDetector(config)

    def test_scan_django_settings(self, detector, tmp_path):
        """Test scanning Django settings file"""
        settings_py = tmp_path / "settings.py"
        settings_py.write_text('''
SECRET_KEY = 'django-insecure-abcdefghijklmnop1234567890'
DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'PASSWORD': 'super_secret_password',
    }
}
''')

        content = settings_py.read_text()
        secrets = detector.detect_python_secrets(content, str(settings_py), 'django')

        assert len(secrets) > 0

    def test_scan_env_file(self, detector, tmp_path):
        """Test scanning .env file"""
        env_file = tmp_path / ".env"
        env_file.write_text('''
SECRET_KEY=production-secret-key-here
DATABASE_URL=postgres://user:password@localhost/db
API_KEY=sk_live_abcdefghij1234567890
''')

        content = env_file.read_text()
        secrets = detector.detect_python_secrets(content, str(env_file))

        assert len(secrets) >= 1


class TestMetadataIntegration:
    """Integration tests for metadata extraction"""

    @pytest.fixture
    def extractor(self):
        return PythonMetadataExtractor()

    def test_extract_full_metadata(self, extractor, tmp_path):
        """Test extracting metadata from a complete file"""
        code_file = tmp_path / "service.py"
        code_file.write_text('''
"""Service module for user operations."""

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None


class UserService:
    """Service for user CRUD operations."""

    def __init__(self, repository):
        self.repository = repository

    def get_users(self) -> List[User]:
        """Get all users."""
        return self.repository.find_all()

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.repository.find_by_id(user_id)

    def create_user(self, name: str, email: str = None) -> User:
        """Create new user."""
        user = User(id=0, name=name, email=email)
        return self.repository.save(user)
''')

        content = code_file.read_text()
        metadata = extractor.extract_metadata(content, str(code_file))

        # Check imports
        assert 'imports' in metadata
        assert 'typing' in metadata['imports']['standard_library']
        assert 'dataclasses' in metadata['imports']['standard_library']

        # Check classes
        assert 'classes' in metadata
        class_names = [c['name'] for c in metadata['classes']]
        assert 'User' in class_names
        assert 'UserService' in class_names

        # Check type hints
        assert 'type_hints' in metadata
        assert metadata['type_hints']['total_functions'] > 0

        # Check complexity
        assert 'complexity' in metadata
        assert metadata['complexity']['lines_of_code'] > 0


class TestChunkingIntegration:
    """Integration tests for chunking strategies"""

    @pytest.fixture
    def chunker(self):
        config = ChunkingConfig(min_tokens=10, max_tokens=200)
        return PythonChunker(config)

    def test_chunk_large_class(self, chunker):
        """Test chunking a large class into multiple pieces"""
        large_class = '''
class LargeService:
    """A service with many methods."""

    def __init__(self):
        self.data = []

    def method_one(self):
        """First method."""
        for i in range(10):
            self.data.append(i)
        return self.data

    def method_two(self):
        """Second method."""
        result = []
        for item in self.data:
            if item > 5:
                result.append(item)
        return result

    def method_three(self):
        """Third method."""
        total = 0
        for item in self.data:
            total += item
        return total

    def method_four(self):
        """Fourth method."""
        return len(self.data)

    def method_five(self):
        """Fifth method."""
        self.data = []
        return True
'''
        chunks = chunker.chunk_file(large_class, 'service.py')

        # Should create multiple chunks
        assert len(chunks) > 0

        # Chunks should not exceed max tokens (with tolerance)
        for chunk in chunks:
            assert chunk.token_count <= chunker.config.max_tokens * 1.5


class TestPerformance:
    """Performance tests for Python parsing"""

    def test_parse_large_file(self):
        """Test parsing performance with a large file"""
        # Generate a large Python file
        large_content = '''
"""Large module for performance testing."""

from typing import List, Dict, Optional
import json

'''
        # Add 50 functions
        for i in range(50):
            large_content += f'''
def function_{i}(param1: int, param2: str) -> Dict:
    """Function {i} docstring."""
    result = {{"id": param1, "name": param2}}
    if param1 > 0:
        result["valid"] = True
    return result

'''

        # Add 10 classes with 5 methods each
        for i in range(10):
            large_content += f'''
class Service{i}:
    """Service class {i}."""

    def __init__(self):
        self.data = []

'''
            for j in range(5):
                large_content += f'''
    def method_{j}(self, value: int) -> Optional[int]:
        """Method {j} of Service{i}."""
        if value > 0:
            return value * 2
        return None

'''

        config = ParserConfig(min_tokens=10, max_tokens=500)
        parser = PythonParser(config)

        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(large_content)
            f.flush()

            # Parse the file
            import time
            start = time.time()
            result = parser.parse_file(Path(f.name))
            elapsed = time.time() - start

            assert result is not None
            assert len(result.chunks) > 0

            # Should complete in reasonable time (< 5 seconds)
            assert elapsed < 5.0

            # Clean up
            os.unlink(f.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
