"""
Unit tests for Python security scanning functionality.
Tests cover secret detection patterns for Django, Flask, FastAPI, and cloud providers.
"""

import pytest

from src.security.secret_detector import SecretDetector
from src.security.python_patterns import PythonSecurityPatterns
from src.security.models import SecurityConfig, SecretType


class TestPythonSecurityPatterns:
    """Tests for PythonSecurityPatterns class"""

    def test_get_all_patterns(self):
        """Test getting all patterns"""
        patterns = PythonSecurityPatterns.get_all_patterns()

        assert len(patterns) > 0
        assert SecretType.SECRET_KEY in patterns
        assert SecretType.API_KEY in patterns

    def test_get_patterns_for_django(self):
        """Test getting Django-specific patterns"""
        patterns = PythonSecurityPatterns.get_patterns_for_framework('django')

        assert len(patterns) > 0
        # Should include Django patterns
        assert SecretType.SECRET_KEY in patterns

    def test_get_patterns_for_flask(self):
        """Test getting Flask-specific patterns"""
        patterns = PythonSecurityPatterns.get_patterns_for_framework('flask')

        assert len(patterns) > 0
        assert SecretType.SECRET_KEY in patterns

    def test_get_patterns_for_fastapi(self):
        """Test getting FastAPI-specific patterns"""
        patterns = PythonSecurityPatterns.get_patterns_for_framework('fastapi')

        assert len(patterns) > 0
        assert SecretType.SECRET_KEY in patterns

    def test_compile_patterns(self):
        """Test pattern compilation"""
        patterns = PythonSecurityPatterns.get_all_patterns()
        compiled = PythonSecurityPatterns.compile_patterns(patterns)

        assert len(compiled) > 0
        # Check that patterns are compiled regex objects
        for secret_type, pattern_list in compiled.items():
            for compiled_pattern, name, confidence in pattern_list:
                assert hasattr(compiled_pattern, 'finditer')


class TestSecretDetector:
    """Tests for SecretDetector with Python patterns"""

    @pytest.fixture
    def detector(self):
        config = SecurityConfig(enabled=True, sensitivity_threshold=0.5)
        return SecretDetector(config)

    def test_detect_django_secret_key(self, detector):
        """Test Django SECRET_KEY detection"""
        code = '''
SECRET_KEY = 'django-insecure-abcdefghijklmnopqrstuvwxyz123456'
DEBUG = True
'''
        secrets = detector.detect_python_secrets(code, 'settings.py', 'django')

        secret_keys = [s for s in secrets if s.secret_type == SecretType.SECRET_KEY]
        assert len(secret_keys) > 0

    def test_detect_flask_secret_key(self, detector):
        """Test Flask secret_key detection"""
        code = '''
from flask import Flask
app = Flask(__name__)
app.secret_key = 'super-secret-flask-key-12345678'
'''
        secrets = detector.detect_python_secrets(code, 'app.py', 'flask')

        secret_keys = [s for s in secrets if s.secret_type == SecretType.SECRET_KEY]
        assert len(secret_keys) > 0

    def test_detect_database_password(self, detector):
        """Test database password detection"""
        code = '''
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydb',
        'USER': 'admin',
        'PASSWORD': 'supersecretpassword123',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
'''
        secrets = detector.detect_python_secrets(code, 'settings.py', 'django')

        passwords = [s for s in secrets
                    if s.secret_type in [SecretType.PASSWORD, SecretType.DATABASE_URL]]
        assert len(passwords) > 0

    def test_detect_aws_credentials(self, detector):
        """Test AWS credential detection"""
        code = '''
AWS_ACCESS_KEY_ID = 'AKIAIOSFODNN7EXAMPLE'
AWS_SECRET_ACCESS_KEY = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
'''
        secrets = detector.detect_python_secrets(code, 'config.py')

        aws_secrets = [s for s in secrets
                      if 'aws' in s.pattern_name.lower()]
        assert len(aws_secrets) >= 1

    def test_detect_api_key(self, detector):
        """Test generic API key detection"""
        code = '''
STRIPE_API_KEY = 'sk_live_abcdefghijklmnopqrstuvwxyz123456'
OPENAI_API_KEY = 'sk-proj-abcdefghijklmnopqrstuvwxyz'
'''
        secrets = detector.detect_python_secrets(code, 'config.py')

        api_keys = [s for s in secrets
                   if s.secret_type == SecretType.API_KEY]
        assert len(api_keys) >= 1

    def test_detect_jwt_secret(self, detector):
        """Test JWT secret detection"""
        code = '''
JWT_SECRET_KEY = 'your-super-secret-jwt-key-here-1234567890'
JWT_ALGORITHM = 'HS256'
'''
        secrets = detector.detect_python_secrets(code, 'config.py', 'flask')

        jwt_secrets = [s for s in secrets
                      if s.secret_type in [SecretType.SECRET_KEY, SecretType.TOKEN]]
        assert len(jwt_secrets) >= 1

    def test_detect_github_token(self, detector):
        """Test GitHub token detection"""
        code = '''
GITHUB_TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890'
'''
        secrets = detector.detect_python_secrets(code, 'config.py')

        github_tokens = [s for s in secrets
                        if 'github' in s.pattern_name.lower()]
        assert len(github_tokens) >= 1

    def test_skip_test_files(self, detector):
        """Test that test files have reduced confidence"""
        code = '''
SECRET_KEY = 'test-secret-key-for-testing-purposes'
'''
        # Detection in regular file
        secrets_regular = detector.detect_python_secrets(code, 'settings.py')

        # Detection in test file - should have lower confidence
        secrets_test = detector.detect_python_secrets(code, 'test_settings.py')

        # Both should detect, but test file should have lower confidence
        if secrets_regular and secrets_test:
            assert secrets_test[0].confidence <= secrets_regular[0].confidence

    def test_skip_example_values(self, detector):
        """Test that example values are detected with lower confidence"""
        code_real = '''
SECRET_KEY = 'abc123def456ghi789jkl012mno345pqr678'
'''
        code_example = '''
SECRET_KEY = 'your-secret-key-here-example-placeholder'
'''
        secrets_real = detector.detect_python_secrets(code_real, 'settings.py')
        secrets_example = detector.detect_python_secrets(code_example, 'settings.py')

        # Example values should have lower confidence or not be detected
        if secrets_real and secrets_example:
            assert len(secrets_example) <= len(secrets_real)

    def test_detect_private_key(self, detector):
        """Test private key detection"""
        code = '''
PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAtest...
-----END RSA PRIVATE KEY-----"""
'''
        secrets = detector.detect_python_secrets(code, 'config.py')

        private_keys = [s for s in secrets
                       if s.secret_type == SecretType.PRIVATE_KEY]
        assert len(private_keys) >= 1

    def test_detect_sqlalchemy_url(self, detector):
        """Test SQLAlchemy database URL detection"""
        code = '''
SQLALCHEMY_DATABASE_URI = 'postgresql://user:password123@localhost/dbname'
'''
        secrets = detector.detect_python_secrets(code, 'config.py')

        db_urls = [s for s in secrets
                  if s.secret_type == SecretType.DATABASE_URL]
        assert len(db_urls) >= 1

    def test_env_file_detection(self, detector):
        """Test .env file secret detection"""
        env_content = '''
SECRET_KEY=my-production-secret-key-12345678
DATABASE_URL=postgres://admin:secretpass@db.example.com/app
API_KEY=sk_live_abcdefghijklmnop
'''
        secrets = detector.detect_python_secrets(env_content, '.env')

        assert len(secrets) >= 1


class TestSecurityIntegration:
    """Integration tests for security scanning"""

    @pytest.fixture
    def detector(self):
        config = SecurityConfig(enabled=True, sensitivity_threshold=0.5)
        return SecretDetector(config)

    def test_full_django_settings_scan(self, detector):
        """Test scanning a complete Django settings file"""
        settings_content = '''
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-abcdefghijklmnopqrstuvwxyz1234567890'

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'myapp_db',
        'USER': 'db_admin',
        'PASSWORD': 'SuperSecretDBPass123!',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

EMAIL_HOST_PASSWORD = 'email_secret_password'

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
'''
        secrets = detector.detect_python_secrets(settings_content, 'settings.py', 'django')

        # Should detect multiple secrets
        assert len(secrets) >= 2

        # Check for specific types
        secret_types = [s.secret_type for s in secrets]
        assert SecretType.SECRET_KEY in secret_types or SecretType.PASSWORD in secret_types

    def test_fastapi_config_scan(self, detector):
        """Test scanning a FastAPI configuration"""
        config_content = '''
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    secret_key: str = "my-super-secret-jwt-key-1234567890"
    database_url: str = "postgresql://user:pass@localhost/db"

    class Config:
        env_file = ".env"
'''
        secrets = detector.detect_python_secrets(config_content, 'config.py', 'fastapi')

        assert len(secrets) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
