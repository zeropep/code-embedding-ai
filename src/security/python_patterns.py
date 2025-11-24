"""
Python-specific security patterns for secret detection.
Covers Django, Flask, FastAPI, and common Python security patterns.
"""

import re
from typing import Dict, List, Tuple
from .models import SecretType


class PythonSecurityPatterns:
    """Python-specific security patterns for secret detection"""

    # Django-specific patterns
    DJANGO_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.SECRET_KEY: [
            # Django SECRET_KEY in settings.py
            (r'SECRET_KEY\s*=\s*["\']([^"\']{20,})["\']', 'django_secret_key', 0.95),
            (r'SECRET_KEY\s*=\s*os\.environ\.get\(["\']([^"\']+)["\']', 'django_secret_key_env', 0.3),
            # Django signing key
            (r'SIGNING_KEY\s*=\s*["\']([^"\']{20,})["\']', 'django_signing_key', 0.9),
        ],
        SecretType.DATABASE_URL: [
            # Django database settings
            (r"'PASSWORD'\s*:\s*['\"]([^'\"]+)['\"]", 'django_db_password', 0.85),
            (r"'ENGINE'\s*:\s*['\"]django\.db\.backends\.\w+['\"].*?'PASSWORD'\s*:\s*['\"]([^'\"]+)['\"]",
             'django_db_config', 0.9),
            (r'DATABASE_URL\s*=\s*["\']([^"\']+://[^"\']+)["\']', 'database_url', 0.9),
            (r'DATABASES\s*=\s*\{[^}]*["\']PASSWORD["\']\s*:\s*["\']([^"\']+)["\']', 'django_databases', 0.85),
        ],
        SecretType.API_KEY: [
            # Django REST Framework token
            (r'Token\s+([a-f0-9]{40})', 'drf_token', 0.9),
            # Django API keys
            (r'DJANGO_API_KEY\s*=\s*["\']([^"\']{20,})["\']', 'django_api_key', 0.85),
        ],
        SecretType.TOKEN: [
            # Django CSRF token (in code, not templates)
            (r'csrftoken\s*=\s*["\']([^"\']{32,})["\']', 'django_csrf_token', 0.7),
        ],
    }

    # Flask-specific patterns
    FLASK_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.SECRET_KEY: [
            # Flask SECRET_KEY
            (r'app\.secret_key\s*=\s*["\']([^"\']{16,})["\']', 'flask_secret_key', 0.95),
            (r'SECRET_KEY\s*=\s*["\']([^"\']{16,})["\']', 'flask_secret_key_config', 0.9),
            (r"app\.config\['SECRET_KEY'\]\s*=\s*['\"]([^'\"]{16,})['\"]", 'flask_config_secret', 0.95),
        ],
        SecretType.TOKEN: [
            # Flask-JWT
            (r'JWT_SECRET_KEY\s*=\s*["\']([^"\']{16,})["\']', 'flask_jwt_secret', 0.95),
            (r'JWT_SECRET\s*=\s*["\']([^"\']{16,})["\']', 'flask_jwt_secret_alt', 0.95),
        ],
        SecretType.PASSWORD: [
            # Flask-Login
            (r'SECURITY_PASSWORD_SALT\s*=\s*["\']([^"\']{16,})["\']', 'flask_password_salt', 0.85),
        ],
    }

    # FastAPI-specific patterns
    FASTAPI_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.SECRET_KEY: [
            # FastAPI/Starlette secret key
            (r'SECRET_KEY\s*:\s*str\s*=\s*["\']([^"\']{16,})["\']', 'fastapi_secret_key', 0.95),
            (r'secret_key\s*=\s*["\']([^"\']{16,})["\']', 'fastapi_secret_key_alt', 0.9),
        ],
        SecretType.TOKEN: [
            # FastAPI OAuth2
            (r'SECRET_KEY\s*=\s*["\']([^"\']{32,})["\']', 'fastapi_oauth_secret', 0.9),
            (r'ALGORITHM\s*=\s*["\']HS256["\'].*?SECRET_KEY\s*=\s*["\']([^"\']+)["\']', 'fastapi_jwt_config', 0.95),
        ],
    }

    # AWS patterns
    AWS_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.API_KEY: [
            # AWS Access Key ID (starts with AKIA, ABIA, ACCA, ASIA)
            (r'(A[KBS]IA[A-Z0-9]{16})', 'aws_access_key_id', 0.95),
            (r'AWS_ACCESS_KEY_ID\s*=\s*["\']?(A[KBS]IA[A-Z0-9]{16})["\']?', 'aws_access_key_env', 0.95),
            (r'aws_access_key_id\s*=\s*["\']?(A[KBS]IA[A-Z0-9]{16})["\']?', 'aws_access_key_config', 0.95),
        ],
        SecretType.SECRET_KEY: [
            # AWS Secret Access Key (40 chars, base64-like)
            (r'AWS_SECRET_ACCESS_KEY\s*=\s*["\']?([A-Za-z0-9/+=]{40})["\']?', 'aws_secret_key', 0.95),
            (r'aws_secret_access_key\s*=\s*["\']?([A-Za-z0-9/+=]{40})["\']?', 'aws_secret_key_config', 0.95),
        ],
        SecretType.TOKEN: [
            # AWS Session Token
            (r'AWS_SESSION_TOKEN\s*=\s*["\']?([A-Za-z0-9/+=]+)["\']?', 'aws_session_token', 0.9),
        ],
    }

    # Azure patterns
    AZURE_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.API_KEY: [
            # Azure Storage Account Key
            (r'AccountKey\s*=\s*([A-Za-z0-9+/=]{88})', 'azure_storage_key', 0.95),
            (r'AZURE_STORAGE_KEY\s*=\s*["\']?([A-Za-z0-9+/=]{88})["\']?', 'azure_storage_key_env', 0.95),
        ],
        SecretType.SECRET_KEY: [
            # Azure Client Secret
            (r'AZURE_CLIENT_SECRET\s*=\s*["\']?([A-Za-z0-9~._-]{34,})["\']?', 'azure_client_secret', 0.9),
        ],
        SecretType.CREDENTIAL: [
            # Azure Connection String
            (r'DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=([^;]+);', 'azure_connection_string', 0.95),
        ],
    }

    # GCP patterns
    GCP_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.API_KEY: [
            # GCP API Key
            (r'AIza[0-9A-Za-z_-]{35}', 'gcp_api_key', 0.95),
            (r'GOOGLE_API_KEY\s*=\s*["\']?(AIza[0-9A-Za-z_-]{35})["\']?', 'gcp_api_key_env', 0.95),
        ],
        SecretType.PRIVATE_KEY: [
            # GCP Service Account private key
            (r'"private_key"\s*:\s*"(-----BEGIN[^"]+-----)"', 'gcp_service_account_key', 0.98),
        ],
        SecretType.CREDENTIAL: [
            # GCP OAuth Client Secret
            (r'"client_secret"\s*:\s*"([A-Za-z0-9_-]{24})"', 'gcp_client_secret', 0.9),
        ],
    }

    # Common Python patterns
    PYTHON_COMMON_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.DATABASE_URL: [
            # SQLAlchemy connection strings
            (r'(postgresql://[^:]+:[^@]+@[^\s"\']+)', 'sqlalchemy_postgres', 0.9),
            (r'(mysql://[^:]+:[^@]+@[^\s"\']+)', 'sqlalchemy_mysql', 0.9),
            (r'(sqlite:///[^\s"\']+)', 'sqlalchemy_sqlite', 0.5),
            # Generic database URLs
            (r'SQLALCHEMY_DATABASE_URI\s*=\s*["\']([^"\']+://[^"\']+)["\']', 'sqlalchemy_uri', 0.85),
        ],
        SecretType.API_KEY: [
            # Generic API keys in Python
            (r'api_key\s*=\s*["\']([A-Za-z0-9_-]{20,})["\']', 'python_api_key', 0.8),
            (r'API_KEY\s*=\s*["\']([A-Za-z0-9_-]{20,})["\']', 'python_api_key_const', 0.85),
            # Stripe
            (r'sk_live_[A-Za-z0-9]{24,}', 'stripe_secret_key', 0.98),
            (r'sk_test_[A-Za-z0-9]{24,}', 'stripe_test_key', 0.7),
            # Twilio
            (r'AC[a-f0-9]{32}', 'twilio_account_sid', 0.9),
            # SendGrid
            (r'SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}', 'sendgrid_api_key', 0.95),
            # Mailgun
            (r'key-[a-f0-9]{32}', 'mailgun_api_key', 0.9),
        ],
        SecretType.TOKEN: [
            # GitHub token
            (r'ghp_[A-Za-z0-9]{36}', 'github_personal_token', 0.98),
            (r'gho_[A-Za-z0-9]{36}', 'github_oauth_token', 0.98),
            (r'ghu_[A-Za-z0-9]{36}', 'github_user_token', 0.98),
            (r'ghs_[A-Za-z0-9]{36}', 'github_server_token', 0.98),
            (r'ghr_[A-Za-z0-9]{36}', 'github_refresh_token', 0.98),
            # Slack
            (r'xox[baprs]-[0-9]{10,}-[0-9A-Za-z]+', 'slack_token', 0.95),
            # Discord
            (r'[MN][A-Za-z0-9]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}', 'discord_token', 0.95),
            # PyPI token
            (r'pypi-[A-Za-z0-9_-]{50,}', 'pypi_token', 0.98),
            # npm token
            (r'npm_[A-Za-z0-9]{36}', 'npm_token', 0.95),
        ],
        SecretType.PASSWORD: [
            # Password in Python config
            (r'(?i)password\s*=\s*["\']([^"\']{4,})["\']', 'python_password', 0.7),
            (r'(?i)passwd\s*=\s*["\']([^"\']{4,})["\']', 'python_passwd', 0.7),
            # Redis password
            (r'REDIS_PASSWORD\s*=\s*["\']([^"\']+)["\']', 'redis_password', 0.85),
        ],
        SecretType.SECRET_KEY: [
            # Celery
            (r'CELERY_BROKER_URL\s*=\s*["\']([^"\']+://[^"\']+)["\']', 'celery_broker', 0.8),
            # Session secret
            (r'SESSION_SECRET\s*=\s*["\']([^"\']{16,})["\']', 'session_secret', 0.9),
            # Encryption key
            (r'ENCRYPTION_KEY\s*=\s*["\']([^"\']{16,})["\']', 'encryption_key', 0.95),
            (r'FERNET_KEY\s*=\s*["\']([A-Za-z0-9_-]{44})["\']', 'fernet_key', 0.95),
        ],
        SecretType.PRIVATE_KEY: [
            # SSH private key
            (r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----', 'ssh_private_key', 0.98),
            (r'-----BEGIN PGP PRIVATE KEY BLOCK-----', 'pgp_private_key', 0.98),
        ],
    }

    # .env file patterns
    ENV_FILE_PATTERNS: Dict[SecretType, List[Tuple[str, str, float]]] = {
        SecretType.SECRET_KEY: [
            (r'^SECRET_KEY=(.+)$', 'env_secret_key', 0.9),
            (r'^DJANGO_SECRET_KEY=(.+)$', 'env_django_secret', 0.95),
        ],
        SecretType.DATABASE_URL: [
            (r'^DATABASE_URL=(.+)$', 'env_database_url', 0.9),
            (r'^DB_PASSWORD=(.+)$', 'env_db_password', 0.9),
        ],
        SecretType.API_KEY: [
            (r'^API_KEY=(.+)$', 'env_api_key', 0.85),
            (r'^[A-Z_]+_API_KEY=(.+)$', 'env_generic_api_key', 0.8),
        ],
    }

    @classmethod
    def get_all_patterns(cls) -> Dict[SecretType, List[Tuple[str, str, float]]]:
        """Get all Python security patterns combined"""
        combined = {}

        pattern_sources = [
            cls.DJANGO_PATTERNS,
            cls.FLASK_PATTERNS,
            cls.FASTAPI_PATTERNS,
            cls.AWS_PATTERNS,
            cls.AZURE_PATTERNS,
            cls.GCP_PATTERNS,
            cls.PYTHON_COMMON_PATTERNS,
        ]

        for source in pattern_sources:
            for secret_type, patterns in source.items():
                if secret_type not in combined:
                    combined[secret_type] = []
                combined[secret_type].extend(patterns)

        return combined

    @classmethod
    def get_patterns_for_framework(cls, framework: str) -> Dict[SecretType, List[Tuple[str, str, float]]]:
        """Get patterns specific to a Python framework"""
        patterns = dict(cls.PYTHON_COMMON_PATTERNS)

        framework_lower = framework.lower()
        if framework_lower == 'django':
            for secret_type, pats in cls.DJANGO_PATTERNS.items():
                patterns.setdefault(secret_type, []).extend(pats)
        elif framework_lower == 'flask':
            for secret_type, pats in cls.FLASK_PATTERNS.items():
                patterns.setdefault(secret_type, []).extend(pats)
        elif framework_lower == 'fastapi':
            for secret_type, pats in cls.FASTAPI_PATTERNS.items():
                patterns.setdefault(secret_type, []).extend(pats)

        # Always include cloud provider patterns
        for cloud_patterns in [cls.AWS_PATTERNS, cls.AZURE_PATTERNS, cls.GCP_PATTERNS]:
            for secret_type, pats in cloud_patterns.items():
                patterns.setdefault(secret_type, []).extend(pats)

        return patterns

    @classmethod
    def get_env_patterns(cls) -> Dict[SecretType, List[Tuple[str, str, float]]]:
        """Get patterns for .env files"""
        return cls.ENV_FILE_PATTERNS

    @classmethod
    def compile_patterns(
            cls, patterns: Dict[SecretType, List[Tuple[str, str, float]]]
    ) -> Dict[SecretType, List[Tuple[re.Pattern, str, float]]]:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for secret_type, pattern_list in patterns.items():
            compiled[secret_type] = []
            for pattern, name, confidence in pattern_list:
                try:
                    compiled_pattern = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                    compiled[secret_type].append((compiled_pattern, name, confidence))
                except re.error as e:
                    print(f"Warning: Failed to compile pattern '{name}': {e}")
        return compiled
