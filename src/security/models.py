from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class SensitivityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SecretType(Enum):
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CREDENTIAL = "credential"
    DATABASE_URL = "database_url"
    PRIVATE_KEY = "private_key"
    SECRET_KEY = "secret_key"
    CERTIFICATE = "certificate"
    HASH = "hash"
    EMAIL = "email"
    PHONE = "phone"
    GENERIC = "generic"


@dataclass
class DetectedSecret:
    """Represents a detected sensitive information"""
    content: str
    secret_type: SecretType
    confidence: float
    start_position: int
    end_position: int
    line_number: int
    pattern_name: str
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "secret_type": self.secret_type.value,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "line_number": self.line_number,
            "pattern_name": self.pattern_name,
            "context": self.context
        }


@dataclass
class MaskingResult:
    """Result of masking operation"""
    original_content: str
    masked_content: str
    detected_secrets: List[DetectedSecret]
    sensitivity_level: SensitivityLevel
    masked_count: int
    preserve_syntax: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "masked_content": self.masked_content,
            "detected_secrets": [s.to_dict() for s in self.detected_secrets],
            "sensitivity_level": self.sensitivity_level.value,
            "masked_count": self.masked_count,
            "preserve_syntax": self.preserve_syntax
        }


@dataclass
class SecurityConfig:
    """Configuration for security scanning and masking"""
    enabled: bool = True
    preserve_syntax: bool = True
    sensitivity_threshold: float = 0.7
    placeholder_format: str = "[MASKED_{type}_{index}]"
    scan_comments: bool = True
    scan_strings: bool = True

    # Whitelist patterns
    whitelist_patterns: List[str] = None
    whitelist_files: List[str] = None

    # Custom patterns
    custom_patterns: Dict[str, str] = None

    def __post_init__(self):
        if self.whitelist_patterns is None:
            self.whitelist_patterns = [
                "test_password",
                "example_key",
                "dummy_token",
                "placeholder_secret",
                "fake_api_key",
                "sample_credential"
            ]

        if self.whitelist_files is None:
            self.whitelist_files = [
                "test/**/*",
                "**/test/**/*",
                "**/*Test.java",
                "**/*Spec.java",
                "**/example/**/*"
            ]

        if self.custom_patterns is None:
            self.custom_patterns = {}
