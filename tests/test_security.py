"""
Tests for security scanning and masking functionality
"""

import pytest
from unittest.mock import Mock, patch

from src.security.models import SecurityConfig, SensitivityLevel, SecretType
from src.security.secret_detector import SecretDetector
from src.security.content_masker import ContentMasker
from src.security.security_scanner import SecurityScanner
from src.code_parser.models import CodeChunk, CodeLanguage, LayerType


class TestSecurityModels:
    """Test security model classes"""

    def test_detected_secret_creation(self):
        """Test DetectedSecret creation"""
        from src.security.models import DetectedSecret

        secret = DetectedSecret(
            content="password123",
            secret_type=SecretType.PASSWORD,
            confidence=0.9,
            start_position=10,
            end_position=21,
            line_number=5,
            pattern_name="password_pattern"
        )

        assert secret.content == "password123"
        assert secret.secret_type == SecretType.PASSWORD
        assert secret.confidence == 0.9

        secret_dict = secret.to_dict()
        assert secret_dict["secret_type"] == "password"

    def test_masking_result(self):
        """Test MaskingResult model"""
        from src.security.models import MaskingResult, DetectedSecret

        secrets = [DetectedSecret(
            content="secret123",
            secret_type=SecretType.SECRET_KEY,
            confidence=0.8,
            start_position=0,
            end_position=9,
            line_number=1,
            pattern_name="test_pattern"
        )]

        result = MaskingResult(
            original_content="secret123",
            masked_content="[MASKED_SECRET]",
            detected_secrets=secrets,
            sensitivity_level=SensitivityLevel.HIGH,
            masked_count=1
        )

        assert result.masked_count == 1
        assert result.sensitivity_level == SensitivityLevel.HIGH

        result_dict = result.to_dict()
        assert "masked_content" in result_dict
        assert "sensitivity_level" in result_dict


class TestSecretDetector:
    """Test secret detection functionality"""

    def test_secret_detector_initialization(self, security_config):
        """Test SecretDetector initialization"""
        detector = SecretDetector(security_config)

        assert detector.config == security_config
        assert len(detector.patterns) > 0

    def test_detect_passwords(self, security_config):
        """Test password detection"""
        detector = SecretDetector(security_config)

        code_with_password = """
        public class Config {
            private String password = "mySecretPassword123";
            private String pwd = "another_password";
        }
        """

        secrets = detector.detect_secrets(code_with_password)

        password_secrets = [s for s in secrets if s.secret_type == SecretType.PASSWORD]
        assert len(password_secrets) > 0

        # Check confidence scores
        for secret in password_secrets:
            assert secret.confidence >= security_config.sensitivity_threshold

    def test_detect_api_keys(self, security_config):
        """Test API key detection"""
        detector = SecretDetector(security_config)

        code_with_api_key = """
        public class ApiClient {
            private static final String API_KEY = "sk-1234567890abcdefghij";
            private String apikey = "ak_test_123456789012345";
        }
        """

        secrets = detector.detect_secrets(code_with_api_key)

        api_key_secrets = [s for s in secrets if s.secret_type == SecretType.API_KEY]
        assert len(api_key_secrets) > 0

    def test_detect_database_urls(self, security_config):
        """Test database URL detection"""
        detector = SecretDetector(security_config)

        code_with_db_url = """
        String dbUrl = "jdbc:mysql://username:password@localhost:3306/database";
        String mongoUrl = "mongodb://admin:secret@mongo.example.com:27017/db";
        """

        secrets = detector.detect_secrets(code_with_db_url)

        db_secrets = [s for s in secrets if s.secret_type == SecretType.DATABASE_URL]
        assert len(db_secrets) > 0

    def test_whitelist_filtering(self):
        """Test whitelist pattern filtering"""
        config = SecurityConfig(
            sensitivity_threshold=0.5,
            whitelist_patterns=["test_password", "example_key"]
        )
        detector = SecretDetector(config)

        code_with_whitelist = """
        String testPassword = "test_password";
        String exampleKey = "example_key_123";
        String realSecret = "actualSecretPassword";
        """

        secrets = detector.detect_secrets(code_with_whitelist)

        # Should not detect whitelisted items
        for secret in secrets:
            assert "test_password" not in secret.content.lower()
            assert "example_key" not in secret.content.lower()

    def test_comment_scanning_disabled(self):
        """Test comment scanning can be disabled"""
        config = SecurityConfig(scan_comments=False)
        detector = SecretDetector(config)

        code_with_comment_secret = """
        // Password: secretPassword123
        public void method() {
            String password = "realPassword456";
        }
        """

        secrets = detector.detect_secrets(code_with_comment_secret)

        # Should only detect the non-comment password
        assert len(secrets) >= 1
        comment_secrets = [s for s in secrets if "secretPassword123" in s.content]
        assert len(comment_secrets) == 0


class TestContentMasker:
    """Test content masking functionality"""

    def test_content_masker_initialization(self, security_config):
        """Test ContentMasker initialization"""
        masker = ContentMasker(security_config)

        assert masker.config == security_config
        assert len(masker.masking_rules) > 0

    def test_mask_passwords(self, security_config):
        """Test password masking"""
        masker = ContentMasker(security_config)

        from src.security.models import DetectedSecret

        content = 'String password = "myPassword123";'
        secrets = [DetectedSecret(
            content="myPassword123",
            secret_type=SecretType.PASSWORD,
            confidence=0.9,
            start_position=18,
            end_position=31,
            line_number=1,
            pattern_name="password_pattern",
            context=content
        )]

        result = masker.mask_content(content, secrets)

        assert result.masked_count == 1
        assert "[MASKED_PASSWORD]" in result.masked_content
        assert "myPassword123" not in result.masked_content

    def test_mask_api_keys(self, security_config):
        """Test API key masking"""
        masker = ContentMasker(security_config)

        from src.security.models import DetectedSecret

        content = 'public static final String API_KEY = "sk-1234567890abcdef";'
        secrets = [DetectedSecret(
            content="sk-1234567890abcdef",
            secret_type=SecretType.API_KEY,
            confidence=0.9,
            start_position=38,
            end_position=56,
            line_number=1,
            pattern_name="api_key_pattern",
            context=content
        )]

        result = masker.mask_content(content, secrets)

        assert result.masked_count == 1
        assert "[MASKED_API_KEY]" in result.masked_content
        assert "sk-1234567890abcdef" not in result.masked_content

    def test_preserve_syntax(self, security_config):
        """Test syntax preservation during masking"""
        masker = ContentMasker(security_config)

        from src.security.models import DetectedSecret

        content = 'String password = "secretValue";'
        secrets = [DetectedSecret(
            content="secretValue",
            secret_type=SecretType.PASSWORD,
            confidence=0.9,
            start_position=19,
            end_position=30,
            line_number=1,
            pattern_name="password_pattern",
            context=content
        )]

        result = masker.mask_content(content, secrets)

        # Should preserve quotes and semicolon
        assert result.masked_content.endswith('";')
        assert '"' in result.masked_content

    def test_database_url_masking(self, security_config):
        """Test database URL masking with custom logic"""
        masker = ContentMasker(security_config)

        from src.security.models import DetectedSecret

        content = 'String url = "jdbc:mysql://user:pass@localhost:3306/db";'
        secrets = [DetectedSecret(
            content="jdbc:mysql://user:pass@localhost:3306/db",
            secret_type=SecretType.DATABASE_URL,
            confidence=0.9,
            start_position=14,
            end_position=54,
            line_number=1,
            pattern_name="db_url_pattern",
            context=content
        )]

        result = masker.mask_content(content, secrets)

        # Should mask credentials but preserve host
        assert "localhost" in result.masked_content or "[HOST]" in result.masked_content
        assert "user:pass" not in result.masked_content

    def test_sensitivity_level_calculation(self, security_config):
        """Test sensitivity level calculation"""
        masker = ContentMasker(security_config)

        from src.security.models import DetectedSecret

        # High sensitivity - private key
        high_secrets = [DetectedSecret(
            content="-----BEGIN PRIVATE KEY-----",
            secret_type=SecretType.PRIVATE_KEY,
            confidence=0.95,
            start_position=0,
            end_position=27,
            line_number=1,
            pattern_name="private_key_pattern"
        )]

        result = masker.mask_content("test content", high_secrets)
        assert result.sensitivity_level == SensitivityLevel.HIGH

        # Low sensitivity - regular content
        no_secrets = []
        result = masker.mask_content("regular code", no_secrets)
        assert result.sensitivity_level == SensitivityLevel.LOW


class TestSecurityScanner:
    """Test security scanner orchestrator"""

    def test_security_scanner_initialization(self, security_config):
        """Test SecurityScanner initialization"""
        scanner = SecurityScanner(security_config)

        assert scanner.config == security_config
        assert scanner.detector is not None
        assert scanner.masker is not None

    def test_scan_and_mask_chunk(self, security_config, sample_code_with_secrets):
        """Test scanning and masking a code chunk"""
        scanner = SecurityScanner(security_config)

        chunk = CodeChunk(
            content=sample_code_with_secrets,
            file_path="src/main/java/DatabaseConfig.java",
            language=CodeLanguage.JAVA,
            start_line=1,
            end_line=10,
            class_name="DatabaseConfig",
            layer_type=LayerType.CONFIG,
            metadata={}
        )

        masked_chunk = scanner.scan_and_mask_chunk(chunk)

        # Should have security metadata
        assert "security" in masked_chunk.metadata
        security_meta = masked_chunk.metadata["security"]

        assert "sensitivity_level" in security_meta
        assert "secrets_masked" in security_meta
        assert "detected_secrets" in security_meta

        # Content should be masked
        assert masked_chunk.content != chunk.content
        assert "secretPassword123" not in masked_chunk.content

    def test_scan_chunks_batch(self, security_config, create_test_chunks):
        """Test scanning multiple chunks"""
        scanner = SecurityScanner(security_config)

        chunks = create_test_chunks(3)
        # Add secrets to one chunk
        chunks[1].content = 'String apiKey = "sk-secret123456789";'

        masked_chunks = scanner.scan_chunks(chunks)

        assert len(masked_chunks) == 3

        # Check that the chunk with secrets has security metadata
        secret_chunks = [c for c in masked_chunks if "security" in c.metadata]
        assert len(secret_chunks) >= 1

    def test_generate_security_report(self, security_config, create_test_chunks):
        """Test security report generation"""
        scanner = SecurityScanner(security_config)

        chunks = create_test_chunks(5)

        # Add security metadata to simulate scanned chunks
        chunks[0].metadata["security"] = {
            "sensitivity_level": SensitivityLevel.HIGH.value,
            "secrets_masked": 2,
            "detected_secrets": [
                {"secret_type": "password", "confidence": 0.9},
                {"secret_type": "api_key", "confidence": 0.8}
            ]
        }

        chunks[1].metadata["security"] = {
            "sensitivity_level": SensitivityLevel.MEDIUM.value,
            "secrets_masked": 1,
            "detected_secrets": [
                {"secret_type": "token", "confidence": 0.7}
            ]
        }

        report = scanner.generate_security_report(chunks)

        assert "scan_summary" in report
        assert "secret_types_found" in report
        assert "high_risk_files" in report
        assert "recommendations" in report

        # Check summary statistics
        summary = report["scan_summary"]
        assert summary["total_secrets_found"] >= 3
        assert summary["sensitivity_distribution"][SensitivityLevel.HIGH.value] >= 1

    def test_disabled_security_scanning(self):
        """Test behavior when security scanning is disabled"""
        config = SecurityConfig(enabled=False)
        scanner = SecurityScanner(config)

        chunk = CodeChunk(
            content='String password = "secret123";',
            file_path="test.java",
            language=CodeLanguage.JAVA,
            start_line=1,
            end_line=1,
            metadata={}
        )

        masked_chunk = scanner.scan_and_mask_chunk(chunk)

        # Content should be unchanged
        assert masked_chunk.content == chunk.content
        # No security metadata should be added
        assert "security" not in masked_chunk.metadata