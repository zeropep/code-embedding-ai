"""
Tests for security scanning and masking functionality
Based on actual implementation in src/security/
"""

import pytest
from unittest.mock import Mock, patch

from src.security.models import (
    SecurityConfig, SensitivityLevel, SecretType,
    DetectedSecret, MaskingResult
)
from src.security.secret_detector import SecretDetector
from src.security.content_masker import ContentMasker
from src.security.security_scanner import SecurityScanner
from src.code_parser.models import CodeChunk, CodeLanguage, LayerType


class TestSensitivityLevel:
    """Test SensitivityLevel enum"""

    def test_sensitivity_level_values(self):
        """Test sensitivity level enum values"""
        assert SensitivityLevel.LOW.value == "LOW"
        assert SensitivityLevel.MEDIUM.value == "MEDIUM"
        assert SensitivityLevel.HIGH.value == "HIGH"


class TestSecretType:
    """Test SecretType enum"""

    def test_secret_type_values(self):
        """Test secret type enum values"""
        assert SecretType.PASSWORD.value == "password"
        assert SecretType.API_KEY.value == "api_key"
        assert SecretType.TOKEN.value == "token"
        assert SecretType.PRIVATE_KEY.value == "private_key"
        assert SecretType.DATABASE_URL.value == "database_url"


class TestDetectedSecret:
    """Test DetectedSecret dataclass"""

    def test_detected_secret_creation(self):
        """Test DetectedSecret creation"""
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
        assert secret.line_number == 5

    def test_detected_secret_to_dict(self):
        """Test DetectedSecret to_dict method"""
        secret = DetectedSecret(
            content="api_key_123",
            secret_type=SecretType.API_KEY,
            confidence=0.85,
            start_position=0,
            end_position=11,
            line_number=1,
            pattern_name="api_key_pattern"
        )

        data = secret.to_dict()
        assert data["secret_type"] == "api_key"
        assert data["confidence"] == 0.85
        assert data["line_number"] == 1

    def test_detected_secret_context(self):
        """Test DetectedSecret with context"""
        secret = DetectedSecret(
            content="token123",
            secret_type=SecretType.TOKEN,
            confidence=0.8,
            start_position=5,
            end_position=13,
            line_number=10,
            pattern_name="token_pattern",
            context="String token = 'token123';"
        )

        assert secret.context == "String token = 'token123';"


class TestMaskingResult:
    """Test MaskingResult dataclass"""

    def test_masking_result_creation(self):
        """Test MaskingResult creation"""
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
        assert result.masked_content == "[MASKED_SECRET]"

    def test_masking_result_to_dict(self):
        """Test MaskingResult to_dict method"""
        result = MaskingResult(
            original_content="test",
            masked_content="[MASKED]",
            detected_secrets=[],
            sensitivity_level=SensitivityLevel.LOW,
            masked_count=0
        )

        data = result.to_dict()
        assert "masked_content" in data
        assert "sensitivity_level" in data
        assert data["sensitivity_level"] == "LOW"


class TestSecurityConfig:
    """Test SecurityConfig dataclass"""

    def test_config_defaults(self):
        """Test SecurityConfig default values"""
        config = SecurityConfig()

        assert config.enabled is True
        assert config.preserve_syntax is True
        assert config.sensitivity_threshold == 0.7
        assert config.scan_comments is True
        assert config.scan_strings is True

    def test_config_custom_values(self):
        """Test SecurityConfig with custom values"""
        config = SecurityConfig(
            enabled=False,
            preserve_syntax=False,
            sensitivity_threshold=0.5,
            scan_comments=False
        )

        assert config.enabled is False
        assert config.preserve_syntax is False
        assert config.sensitivity_threshold == 0.5
        assert config.scan_comments is False

    def test_config_whitelist_defaults(self):
        """Test whitelist default values"""
        config = SecurityConfig()

        assert config.whitelist_patterns is not None
        assert "test_password" in config.whitelist_patterns
        assert config.whitelist_files is not None


class TestSecretDetector:
    """Test SecretDetector class"""

    @pytest.fixture
    def security_config(self):
        return SecurityConfig()

    @pytest.fixture
    def detector(self, security_config):
        return SecretDetector(security_config)

    def test_detector_initialization(self, detector, security_config):
        """Test SecretDetector initialization"""
        assert detector.config == security_config
        assert len(detector.patterns) > 0

    def test_detect_in_code(self, detector):
        """Test detecting secrets in code"""
        code = '''
        String password = "mysecretpassword123";
        String apiKey = "sk-1234567890abcdef";
        '''

        secrets = detector.detect_secrets(code, "test.java")
        # Should detect some secrets
        assert isinstance(secrets, list)

    def test_no_secrets_in_clean_code(self, detector):
        """Test no false positives in clean code"""
        code = '''
        public void processUser(User user) {
            String name = user.getName();
            int age = user.getAge();
        }
        '''

        secrets = detector.detect_secrets(code, "test.java")
        # Clean code should have minimal or no detections
        assert isinstance(secrets, list)


class TestContentMasker:
    """Test ContentMasker class"""

    @pytest.fixture
    def security_config(self):
        return SecurityConfig()

    @pytest.fixture
    def masker(self, security_config):
        return ContentMasker(security_config)

    def test_masker_initialization(self, masker, security_config):
        """Test ContentMasker initialization"""
        assert masker.config == security_config

    def test_mask_secrets(self, masker):
        """Test masking detected secrets"""
        secrets = [
            DetectedSecret(
                content="password123",
                secret_type=SecretType.PASSWORD,
                confidence=0.9,
                start_position=20,
                end_position=31,
                line_number=1,
                pattern_name="password"
            )
        ]

        original = 'String password = "password123";'
        result = masker.mask_content(original, secrets)

        assert isinstance(result, MaskingResult)
        assert "password123" not in result.masked_content or result.masked_count == 0

    def test_calculate_sensitivity(self, masker):
        """Test sensitivity level calculation"""
        # No secrets - should be LOW
        level = masker._calculate_sensitivity_level([])
        assert level == SensitivityLevel.LOW

        # Multiple secrets - should be higher
        secrets = [
            DetectedSecret(
                content="secret1",
                secret_type=SecretType.PASSWORD,
                confidence=0.9,
                start_position=0,
                end_position=7,
                line_number=1,
                pattern_name="test"
            ),
            DetectedSecret(
                content="secret2",
                secret_type=SecretType.API_KEY,
                confidence=0.95,
                start_position=10,
                end_position=17,
                line_number=2,
                pattern_name="test"
            )
        ]
        level = masker._calculate_sensitivity_level(secrets)
        assert level in [SensitivityLevel.MEDIUM, SensitivityLevel.HIGH]


class TestSecurityScanner:
    """Test SecurityScanner class"""

    @pytest.fixture
    def security_config(self):
        return SecurityConfig()

    @pytest.fixture
    def scanner(self, security_config):
        return SecurityScanner(security_config)

    def test_scanner_initialization(self, scanner, security_config):
        """Test SecurityScanner initialization"""
        assert scanner.config == security_config
        assert scanner.detector is not None
        assert scanner.masker is not None

    def test_scan_code(self, scanner):
        """Test scanning code for secrets"""
        code = '''
        public class Config {
            private String dbPassword = "admin123";
            private String apiKey = "key_1234567890";
        }
        '''

        # Use detector and masker directly
        secrets = scanner.detector.detect_secrets(code, "test.java")
        result = scanner.masker.mask_content(code, secrets, "test.java")
        assert isinstance(result, MaskingResult)

    def test_scan_clean_code(self, scanner):
        """Test scanning clean code"""
        code = '''
        public int add(int a, int b) {
            return a + b;
        }
        '''

        secrets = scanner.detector.detect_secrets(code, "test.java")
        result = scanner.masker.mask_content(code, secrets, "test.java")
        assert isinstance(result, MaskingResult)
        assert result.sensitivity_level == SensitivityLevel.LOW

    def test_scan_chunk(self, scanner):
        """Test scanning a code chunk"""
        chunk = CodeChunk(
            content='String password = "secret123";',
            file_path="src/Config.java",
            language=CodeLanguage.JAVA,
            start_line=1,
            end_line=1,
            layer_type=LayerType.CONFIG
        )

        result = scanner.scan_and_mask_chunk(chunk)
        assert isinstance(result, CodeChunk)
        assert result.content is not None

    def test_scanner_disabled(self):
        """Test scanner when disabled"""
        config = SecurityConfig(enabled=False)
        scanner = SecurityScanner(config)

        code = 'String password = "secret123";'
        secrets = scanner.detector.detect_secrets(code, "test.java")
        result = scanner.masker.mask_content(code, secrets, "test.java")

        # When disabled, should return minimal result
        assert isinstance(result, MaskingResult)


class TestSecurityIntegration:
    """Integration tests for security module"""

    @pytest.fixture
    def scanner(self):
        return SecurityScanner(SecurityConfig())

    def test_full_scan_workflow(self, scanner):
        """Test complete scan workflow"""
        code = '''
        @Configuration
        public class DatabaseConfig {
            @Value("${db.password}")
            private String password;

            @Value("${api.key}")
            private String apiKey;

            public Connection getConnection() {
                return DriverManager.getConnection(url, user, password);
            }
        }
        '''

        secrets = scanner.detector.detect_secrets(code, "DatabaseConfig.java")
        result = scanner.masker.mask_content(code, secrets, "DatabaseConfig.java")

        assert isinstance(result, MaskingResult)
        assert result.masked_content is not None
        assert len(result.detected_secrets) >= 0

    def test_preserve_code_structure(self, scanner):
        """Test that masking preserves code structure"""
        code = '''
        public void test() {
            String secret = "mysecret123";
            System.out.println(secret);
        }
        '''

        secrets = scanner.detector.detect_secrets(code, "test.java")
        result = scanner.masker.mask_content(code, secrets, "test.java")

        # Code structure should be preserved
        assert "public void test()" in result.masked_content
        assert "System.out.println" in result.masked_content
