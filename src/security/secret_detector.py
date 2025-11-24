import re
import math
from typing import List, Dict, Optional
import structlog

from .models import DetectedSecret, SecretType, SecurityConfig
from .python_patterns import PythonSecurityPatterns


logger = structlog.get_logger(__name__)


class SecretDetector:
    """Detects sensitive information in code using multiple detection methods"""

    def __init__(self, config: SecurityConfig = None):
        if config is None:
            config = SecurityConfig()

        self.config = config
        self._init_patterns()
        self._init_python_patterns()

    def _init_patterns(self):
        """Initialize detection patterns"""
        self.patterns = {
            SecretType.PASSWORD: [
                r'(?i)(password|pwd|pass)\s*[=:]\s*["\']([^"\'\\s]{4,})["\']',
                r'(?i)(password|pwd|pass)\s*[=:]\s*([^\\s]{4,})(?=\\s|$|;|,)',
            ],
            SecretType.API_KEY: [
                r'(?i)(api_key|apikey|key)\s*[=:]\s*["\']([A-Za-z0-9]{20,})["\']',
                r'(?i)(api_key|apikey)\s*[=:]\s*([A-Za-z0-9_-]{20,})(?=\\s|$|;|,)',
            ],
            SecretType.TOKEN: [
                r'(?i)(token|access_token|auth_token)\s*[=:]\s*["\']([A-Za-z0-9_-]{20,})["\']',
                r'(?i)(bearer|token)\s+([A-Za-z0-9_.-]{20,})(?=\\s|$)',
            ],
            SecretType.DATABASE_URL: [
                r'(jdbc:[^\\s"\']+://[^:]+:[^@]+@[^\\s"\']+)',
                r'(mongodb://[^:]+:[^@]+@[^\\s"\']+)',
                r'(postgres://[^:]+:[^@]+@[^\\s"\']+)',
            ],
            SecretType.PRIVATE_KEY: [
                r'-----BEGIN[\\s\\w]*PRIVATE KEY-----[\\s\\S]*?-----END[\\s\\w]*PRIVATE KEY-----',
                r'(?i)(private_key|privatekey)\s*[=:]\s*["\']([^"\']{100,})["\']',
            ],
            SecretType.SECRET_KEY: [
                r'(?i)(secret_key|secretkey|secret)\s*[=:]\s*["\']([A-Za-z0-9_-]{16,})["\']',
                r'(?i)(jwt_secret|session_secret)\s*[=:]\s*([^\\s"\']{16,})(?=\\s|$|;|,)',
            ],
            SecretType.CREDENTIAL: [
                (r'(?i)(username|user)\s*[=:]\s*["\']([^"\']{3,})["\']\s*[,;]?\s*'
                 r'(password|pwd)\s*[=:]\s*["\']([^"\']{3,})["\']'),
                r'(?i)(credential|auth)\s*[=:]\s*["\']([^"\']{10,})["\']',
            ],
            SecretType.HASH: [
                r'\\b([a-fA-F0-9]{32})\\b',  # MD5
                r'\\b([a-fA-F0-9]{40})\\b',  # SHA1
                r'\\b([a-fA-F0-9]{64})\\b',  # SHA256
            ],
            SecretType.EMAIL: [
                r'\\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,})\\b',
            ],
        }

    def detect_secrets(self, content: str, file_path: str = "") -> List[DetectedSecret]:
        """Detect secrets in content using pattern matching"""
        detected = []

        if not self.config.enabled:
            return detected

        # Skip whitelisted files
        if self._is_whitelisted_file(file_path):
            logger.debug("File is whitelisted", file_path=file_path)
            return detected

        lines = content.split('\\n')

        for line_num, line in enumerate(lines, 1):
            # Skip comments if not configured to scan them
            if not self.config.scan_comments and self._is_comment_line(line):
                continue

            detected.extend(self._scan_line(line, line_num, file_path))

        # Use external tools if available
        try:
            external_secrets = self._detect_with_external_tools(content, file_path)
            detected.extend(external_secrets)
        except Exception as e:
            logger.warning("External tool detection failed", error=str(e))

        # Remove duplicates and filter by confidence
        detected = self._filter_and_deduplicate(detected)

        logger.debug("Secrets detected", count=len(detected), file_path=file_path)
        return detected

    def _scan_line(self, line: str, line_num: int, file_path: str) -> List[DetectedSecret]:
        """Scan a single line for secrets"""
        secrets = []

        for secret_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, line)

                for match in matches:
                    # Check if it's a whitelisted pattern
                    if self._is_whitelisted_content(match.group()):
                        continue

                    # Extract the secret content (usually the last group)
                    if match.lastindex and match.lastindex > 0:
                        secret_content = match.group(match.lastindex)
                    else:
                        secret_content = match.group()

                    # Calculate confidence based on pattern strength and content characteristics
                    confidence = self._calculate_confidence(secret_content, secret_type, pattern)

                    if confidence >= self.config.sensitivity_threshold:
                        secret = DetectedSecret(
                            content=secret_content,
                            secret_type=secret_type,
                            confidence=confidence,
                            start_position=match.start(),
                            end_position=match.end(),
                            line_number=line_num,
                            pattern_name=f"{secret_type.value}_pattern",
                            context=line.strip()
                        )
                        secrets.append(secret)

        return secrets

    def _detect_with_external_tools(self, content: str, file_path: str) -> List[DetectedSecret]:
        """Use external tools like truffleHog for detection"""
        detected = []

        # Try detect-secrets first (lighter weight)
        try:
            detected.extend(self._run_detect_secrets(content))
        except Exception as e:
            logger.debug("detect-secrets failed", error=str(e))

        return detected

    def _run_detect_secrets(self, content: str) -> List[DetectedSecret]:
        """Run detect-secrets on content"""
        # This is a simplified implementation
        # In practice, you'd write content to a temp file and run detect-secrets
        # For now, we'll use basic heuristics

        detected = []

        # Look for high-entropy strings
        high_entropy_pattern = r'[A-Za-z0-9+/=]{20,}'

        for match in re.finditer(high_entropy_pattern, content):
            entropy = self._calculate_entropy(match.group())
            if entropy > 4.5:  # High entropy threshold
                detected.append(DetectedSecret(
                    content=match.group(),
                    secret_type=SecretType.GENERIC,
                    confidence=min(entropy / 6.0, 1.0),
                    start_position=match.start(),
                    end_position=match.end(),
                    line_number=content[:match.start()].count('\\n') + 1,
                    pattern_name="high_entropy"
                ))

        return detected

    def _calculate_confidence(self, content: str, secret_type: SecretType, pattern: str) -> float:
        """Calculate confidence score for detected secret"""
        confidence = 0.5  # Base confidence

        # Length-based scoring
        if len(content) >= 20:
            confidence += 0.2
        elif len(content) >= 10:
            confidence += 0.1

        # Entropy-based scoring
        entropy = self._calculate_entropy(content)
        if entropy > 4.0:
            confidence += 0.2
        elif entropy > 3.0:
            confidence += 0.1

        # Type-specific scoring
        type_bonus = {
            SecretType.PRIVATE_KEY: 0.3,
            SecretType.DATABASE_URL: 0.2,
            SecretType.API_KEY: 0.15,
            SecretType.SECRET_KEY: 0.15,
            SecretType.TOKEN: 0.1,
            SecretType.PASSWORD: 0.05
        }

        confidence += type_bonus.get(secret_type, 0.0)

        return min(confidence, 1.0)

    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0.0

        # Count frequency of each character
        frequency = {}
        for char in string:
            frequency[char] = frequency.get(char, 0) + 1

        # Calculate Shannon entropy: H = -Σ(p * log2(p))
        entropy = 0.0
        string_length = len(string)

        for count in frequency.values():
            probability = count / string_length
            if probability > 0:  # Avoid log(0)
                entropy -= probability * math.log2(probability)

        return entropy

    def _is_comment_line(self, line: str) -> bool:
        """Check if line is a comment"""
        stripped = line.strip()
        return (stripped.startswith('//') or
                stripped.startswith('/*') or
                stripped.startswith('*') or
                stripped.startswith('#') or
                stripped.startswith('<!--'))

    def _is_whitelisted_file(self, file_path: str) -> bool:
        """Check if file is in whitelist"""
        for pattern in self.config.whitelist_files:
            if self._matches_pattern(file_path, pattern):
                return True
        return False

    def _is_whitelisted_content(self, content: str) -> bool:
        """Check if content matches whitelist patterns"""
        content_lower = content.lower()
        for pattern in self.config.whitelist_patterns:
            if pattern.lower() in content_lower:
                return True
        return False

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches glob-like pattern"""
        # Convert glob pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(regex_pattern, text))

    def _filter_and_deduplicate(self, secrets: List[DetectedSecret]) -> List[DetectedSecret]:
        """Filter secrets by confidence and remove duplicates"""
        # Filter by confidence threshold
        filtered = [s for s in secrets if s.confidence >= self.config.sensitivity_threshold]

        # Remove duplicates based on content and position
        seen = set()
        deduplicated = []

        for secret in filtered:
            key = (secret.content, secret.line_number, secret.start_position)
            if key not in seen:
                seen.add(key)
                deduplicated.append(secret)

        return deduplicated

    def _init_python_patterns(self):
        """Initialize Python-specific detection patterns"""
        self.python_patterns = PythonSecurityPatterns.compile_patterns(
            PythonSecurityPatterns.get_all_patterns()
        )
        self.env_patterns = PythonSecurityPatterns.compile_patterns(
            PythonSecurityPatterns.get_env_patterns()
        )

    def detect_python_secrets(self, content: str, file_path: str = "",
                              framework: Optional[str] = None) -> List[DetectedSecret]:
        """Detect secrets in Python code with framework-specific patterns"""
        detected = []

        if not self.config.enabled:
            return detected

        # Skip whitelisted files
        if self._is_whitelisted_file(file_path):
            logger.debug("File is whitelisted", file_path=file_path)
            return detected

        # Determine which patterns to use
        if file_path.endswith('.env') or '.env.' in file_path:
            patterns = self.env_patterns
        elif framework:
            patterns = PythonSecurityPatterns.compile_patterns(
                PythonSecurityPatterns.get_patterns_for_framework(framework)
            )
        else:
            patterns = self.python_patterns

        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Skip comments if not configured to scan them
            if not self.config.scan_comments and self._is_python_comment(line):
                continue

            detected.extend(self._scan_line_python(line, line_num, file_path, patterns))

        # Also run standard detection
        standard_secrets = self.detect_secrets(content, file_path)
        detected.extend(standard_secrets)

        # Remove duplicates and filter by confidence
        detected = self._filter_and_deduplicate(detected)

        logger.debug("Python secrets detected", count=len(detected), file_path=file_path)
        return detected

    def _scan_line_python(self, line: str, line_num: int, file_path: str,
                          patterns: Dict) -> List[DetectedSecret]:
        """Scan a single line for Python-specific secrets"""
        secrets = []

        for secret_type, pattern_list in patterns.items():
            for compiled_pattern, pattern_name, base_confidence in pattern_list:
                matches = compiled_pattern.finditer(line)

                for match in matches:
                    # Check if it's a whitelisted pattern
                    if self._is_whitelisted_content(match.group()):
                        continue

                    # Extract the secret content
                    if match.groups():
                        secret_content = match.group(1)
                    else:
                        secret_content = match.group()

                    # Skip very short matches
                    if len(secret_content) < 4:
                        continue

                    # Calculate final confidence
                    confidence = self._adjust_python_confidence(
                        secret_content, secret_type, base_confidence, file_path
                    )

                    if confidence >= self.config.sensitivity_threshold:
                        secret = DetectedSecret(
                            content=secret_content,
                            secret_type=secret_type,
                            confidence=confidence,
                            start_position=match.start(),
                            end_position=match.end(),
                            line_number=line_num,
                            pattern_name=pattern_name,
                            context=line.strip()
                        )
                        secrets.append(secret)

        return secrets

    def _adjust_python_confidence(self, content: str, secret_type: SecretType,
                                  base_confidence: float, file_path: str) -> float:
        """Adjust confidence based on Python-specific heuristics"""
        confidence = base_confidence

        # Increase confidence for production-like files
        if 'settings' in file_path.lower() or 'config' in file_path.lower():
            confidence = min(confidence + 0.1, 1.0)

        # Decrease confidence for test files
        if 'test' in file_path.lower() or 'mock' in file_path.lower():
            confidence = max(confidence - 0.2, 0.0)

        # Decrease for example/sample values
        if any(x in content.lower() for x in ['example', 'sample', 'placeholder', 'xxx', 'your-']):
            confidence = max(confidence - 0.3, 0.0)

        # Increase for high-entropy content
        entropy = self._calculate_entropy(content)
        if entropy > 4.5:
            confidence = min(confidence + 0.1, 1.0)

        return confidence

    def _is_python_comment(self, line: str) -> bool:
        """Check if line is a Python comment"""
        stripped = line.strip()
        return stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")
