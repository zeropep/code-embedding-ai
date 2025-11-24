import re
from typing import List, Dict, Tuple, Optional
import structlog

from .models import DetectedSecret, MaskingResult, SensitivityLevel, SecretType, SecurityConfig


logger = structlog.get_logger(__name__)


class ContentMasker:
    """Masks sensitive content while preserving code syntax"""

    def __init__(self, config: SecurityConfig = None):
        if config is None:
            config = SecurityConfig()

        self.config = config
        self._init_masking_rules()

    def _init_masking_rules(self):
        """Initialize masking rules for different secret types"""
        self.masking_rules = {
            SecretType.PASSWORD: {
                'placeholder': '[MASKED_PASSWORD]',
                'preserve_quotes': True,
                'min_length': 4
            },
            SecretType.API_KEY: {
                'placeholder': '[MASKED_API_KEY]',
                'preserve_quotes': True,
                'min_length': 20
            },
            SecretType.TOKEN: {
                'placeholder': '[MASKED_TOKEN]',
                'preserve_quotes': True,
                'min_length': 16
            },
            SecretType.DATABASE_URL: {
                'placeholder': '[MASKED_DB_URL]',
                'preserve_quotes': True,
                'custom_mask': self._mask_database_url
            },
            SecretType.PRIVATE_KEY: {
                'placeholder': '[MASKED_PRIVATE_KEY]',
                'preserve_quotes': False,
                'preserve_structure': True
            },
            SecretType.SECRET_KEY: {
                'placeholder': '[MASKED_SECRET]',
                'preserve_quotes': True,
                'min_length': 16
            },
            SecretType.CREDENTIAL: {
                'placeholder': '[MASKED_CREDENTIAL]',
                'preserve_quotes': True,
                'custom_mask': self._mask_credentials
            },
            SecretType.HASH: {
                'placeholder': '[MASKED_HASH]',
                'preserve_quotes': True,
                'min_length': 32
            },
            SecretType.EMAIL: {
                'placeholder': '[MASKED_EMAIL]',
                'preserve_quotes': True,
                'custom_mask': self._mask_email
            }
        }

    def mask_content(self, content: str, detected_secrets: List[DetectedSecret],
                     file_path: str = "") -> MaskingResult:
        """Mask all detected secrets in content"""
        if not self.config.enabled or not detected_secrets:
            return MaskingResult(
                original_content=content,
                masked_content=content,
                detected_secrets=[],
                sensitivity_level=SensitivityLevel.LOW,
                masked_count=0
            )

        masked_content = content
        masked_count = 0
        processed_secrets = []

        # Sort secrets by position (reverse order to maintain positions)
        sorted_secrets = sorted(detected_secrets,
                                key=lambda x: (x.line_number, x.start_position),
                                reverse=True)

        lines = masked_content.split('\n')

        for secret in sorted_secrets:
            try:
                masked_line, was_masked = self._mask_secret_in_line(
                    lines[secret.line_number - 1], secret
                )
                if was_masked:
                    lines[secret.line_number - 1] = masked_line
                    masked_count += 1
                    processed_secrets.append(secret)

            except Exception as e:
                logger.warning("Failed to mask secret",
                               error=str(e),
                               secret_type=secret.secret_type.value,
                               line=secret.line_number)

        masked_content = '\n'.join(lines)

        # Determine overall sensitivity level
        sensitivity_level = self._calculate_sensitivity_level(processed_secrets)

        logger.debug("Content masking completed",
                     file_path=file_path,
                     secrets_detected=len(detected_secrets),
                     secrets_masked=masked_count,
                     sensitivity=sensitivity_level.value)

        return MaskingResult(
            original_content=content,
            masked_content=masked_content,
            detected_secrets=processed_secrets,
            sensitivity_level=sensitivity_level,
            masked_count=masked_count,
            preserve_syntax=self.config.preserve_syntax
        )

    def _mask_secret_in_line(self, line: str, secret: DetectedSecret) -> Tuple[str, bool]:
        """Mask a specific secret in a line"""
        rule = self.masking_rules.get(secret.secret_type)
        if not rule:
            return line, False

        # Use custom masking function if available
        if 'custom_mask' in rule:
            return rule['custom_mask'](line, secret)

        # Standard masking
        return self._apply_standard_mask(line, secret, rule)

    def _apply_standard_mask(self, line: str, secret: DetectedSecret,
                             rule: Dict) -> Tuple[str, bool]:
        """Apply standard masking rules"""
        placeholder = rule['placeholder']

        # Find the secret in the line
        secret.start_position
        secret.end_position

        # Adjust positions relative to the line
        line_start = secret.context.find(secret.content)
        if line_start == -1:
            # Fallback: find by content
            line_start = line.find(secret.content)
            if line_start == -1:
                return line, False

        line_end = line_start + len(secret.content)

        # Preserve quotes if needed
        if rule.get('preserve_quotes', False):
            # Check if secret is within quotes
            before_secret = line[:line_start]
            after_secret = line[line_end:]

            quote_before = self._get_quote_char(before_secret, reverse=True)
            quote_after = self._get_quote_char(after_secret)

            if quote_before and quote_after and quote_before == quote_after:
                # Secret is quoted, just replace the content
                masked_line = line[:line_start] + placeholder + line[line_end:]
            else:
                # Not quoted or mismatched quotes, add quotes to placeholder
                masked_line = (line[:line_start] +
                               f'"{placeholder}"' +
                               line[line_end:])
        else:
            # Simple replacement
            masked_line = line[:line_start] + placeholder + line[line_end:]

        return masked_line, True

    def _mask_database_url(self, line: str, secret: DetectedSecret) -> Tuple[str, bool]:
        """Custom masking for database URLs"""
        db_url_pattern = r'(jdbc:|mongodb:|postgres://)[^:]+:[^@]+@([^/\s"\']+)'

        def replace_credentials(match):
            protocol = match.group(1)
            host = match.group(2) if len(match.groups()) > 1 else '[HOST]'
            return f'{protocol}[MASKED_USER]:[MASKED_PASSWORD]@{host}'

        masked_line = re.sub(db_url_pattern, replace_credentials, line)
        return masked_line, masked_line != line

    def _mask_credentials(self, line: str, secret: DetectedSecret) -> Tuple[str, bool]:
        """Custom masking for username/password pairs"""
        # Pattern to find username/password pairs
        cred_pattern = (r'((?i)(username|user)\s*[=:]\s*["\'])[^"\']+'
                        r'(["\'])\s*[,;]?\s*((?i)(password|pwd)\s*[=:]\s*["\'])[^"\']+(["\'])')

        def replace_creds(match):
            return f'{match.group(1)}[MASKED_USER]{match.group(3)}, {match.group(4)}[MASKED_PASSWORD]{match.group(6)}'

        masked_line = re.sub(cred_pattern, replace_creds, line)
        return masked_line, masked_line != line

    def _mask_email(self, line: str, secret: DetectedSecret) -> Tuple[str, bool]:
        """Custom masking for email addresses"""
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'

        def replace_email(match):
            email = match.group()
            domain = email.split('@')[1]
            return f'[MASKED_USER]@{domain}'

        masked_line = re.sub(email_pattern, replace_email, line)
        return masked_line, masked_line != line

    def _get_quote_char(self, text: str, reverse: bool = False) -> Optional[str]:
        """Get the last/first quote character from text"""
        quotes = ['"', "'", '`']

        if reverse:
            for char in reversed(text):
                if char in quotes:
                    return char
        else:
            for char in text:
                if char in quotes:
                    return char

        return None

    def _calculate_sensitivity_level(self, secrets: List[DetectedSecret]) -> SensitivityLevel:
        """Calculate overall sensitivity level based on detected secrets"""
        if not secrets:
            return SensitivityLevel.LOW

        max_confidence = max(secret.confidence for secret in secrets)

        high_risk_types = {SecretType.PRIVATE_KEY, SecretType.DATABASE_URL,
                           SecretType.SECRET_KEY}
        medium_risk_types = {SecretType.API_KEY, SecretType.TOKEN,
                             SecretType.CREDENTIAL}

        # Check for high-risk secret types
        for secret in secrets:
            if secret.secret_type in high_risk_types:
                return SensitivityLevel.HIGH

        # Check for medium-risk types with high confidence
        for secret in secrets:
            if secret.secret_type in medium_risk_types and secret.confidence > 0.8:
                return SensitivityLevel.HIGH

        # Check for multiple medium-risk secrets
        medium_secrets = [s for s in secrets if s.secret_type in medium_risk_types]
        if len(medium_secrets) >= 3:
            return SensitivityLevel.HIGH

        # Check for any medium-risk secrets
        if medium_secrets or max_confidence > 0.7:
            return SensitivityLevel.MEDIUM

        return SensitivityLevel.LOW

    def create_masking_summary(self, result: MaskingResult) -> Dict:
        """Create a summary of masking operations"""
        secret_counts = {}
        for secret in result.detected_secrets:
            secret_type = secret.secret_type.value
            secret_counts[secret_type] = secret_counts.get(secret_type, 0) + 1

        return {
            "total_secrets_masked": result.masked_count,
            "sensitivity_level": result.sensitivity_level.value,
            "secret_types_found": secret_counts,
            "syntax_preserved": result.preserve_syntax,
            "masking_ratio": len(result.masked_content) / len(result.original_content) if result.original_content else 0
        }
