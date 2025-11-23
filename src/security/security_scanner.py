from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from .secret_detector import SecretDetector
from .content_masker import ContentMasker
from .models import SecurityConfig, MaskingResult, SensitivityLevel
from ..code_parser.models import CodeChunk


logger = structlog.get_logger(__name__)


class SecurityScanner:
    """Main security scanning orchestrator"""

    def __init__(self, config: SecurityConfig = None):
        if config is None:
            config = SecurityConfig()

        self.config = config
        self.detector = SecretDetector(config)
        self.masker = ContentMasker(config)

        logger.info("SecurityScanner initialized", enabled=config.enabled)

    def scan_and_mask_chunk(self, chunk: CodeChunk) -> CodeChunk:
        """Scan and mask a single code chunk"""
        if not self.config.enabled:
            return chunk

        logger.debug("Scanning chunk for secrets",
                   file_path=chunk.file_path,
                   function_name=chunk.function_name,
                   chunk_size=len(chunk.content))

        # Detect secrets
        detected_secrets = self.detector.detect_secrets(
            chunk.content,
            chunk.file_path
        )

        if not detected_secrets:
            logger.debug("No secrets detected in chunk")
            return chunk

        # Mask content
        masking_result = self.masker.mask_content(
            chunk.content,
            detected_secrets,
            chunk.file_path
        )

        # Update chunk with masked content and metadata
        masked_chunk = self._create_masked_chunk(chunk, masking_result)

        logger.info("Chunk security scan completed",
                   file_path=chunk.file_path,
                   secrets_found=len(detected_secrets),
                   secrets_masked=masking_result.masked_count,
                   sensitivity_level=masking_result.sensitivity_level.value)

        return masked_chunk

    def scan_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Scan and mask multiple code chunks"""
        if not self.config.enabled:
            return chunks

        logger.info("Starting security scan for chunks", count=len(chunks))

        masked_chunks = []
        total_secrets = 0
        high_sensitivity_files = []

        for chunk in chunks:
            masked_chunk = self.scan_and_mask_chunk(chunk)
            masked_chunks.append(masked_chunk)

            # Track statistics
            if 'security' in masked_chunk.metadata:
                security_meta = masked_chunk.metadata['security']
                total_secrets += len(security_meta.get('detected_secrets', []))

                if security_meta.get('sensitivity_level') == SensitivityLevel.HIGH.value:
                    if chunk.file_path not in high_sensitivity_files:
                        high_sensitivity_files.append(chunk.file_path)

        logger.info("Security scan completed",
                   total_chunks=len(chunks),
                   total_secrets_found=total_secrets,
                   high_sensitivity_files=len(high_sensitivity_files))

        return masked_chunks

    def _create_masked_chunk(self, original_chunk: CodeChunk,
                           masking_result: MaskingResult) -> CodeChunk:
        """Create a new chunk with masked content and security metadata"""
        # Copy original chunk
        masked_chunk = CodeChunk(
            content=masking_result.masked_content,
            file_path=original_chunk.file_path,
            language=original_chunk.language,
            start_line=original_chunk.start_line,
            end_line=original_chunk.end_line,
            function_name=original_chunk.function_name,
            class_name=original_chunk.class_name,
            layer_type=original_chunk.layer_type,
            token_count=original_chunk.token_count,  # Keep original token count
            metadata=original_chunk.metadata.copy() if original_chunk.metadata else {}
        )

        # Add security metadata
        security_metadata = {
            'sensitivity_level': masking_result.sensitivity_level.value,
            'secrets_masked': masking_result.masked_count,
            'detected_secrets': [secret.to_dict() for secret in masking_result.detected_secrets],
            'masking_summary': self.masker.create_masking_summary(masking_result),
            'original_hash': hash(original_chunk.content),
            'masked_hash': hash(masking_result.masked_content)
        }

        masked_chunk.metadata['security'] = security_metadata

        return masked_chunk

    def generate_security_report(self, chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        report = {
            "scan_summary": {
                "total_chunks_scanned": len(chunks),
                "total_secrets_found": 0,
                "total_secrets_masked": 0,
                "files_with_secrets": set(),
                "sensitivity_distribution": {
                    SensitivityLevel.LOW.value: 0,
                    SensitivityLevel.MEDIUM.value: 0,
                    SensitivityLevel.HIGH.value: 0
                }
            },
            "secret_types_found": {},
            "high_risk_files": [],
            "masking_statistics": {
                "total_content_size": 0,
                "masked_content_size": 0,
                "masking_ratio": 0.0
            },
            "recommendations": []
        }

        total_original_size = 0
        total_masked_size = 0

        for chunk in chunks:
            total_original_size += len(chunk.content)

            if 'security' not in chunk.metadata:
                total_masked_size += len(chunk.content)
                continue

            security_meta = chunk.metadata['security']

            # Count secrets
            detected_secrets = security_meta.get('detected_secrets', [])
            masked_count = security_meta.get('secrets_masked', 0)

            report["scan_summary"]["total_secrets_found"] += len(detected_secrets)
            report["scan_summary"]["total_secrets_masked"] += masked_count

            if detected_secrets:
                report["scan_summary"]["files_with_secrets"].add(chunk.file_path)

            # Count by sensitivity level
            sensitivity = security_meta.get('sensitivity_level', SensitivityLevel.LOW.value)
            report["scan_summary"]["sensitivity_distribution"][sensitivity] += 1

            # Count secret types
            for secret in detected_secrets:
                secret_type = secret.get('secret_type', 'unknown')
                report["secret_types_found"][secret_type] = report["secret_types_found"].get(secret_type, 0) + 1

            # Track high-risk files
            if sensitivity == SensitivityLevel.HIGH.value:
                file_info = {
                    "file_path": chunk.file_path,
                    "secrets_count": len(detected_secrets),
                    "function_name": chunk.function_name,
                    "class_name": chunk.class_name
                }
                report["high_risk_files"].append(file_info)

            total_masked_size += len(chunk.content)

        # Calculate masking statistics
        report["scan_summary"]["files_with_secrets"] = len(report["scan_summary"]["files_with_secrets"])
        report["masking_statistics"]["total_content_size"] = total_original_size
        report["masking_statistics"]["masked_content_size"] = total_masked_size
        report["masking_statistics"]["masking_ratio"] = (
            total_masked_size / total_original_size if total_original_size > 0 else 0
        )

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []

        total_secrets = report["scan_summary"]["total_secrets_found"]
        high_risk_count = report["scan_summary"]["sensitivity_distribution"][SensitivityLevel.HIGH.value]
        files_with_secrets = report["scan_summary"]["files_with_secrets"]

        if total_secrets == 0:
            recommendations.append("✅ No secrets detected in the codebase.")
            return recommendations

        if high_risk_count > 0:
            recommendations.append(
                f"🚨 {high_risk_count} high-risk chunks found with sensitive data. "
                "Review and move secrets to secure configuration."
            )

        if files_with_secrets > 5:
            recommendations.append(
                f"⚠️ {files_with_secrets} files contain secrets. "
                "Consider centralizing secret management."
            )

        # Secret type specific recommendations
        secret_types = report["secret_types_found"]

        if secret_types.get("private_key", 0) > 0:
            recommendations.append(
                "🔑 Private keys detected. Store in secure key management system."
            )

        if secret_types.get("database_url", 0) > 0:
            recommendations.append(
                "🗄️ Database URLs with credentials found. Use environment variables."
            )

        if secret_types.get("password", 0) > 0:
            recommendations.append(
                "🔒 Hardcoded passwords detected. Use secure password storage."
            )

        if secret_types.get("api_key", 0) > 0:
            recommendations.append(
                "🔐 API keys found in code. Move to secure configuration."
            )

        # General recommendations
        if total_secrets > 20:
            recommendations.append(
                "📋 Consider implementing automated secret scanning in CI/CD pipeline."
            )

        recommendations.append(
            "💡 Ensure all secrets are properly rotated and use short-lived credentials where possible."
        )

        return recommendations

    def is_file_sensitive(self, file_path: str, chunks: List[CodeChunk]) -> bool:
        """Check if a file contains sensitive information"""
        file_chunks = [c for c in chunks if c.file_path == file_path]

        for chunk in file_chunks:
            if 'security' in chunk.metadata:
                sensitivity = chunk.metadata['security'].get('sensitivity_level')
                if sensitivity in [SensitivityLevel.MEDIUM.value, SensitivityLevel.HIGH.value]:
                    return True

        return False

    def update_config(self, **kwargs):
        """Update security configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Security config updated", key=key, value=value)

        # Update sub-components
        self.detector.config = self.config
        self.masker.config = self.config