from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import hashlib
import os
from .models import ParsedFile, CodeChunk, ParserConfig, CodeLanguage


class BaseParser(ABC):
    """Abstract base class for all code parsers"""

    def __init__(self, config: ParserConfig):
        self.config = config

    @abstractmethod
    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Parse a single file and return ParsedFile object"""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""

    @abstractmethod
    def get_language(self) -> CodeLanguage:
        """Return the language this parser handles"""

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""

    def get_file_stats(self, file_path: Path) -> tuple[int, float]:
        """Get file line count and last modified timestamp"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            modified = os.path.getmtime(file_path)
            return lines, modified
        except Exception:
            return 0, 0.0

    def chunk_content(self, content: str, file_path: str, language: CodeLanguage) -> List[CodeChunk]:
        """Split content into chunks based on token limits"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_tokens = len(line) // 4  # Rough estimation

            if (current_tokens + line_tokens > self.config.max_tokens and
                    current_tokens >= self.config.min_tokens):
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                if chunk_content.strip():
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=language,
                        start_line=start_line,
                        end_line=i - 1,
                        token_count=current_tokens
                    )
                    chunks.append(chunk)

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_tokens = sum(len(ln) // 4 for ln in current_chunk)
                start_line = i - len(overlap_lines) + 1
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Add final chunk if it has content
        if current_chunk and current_tokens >= self.config.min_tokens:
            chunk_content = '\n'.join(current_chunk)
            if chunk_content.strip():
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    language=language,
                    start_line=start_line,
                    end_line=len(lines),
                    token_count=current_tokens
                )
                chunks.append(chunk)

        return chunks

    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines for chunk continuity"""
        if not lines:
            return []

        overlap_chars = self.config.overlap_tokens * 4
        overlap_lines = []
        char_count = 0

        for line in reversed(lines):
            if char_count + len(line) <= overlap_chars:
                overlap_lines.insert(0, line)
                char_count += len(line)
            else:
                break

        return overlap_lines
