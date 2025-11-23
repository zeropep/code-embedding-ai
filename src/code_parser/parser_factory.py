from pathlib import Path
from typing import List, Optional, Dict
from .base_parser import BaseParser
from .java_parser import JavaParser
from .html_parser import HTMLParser
from .models import ParserConfig, ParsedFile, CodeLanguage


class SimpleParser(BaseParser):
    """Simple parser for unsupported file types"""

    def __init__(self, config: ParserConfig, language: CodeLanguage):
        super().__init__(config)
        self.language = language

    def can_parse(self, file_path: Path) -> bool:
        return True  # Can handle any file

    def get_language(self) -> CodeLanguage:
        return self.language

    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Simple content-based parsing for unsupported file types"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            total_lines, last_modified = self.get_file_stats(file_path)
            file_hash = self.calculate_file_hash(file_path)

            chunks = self.chunk_content(content, str(file_path), self.language)

            return ParsedFile(
                file_path=str(file_path),
                language=self.language,
                chunks=chunks,
                total_lines=total_lines,
                file_hash=file_hash,
                last_modified=last_modified
            )

        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None


class ParserFactory:
    """Factory class to create appropriate parsers for different file types"""

    def __init__(self, config: ParserConfig = None):
        if config is None:
            config = ParserConfig()
        self.config = config
        self._parsers: List[BaseParser] = []
        self._initialize_parsers()

    def _initialize_parsers(self):
        """Initialize all available parsers"""
        self._parsers = [
            JavaParser(self.config),
            HTMLParser(self.config),
        ]

    def get_parser(self, file_path: Path) -> BaseParser:
        """Get appropriate parser for the given file"""
        # Try specialized parsers first
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser

        # Fall back to simple parsers for other file types
        suffix = file_path.suffix.lower()

        if suffix in ['.kt', '.kts']:
            return SimpleParser(self.config, CodeLanguage.KOTLIN)
        elif suffix in ['.xml']:
            return SimpleParser(self.config, CodeLanguage.XML)
        elif suffix in ['.yml', '.yaml']:
            return SimpleParser(self.config, CodeLanguage.YAML)
        elif suffix in ['.properties']:
            return SimpleParser(self.config, CodeLanguage.PROPERTIES)
        else:
            # Default to Java for unknown files
            return SimpleParser(self.config, CodeLanguage.JAVA)

    def can_parse_file(self, file_path: Path) -> bool:
        """Check if we can parse the given file"""
        suffix = file_path.suffix.lower()
        return suffix in self.config.supported_extensions

    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Parse a file using the appropriate parser"""
        if not self.can_parse_file(file_path):
            return None

        parser = self.get_parser(file_path)
        return parser.parse_file(file_path)

    def parse_directory(self, directory_path: Path,
                       recursive: bool = True) -> List[ParsedFile]:
        """Parse all supported files in a directory"""
        parsed_files = []

        if not directory_path.exists() or not directory_path.is_dir():
            print(f"Directory does not exist: {directory_path}")
            return parsed_files

        # Get file pattern
        pattern = "**/*" if recursive else "*"

        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and self.can_parse_file(file_path):
                # Skip files that should be ignored
                if self._should_ignore_file(file_path):
                    continue

                parsed_file = self.parse_file(file_path)
                if parsed_file:
                    parsed_files.append(parsed_file)

        return parsed_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on patterns"""
        ignore_patterns = [
            "*.class", "*.jar", "*.war",
            "target/", "build/", ".git/",
            "node_modules/", ".idea/", "*.log"
        ]

        file_str = str(file_path).replace('\\', '/')

        for pattern in ignore_patterns:
            if pattern.endswith('/'):
                # Directory pattern
                if f"/{pattern}" in file_str or file_str.startswith(pattern):
                    return True
            else:
                # File pattern
                if file_str.endswith(pattern.replace('*', '')):
                    return True

        return False

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return self.config.supported_extensions

    def get_parser_stats(self) -> Dict[str, int]:
        """Get statistics about available parsers"""
        return {
            "total_parsers": len(self._parsers),
            "supported_extensions": len(self.config.supported_extensions)
        }