import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog

from .parser_factory import ParserFactory
from .models import ParsedFile, ParserConfig, CodeChunk


logger = structlog.get_logger(__name__)


class CodeParser:
    """Main code parsing orchestrator"""

    def __init__(self, config: ParserConfig = None, max_workers: int = 4):
        if config is None:
            config = ParserConfig()

        self.config = config
        self.max_workers = max_workers
        self.parser_factory = ParserFactory(config)

        logger.info("CodeParser initialized",
                   max_workers=max_workers,
                   supported_extensions=config.supported_extensions)

    def parse_repository(self, repo_path: str) -> List[ParsedFile]:
        """Parse entire repository"""
        repo_path_obj = Path(repo_path)

        if not repo_path_obj.exists():
            logger.error("Repository path does not exist", path=repo_path)
            return []

        logger.info("Starting repository parsing", path=repo_path)

        # Get all files to parse
        files_to_parse = self._get_files_to_parse(repo_path_obj)
        logger.info("Found files to parse", count=len(files_to_parse))

        # Parse files in parallel
        parsed_files = self._parse_files_parallel(files_to_parse)

        # Calculate statistics
        stats = self._calculate_stats(parsed_files)
        logger.info("Repository parsing completed", **stats)

        return parsed_files

    def parse_files(self, file_paths: List[str]) -> List[ParsedFile]:
        """Parse specific files"""
        path_objects = [Path(path) for path in file_paths]
        existing_files = [p for p in path_objects if p.exists() and p.is_file()]

        if len(existing_files) != len(file_paths):
            missing = len(file_paths) - len(existing_files)
            logger.warning("Some files not found", missing_count=missing)

        return self._parse_files_parallel(existing_files)

    def parse_single_file(self, file_path: str) -> Optional[ParsedFile]:
        """Parse a single file"""
        file_path_obj = Path(file_path)

        if not file_path_obj.exists() or not file_path_obj.is_file():
            logger.error("File does not exist", path=file_path)
            return None

        if not self.parser_factory.can_parse_file(file_path_obj):
            logger.warning("File type not supported", path=file_path,
                          suffix=file_path_obj.suffix)
            return None

        logger.debug("Parsing single file", path=file_path)
        return self.parser_factory.parse_file(file_path_obj)

    def _get_files_to_parse(self, repo_path: Path) -> List[Path]:
        """Get list of files that should be parsed"""
        files = []

        for extension in self.config.supported_extensions:
            # Use glob pattern to find files
            pattern = f"**/*{extension}"
            for file_path in repo_path.glob(pattern):
                if file_path.is_file() and not self._should_ignore_file(file_path):
                    files.append(file_path)

        return files

    def _parse_files_parallel(self, file_paths: List[Path]) -> List[ParsedFile]:
        """Parse files in parallel using ThreadPoolExecutor"""
        parsed_files = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self.parser_factory.parse_file, file_path): file_path
                for file_path in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    parsed_file = future.result()
                    if parsed_file:
                        parsed_files.append(parsed_file)
                        logger.debug("File parsed successfully",
                                   path=str(file_path),
                                   chunks=len(parsed_file.chunks))
                    else:
                        logger.warning("Failed to parse file", path=str(file_path))
                except Exception as e:
                    logger.error("Error parsing file",
                               path=str(file_path), error=str(e))

        return parsed_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = [
            ".git/", "target/", "build/", "node_modules/",
            ".idea/", ".vscode/", "__pycache__/",
            "*.class", "*.jar", "*.war", "*.log"
        ]

        file_str = str(file_path).replace('\\', '/')

        for pattern in ignore_patterns:
            if pattern.endswith('/'):
                if pattern[:-1] in file_str:
                    return True
            else:
                if file_str.endswith(pattern.replace('*', '')):
                    return True

        return False

    def _calculate_stats(self, parsed_files: List[ParsedFile]) -> Dict[str, Any]:
        """Calculate parsing statistics"""
        if not parsed_files:
            return {
                "total_files": 0,
                "total_chunks": 0,
                "total_tokens": 0,
                "languages": {},
                "layer_types": {}
            }

        languages = {}
        layer_types = {}
        total_chunks = 0
        total_tokens = 0

        for parsed_file in parsed_files:
            # Count by language
            lang = parsed_file.language.value
            languages[lang] = languages.get(lang, 0) + 1

            # Count chunks and tokens
            total_chunks += len(parsed_file.chunks)
            total_tokens += parsed_file.total_tokens

            # Count by layer type
            for chunk in parsed_file.chunks:
                layer = chunk.layer_type.value
                layer_types[layer] = layer_types.get(layer, 0) + 1

        return {
            "total_files": len(parsed_files),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / total_chunks if total_chunks > 0 else 0,
            "languages": languages,
            "layer_types": layer_types
        }

    def get_chunks_for_embedding(self, parsed_files: List[ParsedFile]) -> List[CodeChunk]:
        """Extract all chunks ready for embedding generation"""
        chunks = []

        for parsed_file in parsed_files:
            for chunk in parsed_file.chunks:
                # Filter chunks that are too small or too large
                if (self.config.min_tokens <= chunk.token_count <= self.config.max_tokens):
                    chunks.append(chunk)
                else:
                    logger.debug("Chunk filtered by size",
                               file=chunk.file_path,
                               tokens=chunk.token_count,
                               min_tokens=self.config.min_tokens,
                               max_tokens=self.config.max_tokens)

        logger.info("Chunks prepared for embedding",
                   total_chunks=len(chunks))
        return chunks

    async def parse_repository_async(self, repo_path: str) -> List[ParsedFile]:
        """Async version of repository parsing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse_repository, repo_path)

    def update_config(self, **kwargs):
        """Update parser configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Parser config updated", key=key, value=value)
            else:
                logger.warning("Invalid config key", key=key)