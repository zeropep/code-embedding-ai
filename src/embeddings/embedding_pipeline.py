import asyncio
from typing import List, Dict, Any, Optional
import structlog

from .embedding_service import EmbeddingService
from .models import EmbeddingConfig
from ..code_parser.code_parser import CodeParser
from ..code_parser.models import ParserConfig, CodeChunk
from ..security.security_scanner import SecurityScanner
from ..security.models import SecurityConfig
from ..database.vector_store import VectorStore
from ..database.models import VectorDBConfig


logger = structlog.get_logger(__name__)


class EmbeddingPipeline:
    """Complete pipeline for code parsing, security scanning, and embedding generation"""

    def __init__(self,
                 parser_config: ParserConfig = None,
                 security_config: SecurityConfig = None,
                 embedding_config: EmbeddingConfig = None,
                 vector_config: VectorDBConfig = None,
                 auto_save: bool = True,
                 chunk_batch_size: int = 1000):

        # Initialize configurations
        self.parser_config = parser_config or ParserConfig()
        self.security_config = security_config or SecurityConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_config = vector_config or VectorDBConfig()
        self.auto_save = auto_save
        self.chunk_batch_size = chunk_batch_size  # Process this many chunks at a time

        # Initialize components
        self.code_parser = CodeParser(self.parser_config)
        self.security_scanner = SecurityScanner(self.security_config)
        self.embedding_service = EmbeddingService(self.embedding_config)
        self.vector_store = VectorStore(self.vector_config)

        logger.info("EmbeddingPipeline initialized",
                   auto_save=auto_save,
                   chunk_batch_size=chunk_batch_size)

    async def process_repository(self, repo_path: str, project_id: Optional[str] = None, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Process entire repository through the pipeline"""
        logger.info("Starting repository processing", repo_path=repo_path, project_id=project_id, project_name=project_name)

        try:
            # Step 1: Parse repository
            logger.info("Step 1: Parsing repository")
            parsed_files = await self.code_parser.parse_repository_async(repo_path)

            if not parsed_files:
                logger.warning("No files parsed from repository")
                return self._create_empty_result("No files found or parsed")

            # Extract chunks
            all_chunks = []
            for parsed_file in parsed_files:
                all_chunks.extend(parsed_file.chunks)

            # Set project metadata on all chunks
            if project_id or project_name:
                for chunk in all_chunks:
                    chunk.project_id = project_id
                    chunk.project_name = project_name

            logger.info("Parsing completed",
                        total_files=len(parsed_files),
                        total_chunks=len(all_chunks))

            # Step 2: Security scanning and masking
            logger.info("Step 2: Security scanning and masking")
            secured_chunks = self.security_scanner.scan_chunks(all_chunks)

            # Step 3 & 4: Generate embeddings and store in batches
            logger.info("Step 3: Generating embeddings in batches",
                       total_chunks=len(secured_chunks),
                       batch_size=self.chunk_batch_size)

            # Connect to database if auto_save is enabled
            if self.auto_save:
                if not self.vector_store.connect():
                    logger.error("Failed to connect to vector database")
                    return self._create_error_result("Failed to connect to vector database")

            await self.embedding_service.start()
            embedded_chunks = []
            total_stored = 0

            try:
                # Process in batches
                for batch_idx in range(0, len(secured_chunks), self.chunk_batch_size):
                    batch_end = min(batch_idx + self.chunk_batch_size, len(secured_chunks))
                    batch = secured_chunks[batch_idx:batch_end]
                    batch_num = (batch_idx // self.chunk_batch_size) + 1
                    total_batches = (len(secured_chunks) + self.chunk_batch_size - 1) // self.chunk_batch_size

                    logger.info(f"Processing batch {batch_num}/{total_batches}",
                               batch_chunks=len(batch),
                               progress=f"{batch_end}/{len(secured_chunks)}")

                    # Generate embeddings for this batch
                    batch_embedded = await self.embedding_service.generate_chunk_embeddings(batch)
                    embedded_chunks.extend(batch_embedded)

                    # Store this batch immediately if auto_save enabled
                    if self.auto_save:
                        store_result = self.vector_store.store_chunks(batch_embedded)
                        total_stored += store_result.successful_items
                        logger.info(f"Batch {batch_num}/{total_batches} stored",
                                   stored=store_result.successful_items,
                                   failed=store_result.failed_items,
                                   total_stored=total_stored)

                if self.auto_save:
                    logger.info("All batches stored in database", total_stored=total_stored)

            finally:
                await self.embedding_service.stop()

            # Step 5: Compile results
            logger.info("Step 5: Compiling results")
            result = self._compile_pipeline_result(
                repo_path, parsed_files, all_chunks, secured_chunks, embedded_chunks
            )

            logger.info("Repository processing completed",
                        total_files=len(parsed_files),
                        total_chunks=len(embedded_chunks),
                        successful_embeddings=sum(1 for c in embedded_chunks
                                                  if 'embedding' in c.metadata))

            return result

        except Exception as e:
            logger.error("Repository processing failed", repo_path=repo_path, error=str(e))
            return self._create_error_result(str(e))

    async def process_files(self, file_paths: List[str], project_id: Optional[str] = None, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Process specific files through the pipeline"""
        logger.info("Starting file processing", file_count=len(file_paths), project_id=project_id, project_name=project_name)

        try:
            # Step 1: Parse files
            parsed_files = self.code_parser.parse_files(file_paths)

            if not parsed_files:
                return self._create_empty_result("No files parsed")

            # Extract chunks
            all_chunks = []
            for parsed_file in parsed_files:
                all_chunks.extend(parsed_file.chunks)

            # Set project metadata on all chunks
            if project_id or project_name:
                for chunk in all_chunks:
                    chunk.project_id = project_id
                    chunk.project_name = project_name

            # Step 2: Security scanning
            secured_chunks = self.security_scanner.scan_chunks(all_chunks)

            # Step 3 & 4: Generate embeddings and store in batches
            logger.info("Generating embeddings in batches",
                       total_chunks=len(secured_chunks),
                       batch_size=self.chunk_batch_size)

            # Connect to database if auto_save is enabled
            if self.auto_save:
                if not self.vector_store.connect():
                    logger.error("Failed to connect to vector database")
                    return self._create_error_result("Failed to connect to vector database")

            await self.embedding_service.start()
            embedded_chunks = []
            total_stored = 0

            try:
                # Process in batches
                for batch_idx in range(0, len(secured_chunks), self.chunk_batch_size):
                    batch_end = min(batch_idx + self.chunk_batch_size, len(secured_chunks))
                    batch = secured_chunks[batch_idx:batch_end]
                    batch_num = (batch_idx // self.chunk_batch_size) + 1
                    total_batches = (len(secured_chunks) + self.chunk_batch_size - 1) // self.chunk_batch_size

                    logger.info(f"Processing batch {batch_num}/{total_batches}",
                               batch_chunks=len(batch),
                               progress=f"{batch_end}/{len(secured_chunks)}")

                    # Generate embeddings for this batch
                    batch_embedded = await self.embedding_service.generate_chunk_embeddings(batch)
                    embedded_chunks.extend(batch_embedded)

                    # Store this batch immediately if auto_save enabled
                    if self.auto_save:
                        store_result = self.vector_store.store_chunks(batch_embedded)
                        total_stored += store_result.successful_items
                        logger.info(f"Batch {batch_num}/{total_batches} stored",
                                   stored=store_result.successful_items,
                                   failed=store_result.failed_items,
                                   total_stored=total_stored)

                if self.auto_save:
                    logger.info("All batches stored in database", total_stored=total_stored)

            finally:
                await self.embedding_service.stop()

            # Compile results
            result = self._compile_pipeline_result(
                "files", parsed_files, all_chunks, secured_chunks, embedded_chunks
            )

            return result

        except Exception as e:
            logger.error("File processing failed", error=str(e))
            return self._create_error_result(str(e))

    def process_repository_sync(self, repo_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for repository processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_repository(repo_path))
        finally:
            loop.close()

    def process_files_sync(self, file_paths: List[str]) -> Dict[str, Any]:
        """Synchronous wrapper for file processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_files(file_paths))
        finally:
            loop.close()

    def _compile_pipeline_result(self,
                                 source: str,
                                 parsed_files: List,
                                 original_chunks: List[CodeChunk],
                                 secured_chunks: List[CodeChunk],
                                 embedded_chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Compile comprehensive pipeline results"""

        # Calculate statistics
        embedding_success_count = sum(1 for chunk in embedded_chunks
                                      if 'embedding' in chunk.metadata)

        security_stats = self.security_scanner.generate_security_report(secured_chunks)
        embedding_metrics = self.embedding_service.get_metrics()

        # Language distribution
        language_dist = {}
        for chunk in embedded_chunks:
            lang = chunk.language.value
            language_dist[lang] = language_dist.get(lang, 0) + 1

        # Layer distribution
        layer_dist = {}
        for chunk in embedded_chunks:
            layer = chunk.layer_type.value
            layer_dist[layer] = layer_dist.get(layer, 0) + 1

        # Sensitivity distribution
        sensitivity_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for chunk in secured_chunks:
            if 'security' in chunk.metadata:
                sensitivity = chunk.metadata['security'].get('sensitivity_level', 'LOW')
                sensitivity_dist[sensitivity] = sensitivity_dist.get(sensitivity, 0) + 1

        result = {
            "status": "success",
            "source": source,
            "processing_summary": {
                "total_files_parsed": len(parsed_files),
                "total_chunks_created": len(original_chunks),
                "chunks_after_security": len(secured_chunks),
                "chunks_with_embeddings": embedding_success_count,
                "embedding_success_rate": embedding_success_count / len(embedded_chunks) if embedded_chunks else 0
            },
            "parsing_stats": {
                "supported_files": len(parsed_files),
                "total_tokens": sum(chunk.token_count for chunk in original_chunks),
                "avg_tokens_per_chunk": (sum(chunk.token_count for chunk in original_chunks) /
                                         len(original_chunks) if original_chunks else 0),
                "language_distribution": language_dist,
                "layer_distribution": layer_dist
            },
            "security_stats": security_stats,
            "embedding_stats": {
                "total_embeddings_generated": embedding_success_count,
                "failed_embeddings": len(embedded_chunks) - embedding_success_count,
                "model_used": self.embedding_config.model_name,
                "dimensions": self.embedding_config.dimensions,
                "metrics": embedding_metrics,
                "sensitivity_distribution": sensitivity_dist
            },
            "chunks": [chunk.to_dict() for chunk in embedded_chunks],
            "file_details": [
                {
                    "file_path": pf.file_path,
                    "language": pf.language.value,
                    "chunk_count": len(pf.chunks),
                    "total_tokens": pf.total_tokens,
                    "file_hash": pf.file_hash,
                    "last_modified": pf.last_modified
                }
                for pf in parsed_files
            ]
        }

        return result

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result with reason"""
        return {
            "status": "empty",
            "reason": reason,
            "processing_summary": {
                "total_files_parsed": 0,
                "total_chunks_created": 0,
                "chunks_after_security": 0,
                "chunks_with_embeddings": 0,
                "embedding_success_rate": 0.0
            },
            "chunks": [],
            "file_details": []
        }

    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "status": "error",
            "error": error,
            "processing_summary": {
                "total_files_parsed": 0,
                "total_chunks_created": 0,
                "chunks_after_security": 0,
                "chunks_with_embeddings": 0,
                "embedding_success_rate": 0.0
            },
            "chunks": [],
            "file_details": []
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all pipeline components"""
        health = {
            "overall_status": "healthy",
            "components": {}
        }

        try:
            # Check embedding service
            await self.embedding_service.start()
            embedding_health = await self.embedding_service.health_check()
            await self.embedding_service.stop()
            health["components"]["embedding_service"] = embedding_health

            # Check parser (basic check)
            health["components"]["code_parser"] = {
                "status": "healthy",
                "supported_extensions": len(self.parser_config.supported_extensions)
            }

            # Check security scanner
            health["components"]["security_scanner"] = {
                "status": "healthy",
                "enabled": self.security_config.enabled
            }

            # Determine overall status
            if any(comp.get("status") == "unhealthy" for comp in health["components"].values()):
                health["overall_status"] = "unhealthy"

        except Exception as e:
            health["overall_status"] = "unhealthy"
            health["error"] = str(e)
            logger.error("Pipeline health check failed", error=str(e))

        return health

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "parser": {
                "supported_extensions": self.parser_config.supported_extensions,
                "min_tokens": self.parser_config.min_tokens,
                "max_tokens": self.parser_config.max_tokens
            },
            "security": {
                "enabled": self.security_config.enabled,
                "preserve_syntax": self.security_config.preserve_syntax,
                "sensitivity_threshold": self.security_config.sensitivity_threshold
            },
            "embedding": {
                "model": self.embedding_config.model_name,
                "dimensions": self.embedding_config.dimensions,
                "batch_size": self.embedding_config.batch_size,
                "metrics": self.embedding_service.get_metrics()
            }
        }
