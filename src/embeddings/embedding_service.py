import asyncio
import uuid
from typing import List, Dict, Any, Optional
import structlog

from .jina_client import JinaEmbeddingClient
from .models import (EmbeddingConfig, EmbeddingRequest, EmbeddingResult,
                     EmbeddingStatus, EmbeddingMetrics)
from ..code_parser.models import CodeChunk


logger = structlog.get_logger(__name__)


class EmbeddingService:
    """High-level service for generating code embeddings"""

    def __init__(self, config: EmbeddingConfig = None):
        if config is None:
            config = EmbeddingConfig()

        validation_error = config.validate()
        if validation_error:
            raise ValueError(f"Invalid embedding configuration: {validation_error}")

        self.config = config
        self.client = JinaEmbeddingClient(config)
        self.metrics = EmbeddingMetrics()
        self._request_queue: asyncio.Queue = None
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False

        logger.info("EmbeddingService initialized",
                    model=config.model_name,
                    batch_size=config.batch_size)

    async def start(self):
        """Start the embedding service"""
        if self._is_running:
            return

        self._request_queue = asyncio.Queue()
        self._processing_task = asyncio.create_task(self._process_requests())
        self._is_running = True

        logger.info("EmbeddingService started")

    async def stop(self):
        """Stop the embedding service"""
        if not self._is_running:
            return

        self._is_running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        await self.client.close()
        logger.info("EmbeddingService stopped")

    async def generate_chunk_embeddings(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generate embeddings for code chunks"""
        if not chunks:
            return []

        logger.info("Generating embeddings for chunks", count=len(chunks))

        # Prepare embedding requests
        requests = []
        for chunk in chunks:
            request = EmbeddingRequest(
                id=str(uuid.uuid4()),
                content=chunk.content,
                metadata={
                    "file_path": chunk.file_path,
                    "function_name": chunk.function_name,
                    "class_name": chunk.class_name,
                    "layer_type": chunk.layer_type.value,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line
                }
            )
            requests.append(request)

        # Generate embeddings
        results = await self.generate_embeddings_async(requests)

        # Combine results with chunks
        embedded_chunks = []
        for chunk, result in zip(chunks, results):
            if result.status == EmbeddingStatus.COMPLETED and result.vector:
                # Create new chunk with embedding
                embedded_chunk = self._create_embedded_chunk(chunk, result)
                embedded_chunks.append(embedded_chunk)
            else:
                logger.warning("Failed to generate embedding for chunk",
                               file_path=chunk.file_path,
                               function_name=chunk.function_name,
                               error=result.error_message)
                # Add chunk without embedding but with error info
                chunk.metadata = chunk.metadata or {}
                chunk.metadata['embedding_error'] = result.error_message
                embedded_chunks.append(chunk)

        logger.info("Chunk embeddings completed",
                    total_chunks=len(chunks),
                    successful_embeddings=sum(1 for r in results if r.status == EmbeddingStatus.COMPLETED),
                    failed_embeddings=sum(1 for r in results if r.status == EmbeddingStatus.FAILED))

        return embedded_chunks

    async def generate_embeddings_async(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResult]:
        """Generate embeddings asynchronously"""
        if not requests:
            return []

        # Extract contents for batch processing
        contents = [req.content for req in requests]
        request_ids = [req.id for req in requests]

        try:
            async with self.client as client:
                results = await client.generate_embeddings_batch(contents, request_ids)

            # Update metrics
            for result in results:
                if result.status == EmbeddingStatus.COMPLETED:
                    self.metrics.update_success(result.processing_time or 0)
                else:
                    self.metrics.update_failure()

            return results

        except Exception as e:
            logger.error("Batch embedding generation failed", error=str(e))
            # Return failed results
            return [
                EmbeddingResult(
                    request_id=req.id,
                    vector=None,
                    status=EmbeddingStatus.FAILED,
                    error_message=str(e)
                )
                for req in requests
            ]

    def generate_embeddings_sync(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Synchronous wrapper for embedding generation"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_chunk_embeddings(chunks))
        finally:
            loop.close()

    async def _process_requests(self):
        """Background task to process embedding requests"""
        batch = []
        last_batch_time = asyncio.get_event_loop().time()

        while self._is_running:
            try:
                # Wait for request or timeout
                timeout = 1.0  # Process batch every second
                try:
                    request = await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
                    batch.append(request)
                except asyncio.TimeoutError:
                    pass

                current_time = asyncio.get_event_loop().time()

                # Process batch if it's full or timeout reached
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_batch_time >= timeout)
                )

                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time

            except Exception as e:
                logger.error("Error in request processing loop", error=str(e))
                await asyncio.sleep(1.0)

    async def _process_batch(self, requests: List[EmbeddingRequest]):
        """Process a batch of embedding requests"""
        logger.debug("Processing embedding batch", size=len(requests))

        try:
            await self.generate_embeddings_async(requests)
            # Results are handled by the async method
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))

    def _create_embedded_chunk(self, chunk: CodeChunk, result: EmbeddingResult) -> CodeChunk:
        """Create a new chunk with embedding data"""
        # Create new chunk with embedding
        embedded_chunk = CodeChunk(
            content=chunk.content,
            file_path=chunk.file_path,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            function_name=chunk.function_name,
            class_name=chunk.class_name,
            layer_type=chunk.layer_type,
            token_count=chunk.token_count,
            metadata=chunk.metadata.copy() if chunk.metadata else {}
        )

        # Add embedding metadata
        embedding_metadata = {
            "vector": result.vector,
            "model_version": result.model_version,
            "processing_time": result.processing_time,
            "embedding_id": result.request_id,
            "dimensions": len(result.vector) if result.vector else 0,
            "generated_at": result.created_at
        }

        embedded_chunk.metadata['embedding'] = embedding_metadata

        return embedded_chunk

    def get_metrics(self) -> Dict[str, Any]:
        """Get embedding service metrics"""
        metrics = self.metrics.to_dict()
        metrics.update({
            "cache_stats": self.client.get_cache_stats(),
            "queue_size": self._request_queue.qsize() if self._request_queue else 0,
            "is_running": self._is_running
        })
        return metrics

    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = EmbeddingMetrics()
        logger.info("Embedding metrics reset")

    def clear_cache(self):
        """Clear embedding cache"""
        self.client.clear_cache()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            "status": "healthy",
            "service_running": self._is_running,
            "api_accessible": False,
            "cache_enabled": self.config.enable_caching
        }

        try:
            # Test API with small request
            test_content = "public class Test { }"
            async with self.client as client:
                result = await client.generate_embedding(test_content, "health_check")
                health["api_accessible"] = result.status == EmbeddingStatus.COMPLETED

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error("Health check failed", error=str(e))

        return health

    def update_config(self, **kwargs):
        """Update service configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                logger.info("Config updated", key=key, old_value=old_value, new_value=value)

        # Validate updated config
        validation_error = self.config.validate()
        if validation_error:
            logger.error("Invalid configuration after update", error=validation_error)
            raise ValueError(f"Invalid configuration: {validation_error}")

        # Update client config
        self.client.config = self.config
