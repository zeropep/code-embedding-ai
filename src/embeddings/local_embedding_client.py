import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog

from .models import EmbeddingConfig, EmbeddingResult, EmbeddingStatus, EmbeddingTaskType
from ..utils.retry import retry_async, RetryConfig, NETWORK_RETRY_CONFIG
from ..utils.error_tracker import get_error_tracker, ErrorCategory, ErrorSeverity


logger = structlog.get_logger(__name__)


class LocalEmbeddingClient:
    """Client for local embedding using Sentence Transformers"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._cache: Dict[str, Tuple[List[float], float]] = {}
        self._rate_limiter = asyncio.Semaphore(config.max_concurrent_requests)
        self._model_loaded = False

        logger.info("LocalEmbeddingClient initialized",
                    model=config.model_name,
                    batch_size=config.batch_size)

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_model_loaded()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_model_loaded(self):
        """Ensure model is loaded with retry logic"""
        if self._model_loaded and self.model is not None:
            return

        error_tracker = get_error_tracker()

        try:
            # Import here to avoid loading if not needed
            from sentence_transformers import SentenceTransformer
            import torch

            # Get device to use
            device = self.config.get_device()

            # Log GPU status
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                logger.info("GPU detected",
                           cuda_available=True,
                           device_count=device_count,
                           selected_device=device)
            else:
                logger.info("GPU not available, using CPU",
                           cuda_available=False,
                           selected_device=device)

            logger.info("Loading local embedding model",
                       model=self.config.model_name,
                       device=device)
            start_time = time.time()

            # Define model loading function with retry
            async def load_model():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(
                        self.config.model_name,
                        device=device,
                        trust_remote_code=True
                    )
                )

            # Retry configuration for model loading (fewer retries, longer delays)
            model_retry_config = RetryConfig(
                max_attempts=2,  # Only 1 retry for model loading
                initial_delay=2.0,
                max_delay=10.0,
                exponential_base=2.0,
                jitter=False,
                retryable_exceptions=(ConnectionError, TimeoutError, OSError)
            )

            # Load model with retry
            try:
                self.model = await retry_async(
                    load_model,
                    config=model_retry_config,
                    on_retry=lambda e, attempt, delay: logger.warning(
                        "Retrying model load",
                        attempt=attempt,
                        delay=delay,
                        error=str(e)
                    )
                )
            except (RuntimeError, Exception) as e:
                # CUDA initialization failed, fallback to CPU
                if "CUDA" in str(e) or "cuda" in str(e):
                    logger.warning("CUDA initialization failed, falling back to CPU",
                                  error=str(e))
                    device = "cpu"
                    loop = asyncio.get_event_loop()
                    self.model = await loop.run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            self.config.model_name,
                            device="cpu",
                            trust_remote_code=True
                        )
                    )
                else:
                    raise

            # Set max sequence length (Jina supports up to 8192)
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = 8192

            self._model_loaded = True
            load_time = time.time() - start_time
            logger.info("Local model loaded successfully",
                       model=self.config.model_name,
                       device=device,
                       load_time=f"{load_time:.2f}s")

        except ImportError as e:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            error_tracker.record_error(
                e,
                category=ErrorCategory.EMBEDDING,
                severity=ErrorSeverity.CRITICAL,
                context={"model": self.config.model_name, "reason": "Missing dependency"}
            )
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error("Failed to load local embedding model", error=str(e))
            error_tracker.record_error(
                e,
                category=ErrorCategory.EMBEDDING,
                severity=ErrorSeverity.CRITICAL,
                context={"model": self.config.model_name, "operation": "model_load"}
            )
            raise

    async def close(self):
        """Close the client (cleanup if needed)"""
        # Sentence transformers doesn't need explicit cleanup
        # but we can clear the cache
        if self._model_loaded:
            logger.info("Closing local embedding client")
            self._model_loaded = False

    async def generate_embedding(self, content: str, request_id: str = "",
                                 task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE) -> EmbeddingResult:
        """Generate embedding for single content

        Args:
            content: Text to embed
            request_id: Request identifier
            task_type: Task-specific prefix for jina-code-embeddings-1.5b
        """
        start_time = time.time()

        # Check cache first (include task_type in cache key)
        if self.config.enable_caching:
            cached_vector = self._get_cached_embedding(content, task_type)
            if cached_vector:
                logger.debug("Cache hit for embedding", request_id=request_id, task_type=task_type.value)
                return EmbeddingResult(
                    request_id=request_id,
                    vector=cached_vector,
                    status=EmbeddingStatus.COMPLETED,
                    processing_time=time.time() - start_time,
                    model_version=self.config.model_name
                )

        try:
            await self._ensure_model_loaded()

            # Sanitize content
            sanitized = self._sanitize_text(content)
            if not sanitized:
                raise ValueError("Empty content after sanitization")

            # Generate embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def encode_single(model, text, task_value):
                try:
                    return model.encode(
                        [text],
                        convert_to_numpy=False,
                        prompt_name=task_value,
                        show_progress_bar=False
                    )[0]
                except TypeError:
                    # Fallback if prompt_name is not supported
                    try:
                        return model.encode(
                            [text],
                            convert_to_numpy=False,
                            show_progress_bar=False
                        )[0]
                    except Exception:
                        # Final fallback
                        return model.encode([text], convert_to_numpy=False)[0]

            from functools import partial
            encode_fn = partial(encode_single, self.model, sanitized, task_type.value)
            embedding_vector = await loop.run_in_executor(None, encode_fn)

            # Convert to list if needed
            if not isinstance(embedding_vector, list):
                embedding_vector = embedding_vector.tolist()

            # Cache the result
            if self.config.enable_caching:
                self._cache_embedding(content, embedding_vector, task_type)

            return EmbeddingResult(
                request_id=request_id,
                vector=embedding_vector,
                status=EmbeddingStatus.COMPLETED,
                processing_time=time.time() - start_time,
                model_version=self.config.model_name
            )

        except Exception as e:
            logger.error("Failed to generate embedding",
                         request_id=request_id,
                         error=str(e))

            # Track the error
            error_tracker = get_error_tracker()
            error_tracker.record_error(
                e,
                category=ErrorCategory.EMBEDDING,
                severity=ErrorSeverity.HIGH,
                context={
                    "request_id": request_id,
                    "task_type": task_type.value,
                    "content_length": len(content) if content else 0
                }
            )

            return EmbeddingResult(
                request_id=request_id,
                vector=None,
                status=EmbeddingStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    async def generate_embeddings_batch(self, contents: List[str],
                                        request_ids: List[str] = None,
                                        task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE) -> List[EmbeddingResult]:
        """Generate embeddings for multiple contents in batch

        Args:
            contents: List of texts to embed
            request_ids: Optional list of request identifiers
            task_type: Task-specific prefix for jina-code-embeddings-1.5b
        """
        if request_ids is None:
            request_ids = [f"batch_{i}" for i in range(len(contents))]

        if len(contents) != len(request_ids):
            raise ValueError("Contents and request_ids must have the same length")

        logger.info("Generating batch embeddings (local)",
                    batch_size=len(contents),
                    model=self.config.model_name,
                    task_type=task_type.value)

        # Process in chunks to respect batch size limits
        results = []
        for i in range(0, len(contents), self.config.batch_size):
            chunk_contents = contents[i:i + self.config.batch_size]
            chunk_ids = request_ids[i:i + self.config.batch_size]

            chunk_results = await self._process_batch_chunk(chunk_contents, chunk_ids, task_type)
            results.extend(chunk_results)

        return results

    async def _process_batch_chunk(self, contents: List[str],
                                   request_ids: List[str],
                                   task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE) -> List[EmbeddingResult]:
        """Process a single batch chunk

        Args:
            contents: List of texts to embed
            request_ids: List of request identifiers
            task_type: Task-specific prefix for jina-code-embeddings-1.5b
        """
        start_time = time.time()

        # Separate cached and non-cached contents
        cached_results = []
        uncached_contents = []
        uncached_ids = []
        uncached_indices = []

        if self.config.enable_caching:
            for idx, (content, req_id) in enumerate(zip(contents, request_ids)):
                cached_vector = self._get_cached_embedding(content, task_type)
                if cached_vector:
                    cached_results.append((idx, req_id, cached_vector))
                else:
                    uncached_contents.append(content)
                    uncached_ids.append(req_id)
                    uncached_indices.append(idx)
        else:
            uncached_contents = contents
            uncached_ids = request_ids
            uncached_indices = list(range(len(contents)))

        # Generate embeddings for uncached content
        api_results = []
        if uncached_contents:
            try:
                await self._ensure_model_loaded()

                # Sanitize all contents
                sanitized_contents = [self._sanitize_text(c) for c in uncached_contents]
                valid_indices = [i for i, c in enumerate(sanitized_contents) if c.strip()]
                valid_contents = [sanitized_contents[i] for i in valid_indices]

                if valid_contents:
                    # Generate embeddings in thread pool
                    async with self._rate_limiter:
                        loop = asyncio.get_event_loop()

                        # Create a wrapper function with explicit parameters to avoid closure issues
                        def encode_batch(model, contents, task_value, batch_size):
                            import torch
                            import gc

                            try:
                                # Use no_grad to prevent gradient computation (saves GPU memory)
                                with torch.no_grad():
                                    result = model.encode(
                                        contents,
                                        convert_to_numpy=False,
                                        prompt_name=task_value,
                                        batch_size=batch_size,
                                        show_progress_bar=False  # Disable progress bar in thread
                                    )

                                # Clear GPU cache after encoding
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                # Force garbage collection
                                gc.collect()

                                return result

                            except TypeError:
                                # Fallback if prompt_name is not supported
                                try:
                                    with torch.no_grad():
                                        result = model.encode(
                                            contents,
                                            convert_to_numpy=False,
                                            batch_size=batch_size,
                                            show_progress_bar=False
                                        )

                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    gc.collect()

                                    return result

                                except Exception:
                                    # Final fallback with minimal parameters
                                    with torch.no_grad():
                                        result = model.encode(contents, convert_to_numpy=False)

                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    gc.collect()

                                    return result

                        # Use functools.partial to avoid closure issues on Windows
                        from functools import partial
                        encode_fn = partial(encode_batch, self.model, valid_contents, task_type.value, self.config.batch_size)
                        embedding_vectors = await loop.run_in_executor(None, encode_fn)

                        # Convert to list if needed
                        if not isinstance(embedding_vectors, list):
                            embedding_vectors = embedding_vectors.tolist()

                        # Process results
                        valid_idx_map = {valid_indices[i]: i for i in range(len(valid_indices))}

                        for i, (content, req_id) in enumerate(zip(uncached_contents, uncached_ids)):
                            if i in valid_idx_map:
                                vector_idx = valid_idx_map[i]
                                vector = embedding_vectors[vector_idx]

                                # Convert numpy array to list if needed
                                if not isinstance(vector, list):
                                    vector = vector.tolist()

                                # Cache the result
                                if self.config.enable_caching:
                                    self._cache_embedding(content, vector, task_type)

                                api_results.append((uncached_indices[i], EmbeddingResult(
                                    request_id=req_id,
                                    vector=vector,
                                    status=EmbeddingStatus.COMPLETED,
                                    processing_time=time.time() - start_time,
                                    model_version=self.config.model_name
                                )))
                            else:
                                api_results.append((uncached_indices[i], EmbeddingResult(
                                    request_id=req_id,
                                    vector=None,
                                    status=EmbeddingStatus.FAILED,
                                    error_message="Empty content after sanitization"
                                )))

            except Exception as e:
                logger.error("Batch embedding failed", error=str(e))

                # Track the error
                error_tracker = get_error_tracker()
                error_tracker.record_error(
                    e,
                    category=ErrorCategory.EMBEDDING,
                    severity=ErrorSeverity.HIGH,
                    context={
                        "operation": "batch_embedding",
                        "batch_size": len(uncached_contents),
                        "task_type": task_type.value
                    }
                )

                # Return failed results for all uncached items
                for idx, req_id in zip(uncached_indices, uncached_ids):
                    api_results.append((idx, EmbeddingResult(
                        request_id=req_id,
                        vector=None,
                        status=EmbeddingStatus.FAILED,
                        error_message=str(e)
                    )))

        # Combine cached and API results in original order
        all_results = [None] * len(contents)

        # Add cached results
        for idx, req_id, vector in cached_results:
            all_results[idx] = EmbeddingResult(
                request_id=req_id,
                vector=vector,
                status=EmbeddingStatus.COMPLETED,
                processing_time=0.0,  # Cached, no processing time
                model_version=self.config.model_name
            )

        # Add API results
        for idx, result in api_results:
            all_results[idx] = result

        return all_results

    def _sanitize_text(self, text: str, max_chars: int = 8000) -> str:
        """Sanitize text - remove problematic characters and limit length"""
        if not text:
            return ""

        import re

        # Remove null bytes and other dangerous control characters
        # Keep printable characters and common whitespace (newline, tab, carriage return)
        sanitized = ''.join(
            char for char in text
            if char.isprintable() or char in '\n\t\r '
        )

        # Remove any remaining null bytes explicitly
        sanitized = sanitized.replace('\x00', '')

        # Replace multiple whitespace with single space
        sanitized = re.sub(r'[ \t]+', ' ', sanitized)
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)

        # Limit length to avoid token limit issues
        if len(sanitized) > max_chars:
            sanitized = sanitized[:max_chars]
            logger.debug("Text truncated", original_len=len(text), truncated_len=max_chars)

        return sanitized.strip()

    def _get_cache_key(self, content: str, task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE) -> str:
        """Generate cache key for content and task type"""
        # Include task_type in cache key since embeddings differ by task
        combined = f"{task_type.value}:{content}"
        return hashlib.sha256(combined.encode('utf-8', errors='ignore')).hexdigest()

    def _get_cached_embedding(self, content: str, task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE) -> Optional[List[float]]:
        """Get cached embedding if available and not expired

        Args:
            content: Text content
            task_type: Task-specific prefix (impacts embedding)
        """
        if not self.config.enable_caching:
            return None

        cache_key = self._get_cache_key(content, task_type)
        if cache_key in self._cache:
            vector, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return vector
            else:
                # Expired, remove from cache
                del self._cache[cache_key]

        return None

    def _cache_embedding(self, content: str, vector: List[float], task_type: EmbeddingTaskType = EmbeddingTaskType.CODE2CODE):
        """Cache embedding result

        Args:
            content: Text content
            vector: Embedding vector
            task_type: Task-specific prefix (impacts embedding)
        """
        if not self.config.enable_caching:
            return

        cache_key = self._get_cache_key(content, task_type)
        self._cache[cache_key] = (vector, time.time())

        # Simple cache size management
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )[:100]
            for key in oldest_keys:
                del self._cache[key]

    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.config.enable_caching:
            return {"enabled": False}

        total_entries = len(self._cache)
        total_size = sum(
            len(vector) * 4 if isinstance(vector, list) else 0
            for vector, _ in self._cache.values()
        )  # Rough size in bytes

        return {
            "enabled": True,
            "total_entries": total_entries,
            "estimated_size_mb": total_size / (1024 * 1024),
            "cache_ttl": self.config.cache_ttl
        }
