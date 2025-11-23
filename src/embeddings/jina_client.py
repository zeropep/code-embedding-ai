import asyncio
import aiohttp
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

from .models import EmbeddingConfig, EmbeddingResult, EmbeddingStatus


logger = structlog.get_logger(__name__)


class JinaEmbeddingClient:
    """Client for Jina AI embedding API"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[List[float], float]] = {}
        self._rate_limiter = asyncio.Semaphore(config.max_concurrent_requests)

        logger.info("JinaEmbeddingClient initialized",
                   model=config.model_name,
                   batch_size=config.batch_size,
                   max_concurrent=config.max_concurrent_requests)

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )

    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def generate_embedding(self, content: str, request_id: str = "") -> EmbeddingResult:
        """Generate embedding for single content"""
        start_time = time.time()

        # Check cache first
        if self.config.enable_caching:
            cached_vector = self._get_cached_embedding(content)
            if cached_vector:
                logger.debug("Cache hit for embedding", request_id=request_id)
                return EmbeddingResult(
                    request_id=request_id,
                    vector=cached_vector,
                    status=EmbeddingStatus.COMPLETED,
                    processing_time=time.time() - start_time,
                    model_version=self.config.model_name
                )

        try:
            # Generate embedding via API
            vector = await self._call_jina_api([content])
            if vector and len(vector) > 0:
                embedding_vector = vector[0]

                # Cache the result
                if self.config.enable_caching:
                    self._cache_embedding(content, embedding_vector)

                return EmbeddingResult(
                    request_id=request_id,
                    vector=embedding_vector,
                    status=EmbeddingStatus.COMPLETED,
                    processing_time=time.time() - start_time,
                    model_version=self.config.model_name
                )
            else:
                raise ValueError("Empty embedding returned from API")

        except Exception as e:
            logger.error("Failed to generate embedding",
                        request_id=request_id,
                        error=str(e))
            return EmbeddingResult(
                request_id=request_id,
                vector=None,
                status=EmbeddingStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    async def generate_embeddings_batch(self, contents: List[str],
                                      request_ids: List[str] = None) -> List[EmbeddingResult]:
        """Generate embeddings for multiple contents in batch"""
        if request_ids is None:
            request_ids = [f"batch_{i}" for i in range(len(contents))]

        if len(contents) != len(request_ids):
            raise ValueError("Contents and request_ids must have the same length")

        logger.info("Generating batch embeddings",
                   batch_size=len(contents),
                   model=self.config.model_name)

        # Process in chunks to respect batch size limits
        results = []
        for i in range(0, len(contents), self.config.batch_size):
            chunk_contents = contents[i:i + self.config.batch_size]
            chunk_ids = request_ids[i:i + self.config.batch_size]

            chunk_results = await self._process_batch_chunk(chunk_contents, chunk_ids)
            results.extend(chunk_results)

        return results

    async def _process_batch_chunk(self, contents: List[str],
                                 request_ids: List[str]) -> List[EmbeddingResult]:
        """Process a single batch chunk"""
        start_time = time.time()

        # Separate cached and non-cached contents
        cached_results = []
        uncached_contents = []
        uncached_ids = []

        if self.config.enable_caching:
            for content, req_id in zip(contents, request_ids):
                cached_vector = self._get_cached_embedding(content)
                if cached_vector:
                    cached_results.append((req_id, cached_vector))
                else:
                    uncached_contents.append(content)
                    uncached_ids.append(req_id)
        else:
            uncached_contents = contents
            uncached_ids = request_ids

        # Generate embeddings for uncached content
        api_results = []
        if uncached_contents:
            try:
                async with self._rate_limiter:
                    vectors = await self._call_jina_api(uncached_contents)

                    for i, (content, req_id) in enumerate(zip(uncached_contents, uncached_ids)):
                        if i < len(vectors) and vectors[i]:
                            vector = vectors[i]

                            # Cache the result
                            if self.config.enable_caching:
                                self._cache_embedding(content, vector)

                            api_results.append(EmbeddingResult(
                                request_id=req_id,
                                vector=vector,
                                status=EmbeddingStatus.COMPLETED,
                                processing_time=time.time() - start_time,
                                model_version=self.config.model_name
                            ))
                        else:
                            api_results.append(EmbeddingResult(
                                request_id=req_id,
                                vector=None,
                                status=EmbeddingStatus.FAILED,
                                error_message="No vector returned for content"
                            ))

            except Exception as e:
                logger.error("Batch embedding failed", error=str(e))
                # Create failed results for all uncached contents
                for req_id in uncached_ids:
                    api_results.append(EmbeddingResult(
                        request_id=req_id,
                        vector=None,
                        status=EmbeddingStatus.FAILED,
                        error_message=str(e)
                    ))

        # Combine cached and API results
        all_results = []

        # Add cached results
        for req_id, vector in cached_results:
            all_results.append(EmbeddingResult(
                request_id=req_id,
                vector=vector,
                status=EmbeddingStatus.COMPLETED,
                processing_time=0.0,  # Cached, no processing time
                model_version=self.config.model_name
            ))

        # Add API results
        all_results.extend(api_results)

        # Sort results to match original order
        results_dict = {result.request_id: result for result in all_results}
        ordered_results = [results_dict[req_id] for req_id in request_ids]

        return ordered_results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _call_jina_api(self, contents: List[str]) -> List[List[float]]:
        """Call Jina AI API to generate embeddings"""
        await self._ensure_session()

        payload = {
            "model": self.config.model_name,
            "input": contents,
            "encoding_format": "float"
        }

        logger.debug("Calling Jina API",
                    model=self.config.model_name,
                    input_count=len(contents))

        async with self.session.post(self.config.api_url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                embeddings = []

                if "data" in data:
                    for item in data["data"]:
                        if "embedding" in item:
                            embeddings.append(item["embedding"])

                logger.debug("Jina API call successful",
                           embeddings_returned=len(embeddings))
                return embeddings

            elif response.status == 429:
                # Rate limit hit
                logger.warning("Rate limit hit, retrying...")
                await asyncio.sleep(self.config.retry_delay)
                raise aiohttp.ClientError("Rate limit exceeded")

            else:
                error_text = await response.text()
                logger.error("Jina API error",
                           status=response.status,
                           response=error_text)
                raise aiohttp.ClientError(f"API error {response.status}: {error_text}")

    def _get_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_embedding(self, content: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired"""
        if not self.config.enable_caching:
            return None

        cache_key = self._get_cache_key(content)
        if cache_key in self._cache:
            vector, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return vector
            else:
                # Expired, remove from cache
                del self._cache[cache_key]

        return None

    def _cache_embedding(self, content: str, vector: List[float]):
        """Cache embedding result"""
        if not self.config.enable_caching:
            return

        cache_key = self._get_cache_key(content)
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
        total_size = sum(len(vector[0]) * 4 for vector, _ in self._cache.values())  # Rough size in bytes

        return {
            "enabled": True,
            "total_entries": total_entries,
            "estimated_size_mb": total_size / (1024 * 1024),
            "cache_ttl": self.config.cache_ttl
        }