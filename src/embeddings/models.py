from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import os
from dotenv import load_dotenv


class EmbeddingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    request_id: str
    vector: Optional[List[float]]
    status: EmbeddingStatus
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    created_at: float = None
    retry_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "vector": self.vector,
            "status": self.status.value,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "retry_count": self.retry_count
        }


@dataclass
class BatchEmbeddingRequest:
    """Batch request for multiple embeddings"""
    requests: List[EmbeddingRequest]
    batch_id: str
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    @property
    def size(self) -> int:
        return len(self.requests)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "jina-embeddings-v2-base-code"
    api_url: str = "https://api.jina.ai/v1/embeddings"
    api_key: str = ""
    dimensions: int = 1024
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    batch_size: int = 20
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 200
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds

    def __post_init__(self):
        """Load values from environment variables if not explicitly set"""
        # Ensure .env file is loaded
        load_dotenv()
        
        # Only load from env if api_key is empty (default value)
        if not self.api_key:
            self.api_key = os.getenv("JINA_API_KEY", "")
        # Only load from env if api_url is default value
        if self.api_url == "https://api.jina.ai/v1/embeddings":
            self.api_url = os.getenv("JINA_API_URL", self.api_url)
        # Load other optional configs from environment
        if self.model_name == "jina-embeddings-v2-base-code":
            self.model_name = os.getenv("EMBEDDING_MODEL", self.model_name)
        if self.dimensions == 1024:
            self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(self.dimensions)))
        if self.batch_size == 20:
            self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(self.batch_size)))
        if self.max_concurrent_requests == 10:
            self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", str(self.max_concurrent_requests)))
        if self.timeout == 30:
            self.timeout = int(os.getenv("EMBEDDING_TIMEOUT", str(self.timeout)))
        if self.enable_caching:
            cache_env = os.getenv("ENABLE_EMBEDDING_CACHE", "true")
            self.enable_caching = cache_env.lower() == "true"

    def validate(self) -> Optional[str]:
        """Validate configuration. Returns None if valid, error message if invalid."""
        if not self.api_key:
            return "api_key is required but not set"
        if not self.api_url:
            return "api_url is required but not set"
        if self.batch_size <= 0 or self.batch_size > 100:
            return f"batch_size must be between 1 and 100, got {self.batch_size}"
        if self.max_concurrent_requests <= 0:
            return f"max_concurrent_requests must be greater than 0, got {self.max_concurrent_requests}"
        return None


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding generation"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    requests_per_minute: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

    def update_success(self, processing_time: float):
        """Update metrics for successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_processing_time += processing_time
        self._update_averages()

    def update_failure(self):
        """Update metrics for failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self._update_averages()

    def update_retry(self):
        """Update metrics for retry"""
        self.retried_requests += 1

    def update_cache_hit(self):
        """Update cache hit metrics"""
        self.cache_hits += 1

    def update_cache_miss(self):
        """Update cache miss metrics"""
        self.cache_misses += 1

    def _update_averages(self):
        """Update average calculations"""
        if self.successful_requests > 0:
            self.avg_processing_time = self.total_processing_time / self.successful_requests

        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.requests_per_minute = (self.total_requests * 60) / elapsed_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "avg_processing_time": self.avg_processing_time,
            "requests_per_minute": self.requests_per_minute,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "uptime": time.time() - self.start_time
        }