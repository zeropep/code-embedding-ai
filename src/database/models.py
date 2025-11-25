from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class VectorDBStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    chunk_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    content: str
    embedding_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "content": self.content,
            "embedding_vector": self.embedding_vector
        }


@dataclass
class VectorSearchQuery:
    """Query for vector similarity search"""
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None
    top_k: int = 10
    min_similarity: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    include_embeddings: bool = False

    def __post_init__(self):
        if not self.query_vector and not self.query_text:
            raise ValueError("Either query_vector or query_text must be provided")


@dataclass
class ChunkMetadata:
    """Metadata for stored code chunks"""
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    layer_type: str = "Unknown"
    token_count: int = 0
    sensitivity_level: str = "LOW"
    file_hash: Optional[str] = None
    last_updated: float = None
    embedding_model: str = "jina-code-embeddings-1.5b"
    embedding_dimensions: int = 1024

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function_name": self.function_name or "",
            "class_name": self.class_name or "",
            "layer_type": self.layer_type,
            "token_count": self.token_count,
            "sensitivity_level": self.sensitivity_level,
            "file_hash": self.file_hash or "",
            "last_updated": self.last_updated,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        return cls(**data)


@dataclass
class StoredChunk:
    """Complete stored chunk with content, metadata, and embedding"""
    chunk_id: str
    content: str
    embedding_vector: List[float]
    metadata: ChunkMetadata
    created_at: float = None
    updated_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "embedding_vector": self.embedding_vector,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "code_embeddings"
    persistent: bool = True
    persist_directory: str = "./chroma_db"
    embedding_function: str = "default"
    max_batch_size: int = 100
    connection_timeout: int = 30
    max_retries: int = 3
    enable_indexing: bool = True

    # Search configuration
    default_top_k: int = 10
    max_top_k: int = 100
    min_similarity_threshold: float = 0.0

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.collection_name:
            return False
        if self.max_batch_size <= 0 or self.max_batch_size > 1000:
            return False
        if self.connection_timeout <= 0:
            return False
        if self.default_top_k <= 0 or self.default_top_k > self.max_top_k:
            return False
        return True


@dataclass
class DatabaseStats:
    """Statistics about the vector database"""
    total_chunks: int = 0
    total_files: int = 0
    collection_size_mb: float = 0.0
    last_updated: float = None
    embedding_dimensions: int = 1024

    # Language distribution
    language_counts: Dict[str, int] = None
    layer_counts: Dict[str, int] = None
    sensitivity_counts: Dict[str, int] = None

    # Performance metrics
    avg_search_time_ms: float = 0.0
    total_searches: int = 0
    cache_hit_rate: float = 0.0

    def __post_init__(self):
        if self.language_counts is None:
            self.language_counts = {}
        if self.layer_counts is None:
            self.layer_counts = {}
        if self.sensitivity_counts is None:
            self.sensitivity_counts = {}
        if self.last_updated is None:
            self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "total_files": self.total_files,
            "collection_size_mb": self.collection_size_mb,
            "last_updated": self.last_updated,
            "embedding_dimensions": self.embedding_dimensions,
            "language_counts": self.language_counts,
            "layer_counts": self.layer_counts,
            "sensitivity_counts": self.sensitivity_counts,
            "avg_search_time_ms": self.avg_search_time_ms,
            "total_searches": self.total_searches,
            "cache_hit_rate": self.cache_hit_rate
        }


@dataclass
class BulkOperationResult:
    """Result of bulk database operations"""
    operation_type: str  # "insert", "update", "delete"
    total_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "processing_time": self.processing_time,
            "errors": self.errors
        }
