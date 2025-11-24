"""
Tests for database module
Based on actual implementation in src/database/
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from src.database.models import (
    VectorDBConfig, VectorDBStatus, VectorSearchResult, VectorSearchQuery,
    ChunkMetadata, StoredChunk, DatabaseStats, BulkOperationResult
)


class TestVectorDBConfig:
    """Test VectorDBConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VectorDBConfig()

        assert config.host == "localhost"
        assert config.port == 8000
        assert config.collection_name == "code_embeddings"
        assert config.persistent is True
        assert config.max_batch_size == 100

    def test_custom_config(self):
        """Test custom configuration values"""
        config = VectorDBConfig(
            host="remote-server",
            port=9000,
            collection_name="test_collection",
            persistent=False,
            max_batch_size=50
        )

        assert config.host == "remote-server"
        assert config.port == 9000
        assert config.collection_name == "test_collection"
        assert config.persistent is False
        assert config.max_batch_size == 50

    def test_config_validation_valid(self):
        """Test valid configuration"""
        config = VectorDBConfig()
        assert config.validate() is True

    def test_config_validation_invalid_collection(self):
        """Test invalid collection name"""
        config = VectorDBConfig(collection_name="")
        assert config.validate() is False

    def test_config_validation_invalid_batch_size(self):
        """Test invalid batch size"""
        config = VectorDBConfig(max_batch_size=0)
        assert config.validate() is False

        config2 = VectorDBConfig(max_batch_size=1001)
        assert config2.validate() is False


class TestVectorDBStatus:
    """Test VectorDBStatus enum"""

    def test_status_values(self):
        """Test status enum values"""
        assert VectorDBStatus.CONNECTED.value == "connected"
        assert VectorDBStatus.DISCONNECTED.value == "disconnected"
        assert VectorDBStatus.ERROR.value == "error"
        assert VectorDBStatus.INITIALIZING.value == "initializing"


class TestChunkMetadata:
    """Test ChunkMetadata dataclass"""

    def test_metadata_creation(self):
        """Test creating chunk metadata"""
        metadata = ChunkMetadata(
            chunk_id="chunk-123",
            file_path="src/Main.java",
            language="java",
            start_line=10,
            end_line=50,
            function_name="processData",
            class_name="MainController"
        )

        assert metadata.chunk_id == "chunk-123"
        assert metadata.file_path == "src/Main.java"
        assert metadata.language == "java"
        assert metadata.function_name == "processData"
        assert metadata.last_updated > 0

    def test_metadata_defaults(self):
        """Test metadata default values"""
        metadata = ChunkMetadata(
            chunk_id="chunk-456",
            file_path="test.java",
            language="java",
            start_line=1,
            end_line=10
        )

        assert metadata.layer_type == "Unknown"
        assert metadata.token_count == 0
        assert metadata.sensitivity_level == "LOW"
        assert metadata.embedding_model == "jina-embeddings-v2-base-code"
        assert metadata.embedding_dimensions == 1024

    def test_metadata_to_dict(self):
        """Test metadata to_dict method"""
        metadata = ChunkMetadata(
            chunk_id="chunk-789",
            file_path="test.java",
            language="java",
            start_line=1,
            end_line=20,
            function_name="testFunc"
        )

        data = metadata.to_dict()
        assert data["chunk_id"] == "chunk-789"
        assert data["file_path"] == "test.java"
        assert data["function_name"] == "testFunc"

    def test_metadata_from_dict(self):
        """Test metadata from_dict method"""
        data = {
            "chunk_id": "chunk-abc",
            "file_path": "Test.java",
            "language": "java",
            "start_line": 5,
            "end_line": 15,
            "function_name": "test"
        }

        metadata = ChunkMetadata.from_dict(data)
        assert metadata.chunk_id == "chunk-abc"
        assert metadata.file_path == "Test.java"


class TestStoredChunk:
    """Test StoredChunk dataclass"""

    def test_stored_chunk_creation(self):
        """Test creating stored chunk"""
        metadata = ChunkMetadata(
            chunk_id="chunk-001",
            file_path="src/App.java",
            language="java",
            start_line=1,
            end_line=10
        )

        chunk = StoredChunk(
            chunk_id="chunk-001",
            content="public class App {}",
            embedding_vector=[0.1] * 1024,
            metadata=metadata
        )

        assert chunk.chunk_id == "chunk-001"
        assert chunk.content == "public class App {}"
        assert len(chunk.embedding_vector) == 1024
        assert chunk.created_at > 0
        assert chunk.updated_at > 0

    def test_stored_chunk_to_dict(self):
        """Test stored chunk to_dict method"""
        metadata = ChunkMetadata(
            chunk_id="chunk-002",
            file_path="Test.java",
            language="java",
            start_line=1,
            end_line=5
        )

        chunk = StoredChunk(
            chunk_id="chunk-002",
            content="test content",
            embedding_vector=[0.5] * 10,
            metadata=metadata
        )

        data = chunk.to_dict()
        assert data["chunk_id"] == "chunk-002"
        assert data["content"] == "test content"
        assert "metadata" in data


class TestVectorSearchResult:
    """Test VectorSearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating search result"""
        result = VectorSearchResult(
            chunk_id="result-001",
            similarity_score=0.95,
            metadata={"language": "java"},
            content="public void test()"
        )

        assert result.chunk_id == "result-001"
        assert result.similarity_score == 0.95
        assert result.metadata == {"language": "java"}
        assert result.embedding_vector is None

    def test_search_result_to_dict(self):
        """Test search result to_dict method"""
        result = VectorSearchResult(
            chunk_id="result-002",
            similarity_score=0.85,
            metadata={},
            content="test"
        )

        data = result.to_dict()
        assert data["chunk_id"] == "result-002"
        assert data["similarity_score"] == 0.85


class TestVectorSearchQuery:
    """Test VectorSearchQuery dataclass"""

    def test_query_with_vector(self):
        """Test query with vector"""
        query = VectorSearchQuery(
            query_vector=[0.1] * 1024,
            top_k=5,
            min_similarity=0.7
        )

        assert len(query.query_vector) == 1024
        assert query.top_k == 5
        assert query.min_similarity == 0.7

    def test_query_with_text(self):
        """Test query with text"""
        query = VectorSearchQuery(
            query_text="find authentication logic",
            top_k=10
        )

        assert query.query_text == "find authentication logic"
        assert query.top_k == 10

    def test_query_requires_vector_or_text(self):
        """Test that query requires either vector or text"""
        with pytest.raises(ValueError):
            VectorSearchQuery(top_k=5)


class TestDatabaseStats:
    """Test DatabaseStats dataclass"""

    def test_stats_defaults(self):
        """Test default stats values"""
        stats = DatabaseStats()

        assert stats.total_chunks == 0
        assert stats.total_files == 0
        assert stats.language_counts == {}
        assert stats.layer_counts == {}

    def test_stats_custom_values(self):
        """Test custom stats values"""
        stats = DatabaseStats(
            total_chunks=1000,
            total_files=50,
            language_counts={"java": 800, "kotlin": 200},
            layer_counts={"controller": 100, "service": 400}
        )

        assert stats.total_chunks == 1000
        assert stats.language_counts["java"] == 800

    def test_stats_to_dict(self):
        """Test stats to_dict method"""
        stats = DatabaseStats(total_chunks=500)

        data = stats.to_dict()
        assert data["total_chunks"] == 500
        assert "language_counts" in data


class TestBulkOperationResult:
    """Test BulkOperationResult dataclass"""

    def test_result_creation(self):
        """Test creating bulk operation result"""
        result = BulkOperationResult(
            operation_type="insert",
            total_items=100,
            successful_items=95,
            failed_items=5,
            processing_time=2.5
        )

        assert result.operation_type == "insert"
        assert result.total_items == 100
        assert result.successful_items == 95
        assert result.failed_items == 5

    def test_success_rate(self):
        """Test success rate calculation"""
        result = BulkOperationResult(
            operation_type="delete",
            total_items=100,
            successful_items=80,
            failed_items=20,
            processing_time=1.0
        )

        assert result.success_rate == 0.8

    def test_success_rate_zero_items(self):
        """Test success rate with zero items"""
        result = BulkOperationResult(
            operation_type="update",
            total_items=0,
            successful_items=0,
            failed_items=0,
            processing_time=0.0
        )

        assert result.success_rate == 0.0

    def test_result_to_dict(self):
        """Test result to_dict method"""
        result = BulkOperationResult(
            operation_type="insert",
            total_items=50,
            successful_items=50,
            failed_items=0,
            processing_time=1.5,
            errors=[]
        )

        data = result.to_dict()
        assert data["operation_type"] == "insert"
        assert data["success_rate"] == 1.0

    def test_result_with_errors(self):
        """Test bulk operation result with errors"""
        result = BulkOperationResult(
            operation_type="insert",
            total_items=10,
            successful_items=5,
            failed_items=5,
            processing_time=1.0,
            errors=["Error 1", "Error 2"]
        )

        assert len(result.errors) == 2
        assert result.success_rate == 0.5


class TestVectorStoreIntegration:
    """Test VectorStore class with mocked dependencies"""

    def test_vector_store_config_validation(self):
        """Test VectorStore validates config"""
        from src.database.vector_store import VectorStore

        # Invalid config should raise
        with pytest.raises(ValueError):
            VectorStore(VectorDBConfig(collection_name=""))

    @patch('src.database.chroma_client.chromadb')
    def test_vector_store_initialization(self, mock_chromadb):
        """Test VectorStore initialization"""
        from src.database.vector_store import VectorStore

        config = VectorDBConfig(collection_name="test_collection")
        store = VectorStore(config)

        assert store.config == config
        assert store._is_connected is False

    @patch('src.database.chroma_client.chromadb')
    def test_vector_store_health_check_format(self, mock_chromadb):
        """Test health check returns proper format"""
        from src.database.vector_store import VectorStore

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        config = VectorDBConfig(collection_name="test_collection")
        store = VectorStore(config)

        health = store.health_check()

        assert "vector_store_status" in health
        assert "database_health" in health
        assert "configuration" in health
