"""
Tests for database and vector store functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.database.models import VectorDBConfig
from src.database.chroma_store import ChromaVectorStore
from src.database.vector_store import VectorStore
from src.code_parser.models import CodeChunk, CodeLanguage, LayerType


class TestVectorDBConfig:
    """Test vector database configuration"""

    def test_vector_config_defaults(self):
        """Test default configuration values"""
        config = VectorDBConfig()

        assert config.collection_name == "code_embeddings"
        assert config.persistent is True
        assert config.max_batch_size == 100
        assert config.distance_metric == "cosine"

    def test_vector_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = VectorDBConfig(
            collection_name="test_collection",
            persistent=True,
            max_batch_size=50
        )
        assert valid_config.validate() is True

        # Invalid config - empty collection name
        invalid_config = VectorDBConfig(collection_name="")
        assert invalid_config.validate() is False

        # Invalid config - bad batch size
        invalid_config2 = VectorDBConfig(max_batch_size=0)
        assert invalid_config2.validate() is False


class TestChromaVectorStore:
    """Test ChromaDB vector store implementation"""

    @pytest.fixture
    def vector_store(self, vector_config):
        """Create vector store for testing"""
        return ChromaVectorStore(vector_config)

    def test_chroma_store_initialization(self, vector_config):
        """Test ChromaVectorStore initialization"""
        store = ChromaVectorStore(vector_config)

        assert store.config == vector_config
        assert store.client is None
        assert store.collection is None
        assert store._is_connected is False

    @patch('chromadb.PersistentClient')
    def test_connect_persistent(self, mock_client, vector_store):
        """Test connection to persistent ChromaDB"""
        # Mock ChromaDB client and collection
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        # Connect to database
        result = vector_store.connect()

        assert result is True
        assert vector_store._is_connected is True
        assert vector_store.client is not None
        assert vector_store.collection is not None

        # Verify client creation
        mock_client.assert_called_once_with(path=vector_store.config.persist_directory)

    @patch('chromadb.Client')
    def test_connect_in_memory(self, mock_client, vector_config):
        """Test connection to in-memory ChromaDB"""
        config = vector_config
        config.persistent = False
        store = ChromaVectorStore(config)

        # Mock ChromaDB client and collection
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        # Connect to database
        result = store.connect()

        assert result is True
        assert store._is_connected is True

        # Verify in-memory client creation
        mock_client.assert_called_once()

    def test_disconnect(self, vector_store):
        """Test disconnection from database"""
        # Set up connected state
        vector_store.client = Mock()
        vector_store.collection = Mock()
        vector_store._is_connected = True

        # Disconnect
        vector_store.disconnect()

        assert vector_store.client is None
        assert vector_store.collection is None
        assert vector_store._is_connected is False

    def test_store_chunks(self, vector_store, create_test_chunks):
        """Test storing code chunks with embeddings"""
        # Mock connection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Create test chunks with embeddings
        chunks = create_test_chunks(3)
        for i, chunk in enumerate(chunks):
            chunk.metadata["embedding"] = {
                "vector": [0.1 * i, 0.2 * i, 0.3 * i],
                "model_version": "test-model"
            }

        # Store chunks
        result = vector_store.store_chunks(chunks)

        assert result is True
        vector_store.collection.add.assert_called_once()

        # Verify call arguments
        call_args = vector_store.collection.add.call_args[1]
        assert len(call_args["ids"]) == 3
        assert len(call_args["embeddings"]) == 3
        assert len(call_args["documents"]) == 3
        assert len(call_args["metadatas"]) == 3

    def test_search_similar_chunks(self, vector_store):
        """Test searching for similar code chunks"""
        # Mock connection and collection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Mock search results
        mock_results = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.3]],
            "documents": [["code1", "code2"]],
            "metadatas": [[{"file": "test1.java"}, {"file": "test2.java"}]]
        }
        vector_store.collection.query.return_value = mock_results

        # Perform search
        query_vector = [0.1, 0.2, 0.3]
        results = vector_store.search_similar_chunks(query_vector, limit=2)

        assert len(results) == 2
        assert results[0]["similarity"] > results[1]["similarity"]  # Higher similarity first
        assert "chunk_id" in results[0]
        assert "content" in results[0]
        assert "metadata" in results[0]

        # Verify query call
        vector_store.collection.query.assert_called_once_with(
            query_embeddings=[query_vector],
            n_results=2
        )

    def test_get_chunk_by_id(self, vector_store):
        """Test retrieving chunk by ID"""
        # Mock connection and collection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Mock get results
        mock_results = {
            "ids": ["chunk1"],
            "documents": ["test code"],
            "metadatas": [{"file": "test.java", "language": "java"}],
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        vector_store.collection.get.return_value = mock_results

        # Get chunk
        chunk = vector_store.get_chunk_by_id("chunk1")

        assert chunk is not None
        assert chunk["chunk_id"] == "chunk1"
        assert chunk["content"] == "test code"
        assert chunk["metadata"]["file"] == "test.java"

    def test_update_chunk(self, vector_store):
        """Test updating existing chunk"""
        # Mock connection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Update chunk
        chunk_id = "test_chunk"
        new_embedding = [0.4, 0.5, 0.6]
        new_metadata = {"updated": True}

        result = vector_store.update_chunk(chunk_id, new_embedding, new_metadata)

        assert result is True
        vector_store.collection.update.assert_called_once_with(
            ids=[chunk_id],
            embeddings=[new_embedding],
            metadatas=[new_metadata]
        )

    def test_delete_chunks(self, vector_store):
        """Test deleting chunks"""
        # Mock connection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Delete chunks
        chunk_ids = ["chunk1", "chunk2"]
        result = vector_store.delete_chunks(chunk_ids)

        assert result is True
        vector_store.collection.delete.assert_called_once_with(ids=chunk_ids)

    def test_get_collection_info(self, vector_store):
        """Test getting collection information"""
        # Mock connection and collection
        vector_store._is_connected = True
        vector_store.collection = Mock()
        vector_store.collection.count.return_value = 100

        # Get info
        info = vector_store.get_collection_info()

        assert "total_chunks" in info
        assert info["total_chunks"] == 100
        assert "collection_name" in info

    def test_health_check(self, vector_store):
        """Test vector store health check"""
        # Test when disconnected
        vector_store._is_connected = False
        health = vector_store.health_check()
        assert health["vector_store_status"] == "disconnected"

        # Test when connected
        vector_store._is_connected = True
        vector_store.collection = Mock()
        vector_store.collection.count.return_value = 50

        health = vector_store.health_check()
        assert health["vector_store_status"] == "healthy"
        assert health["total_chunks"] == 50

    def test_batch_operations(self, vector_store, create_test_chunks):
        """Test batch operations with large datasets"""
        # Mock connection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Create large batch of chunks
        chunks = create_test_chunks(150)  # Exceeds batch size
        for i, chunk in enumerate(chunks):
            chunk.metadata["embedding"] = {
                "vector": [float(i)] * 3,
                "model_version": "test-model"
            }

        # Store chunks (should be batched)
        result = vector_store.store_chunks(chunks)

        assert result is True
        # Should be called multiple times due to batching
        assert vector_store.collection.add.call_count >= 2

    def test_error_handling(self, vector_store):
        """Test error handling in database operations"""
        # Test connection error
        with patch('chromadb.PersistentClient', side_effect=Exception("Connection failed")):
            result = vector_store.connect()
            assert result is False

        # Test operation on disconnected store
        vector_store._is_connected = False
        result = vector_store.store_chunks([])
        assert result is False

    def test_filter_search(self, vector_store):
        """Test filtered search functionality"""
        # Mock connection and collection
        vector_store._is_connected = True
        vector_store.collection = Mock()

        # Mock filtered search results
        mock_results = {
            "ids": [["chunk1"]],
            "distances": [[0.2]],
            "documents": [["filtered code"]],
            "metadatas": [[{"language": "java", "layer_type": "service"}]]
        }
        vector_store.collection.query.return_value = mock_results

        # Perform filtered search
        query_vector = [0.1, 0.2, 0.3]
        filter_criteria = {"language": "java"}
        results = vector_store.search_similar_chunks(
            query_vector,
            limit=1,
            filter_metadata=filter_criteria
        )

        assert len(results) == 1
        assert results[0]["metadata"]["language"] == "java"

        # Verify query with filter
        vector_store.collection.query.assert_called_once_with(
            query_embeddings=[query_vector],
            n_results=1,
            where=filter_criteria
        )


class TestVectorStore:
    """Test abstract vector store interface"""

    def test_vector_store_interface(self, vector_config):
        """Test vector store abstract interface"""
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            VectorStore(vector_config)

    def test_chroma_store_implements_interface(self, vector_config):
        """Test that ChromaVectorStore properly implements interface"""
        store = ChromaVectorStore(vector_config)

        # Check that all required methods are implemented
        assert hasattr(store, 'connect')
        assert hasattr(store, 'disconnect')
        assert hasattr(store, 'store_chunks')
        assert hasattr(store, 'search_similar_chunks')
        assert hasattr(store, 'health_check')

        # Check that methods are callable
        assert callable(getattr(store, 'connect'))
        assert callable(getattr(store, 'disconnect'))
        assert callable(getattr(store, 'store_chunks'))
        assert callable(getattr(store, 'search_similar_chunks'))
        assert callable(getattr(store, 'health_check'))