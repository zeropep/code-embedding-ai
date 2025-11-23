"""
Tests for REST API endpoints and web interface
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from src.api.main import create_app
from src.api.models import (
    ProcessRequest, SearchRequest, SearchResponse,
    ProcessStatus, HealthResponse, StatsResponse
)


class TestAPIModels:
    """Test API request/response models"""

    def test_process_request_model(self):
        """Test ProcessRequest model validation"""
        # Valid request
        valid_request = ProcessRequest(
            repository_path="/path/to/repo",
            include_security_scan=True,
            chunk_size_override=200
        )

        assert valid_request.repository_path == "/path/to/repo"
        assert valid_request.include_security_scan is True
        assert valid_request.chunk_size_override == 200

        # Test defaults
        minimal_request = ProcessRequest(repository_path="/path/to/repo")
        assert minimal_request.include_security_scan is True
        assert minimal_request.force_reprocess is False

    def test_search_request_model(self):
        """Test SearchRequest model validation"""
        request = SearchRequest(
            query="find user authentication",
            limit=10,
            language_filter="java",
            layer_filter="service"
        )

        assert request.query == "find user authentication"
        assert request.limit == 10
        assert request.language_filter == "java"

    def test_search_response_model(self):
        """Test SearchResponse model"""
        response = SearchResponse(
            results=[],
            total_results=0,
            query_time=0.05,
            search_metadata={"model": "test-model"}
        )

        assert response.results == []
        assert response.total_results == 0
        assert response.query_time == 0.05

    def test_health_response_model(self):
        """Test HealthResponse model"""
        health = HealthResponse(
            status="healthy",
            components={
                "database": {"status": "healthy"},
                "embedding_service": {"status": "healthy"}
            },
            timestamp="2024-01-01T12:00:00Z"
        )

        assert health.status == "healthy"
        assert "database" in health.components


class TestAPIEndpoints:
    """Test REST API endpoints"""

    @pytest.fixture
    def mock_services(self, mock_embedding_service, mock_vector_store, mock_security_scanner):
        """Create mock services for API testing"""
        return {
            "embedding_service": mock_embedding_service,
            "vector_store": mock_vector_store,
            "security_scanner": mock_security_scanner
        }

    @pytest.fixture
    def test_client(self, mock_services):
        """Create test client with mocked services"""
        app = create_app()

        # Inject mock services
        app.state.embedding_service = mock_services["embedding_service"]
        app.state.vector_store = mock_services["vector_store"]
        app.state.security_scanner = mock_services["security_scanner"]

        return TestClient(app)

    def test_health_endpoint(self, test_client, mock_services):
        """Test health check endpoint"""
        # Mock health checks
        mock_services["embedding_service"].health_check.return_value = {"status": "healthy"}
        mock_services["vector_store"].health_check.return_value = {"status": "healthy"}

        response = test_client.get("/health")

        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        assert "components" in health_data
        assert "timestamp" in health_data

    def test_process_repository_endpoint(self, test_client, mock_git_repo):
        """Test repository processing endpoint"""
        request_data = {
            "repository_path": str(mock_git_repo),
            "include_security_scan": True,
            "force_reprocess": False
        }

        # Mock the processing pipeline
        with patch('src.embeddings.embedding_pipeline.EmbeddingPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.process_repository = AsyncMock(return_value={
                "status": "success",
                "chunks_processed": 10,
                "processing_time": 5.2
            })
            mock_pipeline_class.return_value = mock_pipeline

            response = test_client.post("/api/v1/process", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            assert result["chunks_processed"] == 10

    def test_search_endpoint(self, test_client, mock_services):
        """Test semantic search endpoint"""
        search_request = {
            "query": "user authentication logic",
            "limit": 5,
            "similarity_threshold": 0.7
        }

        # Mock embedding generation and search
        mock_services["embedding_service"].generate_embedding = AsyncMock(return_value=Mock(
            vector=[0.1, 0.2, 0.3],
            status="completed"
        ))

        mock_services["vector_store"].search_similar_chunks.return_value = [
            {
                "chunk_id": "chunk1",
                "content": "authentication code",
                "similarity": 0.85,
                "metadata": {"file": "AuthService.java", "language": "java"}
            }
        ]

        response = test_client.post("/api/v1/search", json=search_request)

        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["results"][0]["similarity"] == 0.85

    def test_get_chunk_endpoint(self, test_client, mock_services):
        """Test getting specific chunk by ID"""
        chunk_id = "test_chunk_123"

        mock_services["vector_store"].get_chunk_by_id.return_value = {
            "chunk_id": chunk_id,
            "content": "test code content",
            "metadata": {
                "file": "TestClass.java",
                "language": "java",
                "class_name": "TestClass"
            }
        }

        response = test_client.get(f"/api/v1/chunks/{chunk_id}")

        assert response.status_code == 200
        chunk_data = response.json()
        assert chunk_data["chunk_id"] == chunk_id
        assert "content" in chunk_data
        assert "metadata" in chunk_data

    def test_get_chunk_not_found(self, test_client, mock_services):
        """Test getting non-existent chunk"""
        mock_services["vector_store"].get_chunk_by_id.return_value = None

        response = test_client.get("/api/v1/chunks/nonexistent")

        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data

    def test_stats_endpoint(self, test_client, mock_services):
        """Test statistics endpoint"""
        # Mock stats from various services
        mock_services["vector_store"].get_collection_info.return_value = {
            "total_chunks": 1500,
            "collection_name": "code_embeddings"
        }
        mock_services["embedding_service"].get_metrics.return_value = {
            "total_requests": 100,
            "successful_requests": 95,
            "average_processing_time": 1.2
        }

        response = test_client.get("/api/v1/stats")

        assert response.status_code == 200
        stats = response.json()
        assert "database" in stats
        assert "embedding_service" in stats
        assert stats["database"]["total_chunks"] == 1500

    def test_list_files_endpoint(self, test_client, mock_services):
        """Test listing processed files"""
        mock_services["vector_store"].search_similar_chunks.return_value = []

        # Mock getting unique files from database
        with patch('src.database.chroma_store.ChromaVectorStore.get_unique_files') as mock_get_files:
            mock_get_files.return_value = [
                {"file_path": "src/User.java", "chunk_count": 5, "last_modified": "2024-01-01T12:00:00Z"},
                {"file_path": "src/UserService.java", "chunk_count": 3, "last_modified": "2024-01-01T12:30:00Z"}
            ]

            response = test_client.get("/api/v1/files")

            assert response.status_code == 200
            files = response.json()
            assert len(files["files"]) == 2
            assert files["files"][0]["file_path"] == "src/User.java"

    def test_delete_chunks_endpoint(self, test_client, mock_services):
        """Test deleting chunks by file path"""
        delete_request = {
            "file_path": "src/OldClass.java"
        }

        mock_services["vector_store"].delete_chunks.return_value = True

        response = test_client.delete("/api/v1/chunks", json=delete_request)

        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Chunks deleted successfully"

    def test_update_monitoring_endpoint(self, test_client):
        """Test update monitoring endpoints"""
        # Mock update manager
        with patch('src.updates.update_manager.UpdateManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.start_monitoring = AsyncMock()
            mock_manager.get_update_stats.return_value = {
                "total_updates_processed": 10,
                "last_update_time": "2024-01-01T12:00:00Z"
            }
            mock_manager_class.return_value = mock_manager

            # Test starting monitoring
            response = test_client.post("/api/v1/monitoring/start")
            assert response.status_code == 200

            # Test getting monitoring stats
            response = test_client.get("/api/v1/monitoring/stats")
            assert response.status_code == 200
            stats = response.json()
            assert "total_updates_processed" in stats

    def test_error_handling(self, test_client, mock_services):
        """Test API error handling"""
        # Test invalid JSON
        response = test_client.post("/api/v1/process", data="invalid json")
        assert response.status_code == 422

        # Test missing required fields
        response = test_client.post("/api/v1/process", json={})
        assert response.status_code == 422

        # Test internal server error
        mock_services["embedding_service"].generate_embedding = AsyncMock(
            side_effect=Exception("Service error")
        )

        search_request = {"query": "test query"}
        response = test_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 500

    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set"""
        response = test_client.options("/api/v1/health")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_request_validation(self, test_client):
        """Test request validation and error messages"""
        # Test search with invalid limit
        invalid_search = {
            "query": "test",
            "limit": -1  # Invalid negative limit
        }

        response = test_client.post("/api/v1/search", json=invalid_search)
        assert response.status_code == 422
        error_detail = response.json()
        assert "detail" in error_detail

    def test_authentication_middleware(self, test_client):
        """Test API authentication if enabled"""
        # Test with API key authentication
        with patch('src.api.main.API_KEY_REQUIRED', True):
            # Request without API key should fail
            response = test_client.get("/api/v1/stats")
            assert response.status_code == 401

            # Request with valid API key should succeed
            headers = {"X-API-Key": "test-api-key"}
            with patch('src.api.main.VALID_API_KEYS', {"test-api-key"}):
                response = test_client.get("/api/v1/stats", headers=headers)
                assert response.status_code == 200

    def test_rate_limiting(self, test_client):
        """Test API rate limiting if enabled"""
        # Mock rate limiting
        with patch('src.api.middleware.rate_limit_check') as mock_rate_limit:
            mock_rate_limit.return_value = True  # Rate limit exceeded

            response = test_client.get("/api/v1/health")
            # Should still work as health check is usually exempt from rate limiting
            assert response.status_code == 200

    def test_streaming_search_results(self, test_client, mock_services):
        """Test streaming search results for large result sets"""
        # Mock large result set
        large_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"code content {i}",
                "similarity": 0.9 - (i * 0.1),
                "metadata": {"file": f"File{i}.java"}
            }
            for i in range(100)
        ]

        mock_services["vector_store"].search_similar_chunks.return_value = large_results

        search_request = {
            "query": "test query",
            "limit": 100,
            "stream": True
        }

        response = test_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200

        # For streaming, check that response is properly formatted
        result = response.json()
        assert "results" in result
        assert len(result["results"]) <= 100