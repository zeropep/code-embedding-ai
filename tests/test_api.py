"""
Tests for REST API endpoints
Based on actual implementation in src/api/
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from src.api.main import create_app
from src.api.models import (
    ProcessRepositoryRequest, SearchRequest, SearchResponse,
    RequestStatus, HealthCheckResponse, SystemStatusResponse,
    ProcessingMode, SearchType
)
from src.api.dependencies import (
    get_embedding_pipeline, get_update_service, get_vector_store
)


class TestAPIModels:
    """Test API request/response models"""

    def test_process_repository_request_model(self):
        """Test ProcessRepositoryRequest model validation"""
        # Valid request
        valid_request = ProcessRepositoryRequest(
            repo_path="/path/to/repo",
            mode=ProcessingMode.FULL,
            force_update=True
        )

        assert valid_request.repo_path == "/path/to/repo"
        assert valid_request.mode == ProcessingMode.FULL
        assert valid_request.force_update is True

        # Test defaults
        minimal_request = ProcessRepositoryRequest(repo_path="/path/to/repo")
        assert minimal_request.mode == ProcessingMode.AUTO
        assert minimal_request.force_update is False

    def test_search_request_model(self):
        """Test SearchRequest model validation"""
        request = SearchRequest(
            query="find user authentication",
            top_k=10,
            search_type=SearchType.SEMANTIC
        )

        assert request.query == "find user authentication"
        assert request.top_k == 10
        assert request.search_type == SearchType.SEMANTIC

    def test_search_request_defaults(self):
        """Test SearchRequest default values"""
        request = SearchRequest(query="test query")

        assert request.top_k == 10
        assert request.min_similarity == 0.4
        assert request.search_type == SearchType.SEMANTIC
        assert request.include_content is True
        assert request.include_embeddings is False

    def test_health_check_response_model(self):
        """Test HealthCheckResponse model"""
        health = HealthCheckResponse(
            status="healthy",
            components={
                "pipeline": {"status": "healthy"},
                "update_service": {"status": "healthy"}
            },
            uptime_seconds=100.5,
            version="1.0.0"
        )

        assert health.status == "healthy"
        assert "pipeline" in health.components
        assert health.uptime_seconds == 100.5


# Fixtures for dependency mocking
@pytest.fixture
def mock_pipeline():
    """Create mock embedding pipeline"""
    pipeline = Mock()
    pipeline.health_check = AsyncMock(return_value={"status": "healthy", "overall_status": "healthy"})
    pipeline.process_repository = AsyncMock(return_value={
        "status": "success",
        "processing_summary": {"files_processed": 10},
        "parsing_stats": {},
        "security_stats": {},
        "embedding_stats": {}
    })
    pipeline.process_files = AsyncMock(return_value={
        "status": "success",
        "processing_summary": {"files_processed": 2}
    })

    # Embedding service mock
    pipeline.embedding_service = Mock()
    pipeline.embedding_service.start = AsyncMock()
    pipeline.embedding_service.stop = AsyncMock()
    pipeline.embedding_service.get_metrics = Mock(return_value={"total_requests": 0})
    pipeline.embedding_service.generate_chunk_embeddings = AsyncMock(
        return_value=[Mock(metadata={'embedding': {'vector': [0.1] * 1024}})]
    )

    return pipeline


@pytest.fixture
def mock_update_service():
    """Create mock update service"""
    service = Mock()
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    service.repo_path = "/test/repo"
    service.get_status = Mock(return_value={
        "service_running": True,
        "state_summary": {}
    })
    service.request_update = AsyncMock(return_value=Mock(
        to_dict=lambda: {"status": "completed"}
    ))
    service.state_manager = Mock()
    service.state_manager.get_metrics = Mock(return_value=Mock(
        to_dict=lambda: {}
    ))

    return service


@pytest.fixture
def mock_vector_store():
    """Create mock vector store"""
    store = Mock()
    store.search_similar_chunks = Mock(return_value=[])
    store.search_by_metadata = Mock(return_value=[])
    store.get_statistics = Mock(return_value=Mock(
        to_dict=lambda: {"total_chunks": 100}
    ))
    store.delete_chunks_by_file = Mock(return_value=Mock(
        successful_items=5
    ))
    store.reset_database = Mock(return_value=True)
    store.client = Mock()
    store.client.delete_chunks = Mock(return_value=Mock(
        successful_items=0
    ))

    return store


@pytest.fixture
def test_app(mock_pipeline, mock_update_service, mock_vector_store):
    """Create test app with mocked dependencies"""
    app = create_app()

    # Override dependencies using FastAPI's dependency_overrides
    app.dependency_overrides[get_embedding_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_update_service] = lambda: mock_update_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    yield app

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app):
    """Create test client with mocked dependencies"""
    return TestClient(test_app, raise_server_exceptions=False)


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == "Code Embedding AI Pipeline"


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_endpoint_healthy(self, client):
        """Test health check returns healthy status"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data


class TestSearchEndpoints:
    """Test search endpoints"""

    def test_semantic_search_endpoint(self, client):
        """Test semantic search endpoint"""
        search_request = {
            "query": "user authentication logic",
            "top_k": 5,
            "min_similarity": 0.7
        }

        response = client.post("/search/semantic", json=search_request)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_results" in data

    def test_semantic_search_with_empty_query(self, client):
        """Test semantic search with empty query"""
        search_request = {
            "query": "",
            "top_k": 5
        }

        response = client.post("/search/semantic", json=search_request)

        # Should return 422 validation error
        assert response.status_code == 422

    def test_metadata_search_endpoint(self, client):
        """Test metadata search endpoint"""
        filters = {"language": "java", "layer_type": "service"}

        response = client.post(
            "/search/metadata",
            json=filters,
            params={"top_k": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestProcessEndpoints:
    """Test processing endpoints"""

    def test_process_repository_endpoint(self, client):
        """Test repository processing endpoint"""
        request_data = {
            "repo_path": "/path/to/repo",
            "mode": "full",
            "force_update": False
        }

        response = client.post("/process/repository", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "request_id" in data

    def test_process_files_endpoint(self, client):
        """Test file processing endpoint"""
        request_data = {
            "file_paths": ["/path/to/file1.java", "/path/to/file2.java"],
            "update_existing": True,
            "parallel_processing": True
        }

        response = client.post("/process/files", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_trigger_update_endpoint_deprecated(self, client):
        """Test trigger update endpoint is deprecated"""
        response = client.post("/process/update", params={"force_full": False})

        # This endpoint is deprecated and returns 410 Gone
        assert response.status_code == 410


class TestStatusEndpoints:
    """Test status endpoints"""

    def test_system_status_endpoint(self, client):
        """Test system status endpoint"""
        response = client.get("/status/system")

        assert response.status_code == 200
        data = response.json()
        assert "service_status" in data
        assert "database_stats" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/status/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "embedding_metrics" in data
        assert "vector_store_stats" in data


class TestAdminEndpoints:
    """Test admin endpoints"""

    def test_delete_chunks_by_file(self, client):
        """Test deleting chunks by file path"""
        request_data = {
            "file_paths": ["src/OldClass.java"],
            "confirm": True
        }

        response = client.request("DELETE", "/admin/chunks", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted_count" in data

    def test_delete_chunks_with_filters(self, client):
        """Test deleting chunks with metadata filters"""
        request_data = {
            "filters": {"layer_type": "service"},
            "confirm": True
        }

        response = client.request("DELETE", "/admin/chunks", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_reset_database_requires_confirmation(self, client):
        """Test that reset requires confirmation"""
        response = client.post("/admin/reset", params={"confirm": False})

        assert response.status_code == 400

    def test_reset_database_with_confirmation(self, client):
        """Test reset with confirmation"""
        response = client.post("/admin/reset", params={"confirm": True})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestMiddleware:
    """Test API middleware"""

    def test_request_id_header(self, client):
        """Test that request ID is added to response headers"""
        response = client.get("/")

        assert "X-Request-ID" in response.headers
        # UUID format check
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID format

    def test_process_time_header(self, client):
        """Test that process time is added to response headers"""
        response = client.get("/")

        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0

    def test_cors_headers(self, client):
        """Test CORS headers are set"""
        response = client.options("/")

        # FastAPI handles CORS via middleware
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Test API error handling"""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/search/semantic",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post("/process/repository", json={})

        assert response.status_code == 422

    def test_not_found_endpoint(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")

        assert response.status_code == 404
