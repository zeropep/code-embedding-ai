"""
Tests for embedding generation functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import aiohttp

from src.embeddings.models import EmbeddingConfig, EmbeddingRequest, EmbeddingResult, EmbeddingStatus
from src.embeddings.jina_client import JinaEmbeddingClient
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.embedding_pipeline import EmbeddingPipeline


class TestEmbeddingModels:
    """Test embedding model classes"""

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig validation"""
        # Valid config
        valid_config = EmbeddingConfig(
            api_key="test_key",
            api_url="https://api.test.com",
            batch_size=10,
            max_concurrent_requests=5
        )
        assert valid_config.validate() is True

        # Invalid config - no API key
        invalid_config = EmbeddingConfig(api_key="")
        assert invalid_config.validate() is False

        # Invalid config - bad batch size
        invalid_config2 = EmbeddingConfig(api_key="test", batch_size=0)
        assert invalid_config2.validate() is False

    def test_embedding_request_creation(self):
        """Test EmbeddingRequest creation"""
        request = EmbeddingRequest(
            id="test_id",
            content="test content",
            metadata={"file": "test.java"}
        )

        assert request.id == "test_id"
        assert request.content == "test content"
        assert request.metadata["file"] == "test.java"
        assert request.created_at is not None

    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation and serialization"""
        result = EmbeddingResult(
            request_id="test_id",
            vector=[0.1, 0.2, 0.3],
            status=EmbeddingStatus.COMPLETED,
            processing_time=1.5,
            model_version="test-model"
        )

        assert result.status == EmbeddingStatus.COMPLETED
        assert len(result.vector) == 3
        assert result.processing_time == 1.5

        result_dict = result.to_dict()
        assert "request_id" in result_dict
        assert "vector" in result_dict
        assert "status" in result_dict
        assert result_dict["status"] == "completed"


class TestJinaEmbeddingClient:
    """Test Jina AI API client"""

    @pytest.fixture
    def embedding_config(self):
        """Embedding configuration for tests"""
        return EmbeddingConfig(
            api_key="test_api_key",
            api_url="https://api.test.com/embeddings",
            model_name="test-model",
            batch_size=5,
            timeout=10
        )

    def test_jina_client_initialization(self, embedding_config):
        """Test JinaEmbeddingClient initialization"""
        client = JinaEmbeddingClient(embedding_config)

        assert client.config == embedding_config
        assert client.session is None
        assert isinstance(client._cache, dict)

    @pytest.mark.asyncio
    async def test_jina_client_session_management(self, embedding_config):
        """Test session creation and cleanup"""
        client = JinaEmbeddingClient(embedding_config)

        # Test session creation
        await client._ensure_session()
        assert client.session is not None

        # Test session cleanup
        await client.close()
        assert client.session is None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_generate_embedding_success(self, mock_post, embedding_config):
        """Test successful embedding generation"""
        # Mock response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]
        })

        mock_post.return_value.__aenter__.return_value = mock_response

        client = JinaEmbeddingClient(embedding_config)

        async with client:
            result = await client.generate_embedding("test content", "test_id")

            assert result.status == EmbeddingStatus.COMPLETED
            assert result.vector == [0.1, 0.2, 0.3, 0.4]
            assert result.request_id == "test_id"

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_generate_embedding_api_error(self, mock_post, embedding_config):
        """Test embedding generation with API error"""
        # Mock error response
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_post.return_value.__aenter__.return_value = mock_response

        client = JinaEmbeddingClient(embedding_config)

        async with client:
            result = await client.generate_embedding("test content", "test_id")

            assert result.status == EmbeddingStatus.FAILED
            assert result.vector is None
            assert result.error_message is not None

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_generate_embeddings_batch(self, mock_post, embedding_config):
        """Test batch embedding generation"""
        # Mock response with multiple embeddings
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
                {"embedding": [0.5, 0.6]}
            ]
        })

        mock_post.return_value.__aenter__.return_value = mock_response

        client = JinaEmbeddingClient(embedding_config)

        contents = ["content1", "content2", "content3"]
        request_ids = ["id1", "id2", "id3"]

        async with client:
            results = await client.generate_embeddings_batch(contents, request_ids)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.status == EmbeddingStatus.COMPLETED
                assert result.request_id == request_ids[i]
                assert len(result.vector) == 2

    def test_embedding_cache(self, embedding_config):
        """Test embedding caching functionality"""
        config = embedding_config
        config.enable_caching = True

        client = JinaEmbeddingClient(config)

        # Test cache storage and retrieval
        test_content = "test content for caching"
        test_vector = [0.1, 0.2, 0.3]

        # Store in cache
        client._cache_embedding(test_content, test_vector)

        # Retrieve from cache
        cached_vector = client._get_cached_embedding(test_content)
        assert cached_vector == test_vector

        # Test cache miss
        cached_vector2 = client._get_cached_embedding("different content")
        assert cached_vector2 is None

    def test_cache_disabled(self, embedding_config):
        """Test behavior when caching is disabled"""
        config = embedding_config
        config.enable_caching = False

        client = JinaEmbeddingClient(config)

        # Should not cache
        client._cache_embedding("test", [0.1, 0.2])
        cached = client._get_cached_embedding("test")
        assert cached is None


class TestEmbeddingService:
    """Test embedding service"""

    @pytest.fixture
    def embedding_service(self, embedding_config):
        """Create embedding service for tests"""
        return EmbeddingService(embedding_config)

    def test_embedding_service_initialization(self, embedding_config):
        """Test EmbeddingService initialization"""
        service = EmbeddingService(embedding_config)

        assert service.config == embedding_config
        assert service.client is not None
        assert service.metrics is not None

    @pytest.mark.asyncio
    async def test_embedding_service_lifecycle(self, embedding_service):
        """Test service start/stop lifecycle"""
        service = embedding_service

        # Test start
        await service.start()
        assert service._is_running is True

        # Test stop
        await service.stop()
        assert service._is_running is False

    @pytest.mark.asyncio
    @patch('src.embeddings.jina_client.JinaEmbeddingClient.generate_embeddings_batch')
    async def test_generate_chunk_embeddings(self, mock_generate_batch, embedding_service, create_test_chunks):
        """Test generating embeddings for code chunks"""
        # Mock successful embedding generation
        mock_generate_batch.return_value = [
            EmbeddingResult(
                request_id="test_id",
                vector=[0.1, 0.2, 0.3],
                status=EmbeddingStatus.COMPLETED,
                model_version="test-model"
            )
        ]

        chunks = create_test_chunks(1)
        service = embedding_service

        await service.start()
        try:
            embedded_chunks = await service.generate_chunk_embeddings(chunks)

            assert len(embedded_chunks) == 1
            assert "embedding" in embedded_chunks[0].metadata
            assert embedded_chunks[0].metadata["embedding"]["vector"] == [0.1, 0.2, 0.3]
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_embedding_service_error_handling(self, embedding_service, create_test_chunks):
        """Test error handling in embedding service"""
        chunks = create_test_chunks(1)
        service = embedding_service

        # Mock client to raise exception
        service.client.generate_embeddings_batch = AsyncMock(side_effect=Exception("API Error"))

        await service.start()
        try:
            embedded_chunks = await service.generate_chunk_embeddings(chunks)

            # Should handle error gracefully
            assert len(embedded_chunks) == 1
            # Chunk should have error metadata
            assert "embedding_error" in embedded_chunks[0].metadata
        finally:
            await service.stop()

    def test_embedding_service_metrics(self, embedding_service):
        """Test metrics collection"""
        service = embedding_service

        # Test initial metrics
        metrics = service.get_metrics()
        assert "total_requests" in metrics
        assert metrics["total_requests"] == 0

        # Update metrics manually for testing
        service.metrics.update_success(1.5, 1, 1)
        updated_metrics = service.get_metrics()
        assert updated_metrics["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_health_check(self, embedding_service):
        """Test service health check"""
        service = embedding_service

        # Mock client health check
        with patch.object(service.client, 'generate_embedding') as mock_generate:
            mock_generate.return_value = EmbeddingResult(
                request_id="health_check",
                vector=[0.1],
                status=EmbeddingStatus.COMPLETED
            )

            health = await service.health_check()

            assert "status" in health
            assert "api_accessible" in health


class TestEmbeddingPipeline:
    """Test embedding pipeline integration"""

    @pytest.fixture
    def embedding_pipeline(self, parser_config, security_config, embedding_config):
        """Create embedding pipeline for tests"""
        return EmbeddingPipeline(
            parser_config=parser_config,
            security_config=security_config,
            embedding_config=embedding_config
        )

    @pytest.mark.asyncio
    @patch('src.embeddings.embedding_service.EmbeddingService.generate_chunk_embeddings')
    @patch('src.security.security_scanner.SecurityScanner.scan_chunks')
    @patch('src.code_parser.code_parser.CodeParser.parse_repository_async')
    async def test_process_repository(self, mock_parse, mock_scan, mock_embed, embedding_pipeline, create_test_chunks):
        """Test full repository processing through pipeline"""
        # Mock parser output
        from src.code_parser.models import ParsedFile, CodeLanguage
        chunks = create_test_chunks(3)
        parsed_file = ParsedFile(
            file_path="test.java",
            language=CodeLanguage.JAVA,
            chunks=chunks,
            total_lines=100,
            file_hash="testhash",
            last_modified=0.0
        )
        mock_parse.return_value = [parsed_file]

        # Mock security scanner (pass through)
        mock_scan.side_effect = lambda x: x

        # Mock embedding service
        def add_embeddings(chunks):
            for chunk in chunks:
                chunk.metadata["embedding"] = {
                    "vector": [0.1, 0.2, 0.3],
                    "model_version": "test-model"
                }
            return chunks

        mock_embed.side_effect = add_embeddings

        pipeline = embedding_pipeline

        # Process repository
        result = await pipeline.process_repository("test_repo")

        assert result["status"] == "success"
        assert "processing_summary" in result
        assert "chunks" in result

        # Verify pipeline steps were called
        mock_parse.assert_called_once()
        mock_scan.assert_called_once()
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_health_check(self, embedding_pipeline):
        """Test pipeline health check"""
        pipeline = embedding_pipeline

        # Mock component health checks
        with patch.object(pipeline.embedding_service, 'health_check') as mock_health:
            mock_health.return_value = {"status": "healthy"}

            health = await pipeline.health_check()

            assert "overall_status" in health
            assert "components" in health

    def test_pipeline_stats(self, embedding_pipeline):
        """Test pipeline statistics"""
        pipeline = embedding_pipeline

        stats = pipeline.get_pipeline_stats()

        assert "parser" in stats
        assert "security" in stats
        assert "embedding" in stats

        # Check parser stats
        assert "supported_extensions" in stats["parser"]
        assert "min_tokens" in stats["parser"]

        # Check embedding stats
        assert "model" in stats["embedding"]
        assert "batch_size" in stats["embedding"]