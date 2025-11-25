from functools import lru_cache
from typing import Optional
import os
import structlog

from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..code_parser.models import ParserConfig
from ..security.models import SecurityConfig
from ..embeddings.models import EmbeddingConfig
from ..database.models import VectorDBConfig
from ..updates.update_service import UpdateService
from ..updates.models import UpdateConfig
from ..database.vector_store import VectorStore


logger = structlog.get_logger(__name__)


class ServiceManager:
    """Singleton service manager for API dependencies"""
    _instance: Optional['ServiceManager'] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.embedding_pipeline: Optional[EmbeddingPipeline] = None
            self.update_service: Optional[UpdateService] = None
            self.vector_store: Optional[VectorStore] = None
            self._initialized = True

    async def initialize_services(self):
        """Initialize all services"""
        try:
            logger.info("Initializing services")

            # Load configurations from environment
            parser_config = self._create_parser_config()
            security_config = self._create_security_config()
            embedding_config = self._create_embedding_config()
            vector_config = self._create_vector_config()
            update_config = self._create_update_config()

            # Initialize vector store
            self.vector_store = VectorStore(vector_config)
            if not self.vector_store.connect():
                raise RuntimeError("Failed to connect to vector store")

            # Initialize embedding pipeline
            self.embedding_pipeline = EmbeddingPipeline(
                parser_config=parser_config,
                security_config=security_config,
                embedding_config=embedding_config,
                vector_config=vector_config,
                auto_save=True
            )

            # Initialize update service
            repo_path = os.getenv("REPO_PATH", "./target_repo")
            state_dir = os.getenv("UPDATE_STATE_DIR", "./update_state")

            self.update_service = UpdateService(
                repo_path=repo_path,
                state_dir=state_dir,
                parser_config=parser_config,
                security_config=security_config,
                embedding_config=embedding_config,
                vector_config=vector_config,
                update_config=update_config
            )

            # Start update service
            await self.update_service.start()

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize services", error=str(e))
            raise

    async def shutdown_services(self):
        """Shutdown all services"""
        try:
            logger.info("Shutting down services")

            if self.update_service:
                await self.update_service.stop()

            if self.vector_store:
                self.vector_store.disconnect()

            logger.info("All services shut down successfully")

        except Exception as e:
            logger.error("Failed to shutdown services", error=str(e))

    def _create_parser_config(self) -> ParserConfig:
        """Create parser configuration from environment"""
        return ParserConfig(
            min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "50")),
            max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "500")),
            overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "20")),
            include_comments=os.getenv("INCLUDE_COMMENTS", "false").lower() == "true",
            supported_extensions=os.getenv(
                "SUPPORTED_EXTENSIONS", ".java,.kt,.html,.xml,.yml,.yaml,.properties").split(",")
        )

    def _create_security_config(self) -> SecurityConfig:
        """Create security configuration from environment"""
        return SecurityConfig(
            enabled=os.getenv("ENABLE_SECRET_SCANNING", "true").lower() == "true",
            preserve_syntax=os.getenv("PRESERVE_SYNTAX", "true").lower() == "true",
            sensitivity_threshold=float(os.getenv("SENSITIVITY_THRESHOLD", "0.7")),
            scan_comments=os.getenv("SCAN_COMMENTS", "true").lower() == "true",
            scan_strings=os.getenv("SCAN_STRINGS", "true").lower() == "true"
        )

    def _create_embedding_config(self) -> EmbeddingConfig:
        """Create embedding configuration from environment"""
        return EmbeddingConfig(
            api_key=os.getenv("JINA_API_KEY", ""),
            api_url=os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings"),
            model_name=os.getenv("EMBEDDING_MODEL", "jina-code-embeddings-1.5b"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "20")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "10")),
            timeout=int(os.getenv("EMBEDDING_TIMEOUT", "30")),
            enable_caching=os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
        )

    def _create_vector_config(self) -> VectorDBConfig:
        """Create vector database configuration from environment"""
        return VectorDBConfig(
            host=os.getenv("CHROMADB_HOST", "localhost"),
            port=int(os.getenv("CHROMADB_PORT", "8000")),
            collection_name=os.getenv("CHROMADB_COLLECTION_NAME", "code_embeddings"),
            persistent=os.getenv("CHROMADB_PERSISTENT", "true").lower() == "true",
            persist_directory=os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db"),
            max_batch_size=int(os.getenv("CHROMADB_BATCH_SIZE", "100"))
        )

    def _create_update_config(self) -> UpdateConfig:
        """Create update configuration from environment"""
        return UpdateConfig(
            check_interval_seconds=int(os.getenv("UPDATE_CHECK_INTERVAL", "300")),
            max_concurrent_updates=int(os.getenv("MAX_CONCURRENT_UPDATES", "3")),
            enable_file_watching=os.getenv("ENABLE_FILE_WATCHING", "false").lower() == "true",
            force_update_threshold_hours=int(os.getenv("FORCE_UPDATE_THRESHOLD_HOURS", "24"))
        )


# Global service manager instance
service_manager = ServiceManager()


# Dependency functions
async def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get embedding pipeline dependency"""
    if not service_manager.embedding_pipeline:
        await service_manager.initialize_services()
    return service_manager.embedding_pipeline


async def get_update_service() -> UpdateService:
    """Get update service dependency"""
    if not service_manager.update_service:
        await service_manager.initialize_services()
    return service_manager.update_service


async def get_vector_store() -> VectorStore:
    """Get vector store dependency"""
    if not service_manager.vector_store:
        await service_manager.initialize_services()
    return service_manager.vector_store


# Configuration dependencies
@lru_cache()
def get_parser_config() -> ParserConfig:
    """Get parser configuration"""
    return service_manager._create_parser_config()


@lru_cache()
def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return service_manager._create_security_config()


@lru_cache()
def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration"""
    return service_manager._create_embedding_config()


@lru_cache()
def get_vector_config() -> VectorDBConfig:
    """Get vector database configuration"""
    return service_manager._create_vector_config()


@lru_cache()
def get_update_config() -> UpdateConfig:
    """Get update configuration"""
    return service_manager._create_update_config()


# Health check dependencies
async def check_service_health() -> dict:
    """Check health of all services"""
    health_status = {
        "embedding_pipeline": False,
        "update_service": False,
        "vector_store": False
    }

    try:
        if service_manager.embedding_pipeline:
            pipeline_health = await service_manager.embedding_pipeline.health_check()
            health_status["embedding_pipeline"] = pipeline_health.get("overall_status") == "healthy"

        if service_manager.update_service:
            update_health = await service_manager.update_service.health_check()
            health_status["update_service"] = update_health.get("status") == "healthy"

        if service_manager.vector_store:
            vector_health = service_manager.vector_store.health_check()
            health_status["vector_store"] = vector_health.get("vector_store_status") == "healthy"

    except Exception as e:
        logger.error("Health check failed", error=str(e))

    return health_status


# Startup and shutdown event handlers
async def startup_event():
    """Application startup event"""
    try:
        logger.info("Starting Code Embedding AI API")
        await service_manager.initialize_services()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error("API startup failed", error=str(e))
        raise


async def shutdown_event():
    """Application shutdown event"""
    try:
        logger.info("Shutting down Code Embedding AI API")
        await service_manager.shutdown_services()
        logger.info("API shutdown completed successfully")
    except Exception as e:
        logger.error("API shutdown failed", error=str(e))
