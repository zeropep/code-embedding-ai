"""
Pytest configuration and shared fixtures for testing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
import asyncio

from src.code_parser.models import ParserConfig, CodeChunk, CodeLanguage, LayerType
from src.security.models import SecurityConfig
from src.embeddings.models import EmbeddingConfig
from src.database.models import VectorDBConfig
from src.updates.models import UpdateConfig
from src.monitoring.models import MonitoringConfig, LogLevel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_java_code():
    """Sample Java code for testing"""
    return """
package com.example.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    /**
     * Get user by ID
     */
    public User getUserById(Long id) {
        if (id == null || id <= 0) {
            throw new IllegalArgumentException("User ID must be positive");
        }
        return userRepository.findById(id)
                .orElseThrow(() -> new UserNotFoundException("User not found: " + id));
    }

    public User createUser(User user) {
        validateUser(user);
        return userRepository.save(user);
    }

    private void validateUser(User user) {
        if (user.getEmail() == null || user.getEmail().isEmpty()) {
            throw new ValidationException("Email is required");
        }
    }
}
"""


@pytest.fixture
def sample_html_code():
    """Sample HTML/Thymeleaf code for testing"""
    return """
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>User Profile</title>
</head>
<body>
    <div th:fragment="userInfo">
        <h2>User Information</h2>
        <form th:object="${user}" th:action="@{/users}" method="post">
            <div>
                <label for="name">Name:</label>
                <input type="text" th:field="*{name}" id="name" required>
            </div>
            <div>
                <label for="email">Email:</label>
                <input type="email" th:field="*{email}" id="email" required>
            </div>
            <button type="submit">Save User</button>
        </form>
    </div>

    <table th:if="${users}">
        <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            <tr th:each="user : ${users}">
                <td th:text="${user.name}">John Doe</td>
                <td th:text="${user.email}">john@example.com</td>
                <td>
                    <a th:href="@{/users/{id}(id=${user.id})}">View</a>
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>
"""


@pytest.fixture
def sample_code_chunk():
    """Create a sample code chunk for testing"""
    return CodeChunk(
        content="public String getName() { return this.name; }",
        file_path="src/main/java/com/example/User.java",
        language=CodeLanguage.JAVA,
        start_line=15,
        end_line=17,
        function_name="getName",
        class_name="User",
        layer_type=LayerType.ENTITY,
        token_count=12,
        metadata={
            "package": "com.example",
            "return_type": "String",
            "modifiers": ["public"]
        }
    )


@pytest.fixture
def sample_code_with_secrets():
    """Sample code containing secrets for security testing"""
    return """
public class DatabaseConfig {
    private static final String DB_URL = "jdbc:mysql://localhost:3306/testdb";
    private static final String DB_USER = "admin";
    private static final String DB_PASSWORD = "secretPassword123";
    private static final String API_KEY = "sk-1234567890abcdef";

    public void connectToDatabase() {
        // Connect using credentials
        Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
    }
}
"""


@pytest.fixture
def parser_config():
    """Default parser configuration for tests"""
    return ParserConfig(
        min_tokens=10,
        max_tokens=100,
        overlap_tokens=5,
        include_comments=False
    )


@pytest.fixture
def security_config():
    """Default security configuration for tests"""
    return SecurityConfig(
        enabled=True,
        preserve_syntax=True,
        sensitivity_threshold=0.5
    )


@pytest.fixture
def embedding_config():
    """Default embedding configuration for tests"""
    return EmbeddingConfig(
        api_key="test_api_key",
        model_name="test-model",
        batch_size=5,
        timeout=10
    )


@pytest.fixture
def vector_config(temp_dir):
    """Default vector database configuration for tests"""
    return VectorDBConfig(
        collection_name="test_collection",
        persistent=True,
        persist_directory=str(temp_dir / "test_chroma"),
        max_batch_size=10
    )


@pytest.fixture
def update_config():
    """Default update configuration for tests"""
    return UpdateConfig(
        check_interval_seconds=60,
        max_concurrent_updates=1,
        enable_file_watching=False
    )


@pytest.fixture
def monitoring_config():
    """Default monitoring configuration for tests"""
    return MonitoringConfig(
        enable_metrics=True,
        enable_alerting=False,  # Disable for tests
        log_level=LogLevel.DEBUG
    )


@pytest.fixture
def mock_git_repo(temp_dir):
    """Create a mock git repository for testing"""
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir()

    # Create some sample files
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "main").mkdir()
    (repo_dir / "src" / "main" / "java").mkdir()

    java_file = repo_dir / "src" / "main" / "java" / "TestClass.java"
    java_file.write_text("""
public class TestClass {
    public void testMethod() {
        System.out.println("Hello World");
    }
}
""")

    # Create .git directory to make it look like a git repo
    (repo_dir / ".git").mkdir()

    return repo_dir


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing"""
    service = Mock()
    service.start = AsyncMock(return_value=None)
    service.stop = AsyncMock(return_value=None)
    service.generate_chunk_embeddings = AsyncMock(side_effect=lambda chunks: chunks)
    service.get_metrics = Mock(return_value={"total_requests": 0})
    service.health_check = AsyncMock(return_value={"status": "healthy"})
    service._is_running = True
    return service


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    store = Mock()
    store.connect = Mock(return_value=True)
    store.disconnect = Mock()
    store.store_chunks = Mock()
    store.search_similar_chunks = Mock(return_value=[])
    store.health_check = Mock(return_value={"vector_store_status": "healthy"})
    store._is_connected = True
    return store


@pytest.fixture
def mock_security_scanner():
    """Mock security scanner for testing"""
    scanner = Mock()
    scanner.scan_chunks = Mock(side_effect=lambda chunks: chunks)
    scanner.generate_security_report = Mock(return_value={
        "scan_summary": {
            "total_secrets_found": 0,
            "files_with_secrets": 0
        }
    })
    return scanner


# Test data generators
@pytest.fixture
def create_test_chunks():
    """Factory to create test chunks"""
    def _create_chunks(count=5):
        chunks = []
        for i in range(count):
            chunk = CodeChunk(
                content=f"public void method{i}() {{ return; }}",
                file_path=f"src/test/TestClass{i}.java",
                language=CodeLanguage.JAVA,
                start_line=i * 10,
                end_line=(i * 10) + 3,
                function_name=f"method{i}",
                class_name=f"TestClass{i}",
                layer_type=LayerType.SERVICE,
                token_count=10 + i,
                metadata={"test": True, "index": i}
            )
            chunks.append(chunk)
        return chunks
    return _create_chunks


# Async test helpers
@pytest.fixture
def async_test():
    """Helper for running async tests"""
    def _async_test(coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    return _async_test


# Cleanup helpers
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup logic if needed
    pass


# Mock HTTP client for API testing
@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing external API calls"""
    import aiohttp
    from unittest.mock import AsyncMock

    session = Mock(spec=aiohttp.ClientSession)
    session.post = AsyncMock()
    session.get = AsyncMock()
    session.close = AsyncMock()

    # Mock response
    response = Mock()
    response.status = 200
    response.json = AsyncMock(return_value={"data": []})
    response.text = AsyncMock(return_value="OK")

    session.post.return_value.__aenter__.return_value = response
    session.get.return_value.__aenter__.return_value = response

    return session


@pytest.fixture
def environment_variables(monkeypatch):
    """Set up test environment variables"""
    test_vars = {
        "JINA_API_KEY": "test_api_key",
        "CHROMADB_COLLECTION_NAME": "test_collection",
        "LOG_LEVEL": "DEBUG",
        "CHUNK_MIN_TOKENS": "10",
        "CHUNK_MAX_TOKENS": "100"
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests requiring running API server"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip E2E tests if API server is not running"""
    import httpx

    # Check if API server is running
    api_running = False
    try:
        # Increased timeout to 60 seconds for slow model loading
        response = httpx.get("http://localhost:8000/health", timeout=60.0)
        api_running = response.status_code == 200
    except:
        pass

    # Skip E2E tests if API server is not running
    skip_e2e = pytest.mark.skip(reason="API server not running (http://localhost:8000)")

    for item in items:
        if "e2e" in item.keywords and not api_running:
            item.add_marker(skip_e2e)