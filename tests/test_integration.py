"""
Integration tests for the complete code embedding pipeline
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.code_parser.models import ParserConfig
from src.security.models import SecurityConfig
from src.embeddings.models import EmbeddingConfig
from src.database.models import VectorDBConfig
from src.updates.models import UpdateConfig
from src.monitoring.models import MonitoringConfig


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality"""

    @pytest.fixture
    def integration_configs(self, temp_dir):
        """Create configurations for integration testing"""
        parser_config = ParserConfig(
            min_tokens=10,
            max_tokens=200,
            overlap_tokens=10
        )

        security_config = SecurityConfig(
            enabled=True,
            preserve_syntax=True,
            sensitivity_threshold=0.5
        )

        embedding_config = EmbeddingConfig(
            api_key="test_api_key",
            model_name="test-model",
            batch_size=5,
            timeout=30
        )

        vector_config = VectorDBConfig(
            collection_name="integration_test",
            persistent=True,
            persist_directory=str(temp_dir / "test_chroma"),
            max_batch_size=10
        )

        update_config = UpdateConfig(
            check_interval_seconds=60,
            max_concurrent_updates=1,
            enable_file_watching=False
        )

        monitoring_config = MonitoringConfig(
            enable_metrics=True,
            enable_alerting=False,
            log_level="DEBUG"
        )

        return {
            "parser": parser_config,
            "security": security_config,
            "embedding": embedding_config,
            "vector": vector_config,
            "update": update_config,
            "monitoring": monitoring_config
        }

    @pytest.fixture
    def sample_spring_boot_project(self, temp_dir):
        """Create a sample Spring Boot project structure for testing"""
        project_dir = temp_dir / "test_spring_project"
        project_dir.mkdir()

        # Create source directories
        src_main = project_dir / "src" / "main" / "java" / "com" / "example"
        src_main.mkdir(parents=True)

        src_resources = project_dir / "src" / "main" / "resources" / "templates"
        src_resources.mkdir(parents=True)

        # Create Java files
        # Entity
        user_entity = src_main / "User.java"
        user_entity.write_text("""
package com.example;

import javax.persistence.*;

@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String email;

    private String name;

    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}
""")

        # Repository
        user_repository = src_main / "UserRepository.java"
        user_repository.write_text("""
package com.example;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);

    @Query("SELECT u FROM User u WHERE u.name LIKE %?1%")
    List<User> findByNameContaining(String name);
}
""")

        # Service
        user_service = src_main / "UserService.java"
        user_service.write_text("""
package com.example;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    public User createUser(User user) {
        validateUser(user);
        return userRepository.save(user);
    }

    public User getUserById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new UserNotFoundException("User not found: " + id));
    }

    public User getUserByEmail(String email) {
        return userRepository.findByEmail(email)
            .orElseThrow(() -> new UserNotFoundException("User not found with email: " + email));
    }

    public List<User> searchUsers(String nameFilter) {
        return userRepository.findByNameContaining(nameFilter);
    }

    public void deleteUser(Long id) {
        User user = getUserById(id);
        userRepository.delete(user);
    }

    private void validateUser(User user) {
        if (user.getEmail() == null || user.getEmail().trim().isEmpty()) {
            throw new ValidationException("Email is required");
        }
        if (user.getName() == null || user.getName().trim().isEmpty()) {
            throw new ValidationException("Name is required");
        }
    }
}
""")

        # Controller
        user_controller = src_main / "UserController.java"
        user_controller.write_text("""
package com.example;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User createdUser = userService.createUser(user);
        return ResponseEntity.ok(createdUser);
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @GetMapping
    public ResponseEntity<List<User>> getUsers(@RequestParam(required = false) String search) {
        List<User> users = search != null ?
            userService.searchUsers(search) :
            userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody User user) {
        user.setId(id);
        User updatedUser = userService.updateUser(user);
        return ResponseEntity.ok(updatedUser);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}
""")

        # Configuration with secrets (for security testing)
        config_file = src_main / "DatabaseConfig.java"
        config_file.write_text("""
package com.example;

import org.springframework.context.annotation.Configuration;

@Configuration
public class DatabaseConfig {

    // These should be detected as secrets
    private static final String DB_PASSWORD = "secretPassword123";
    private static final String API_KEY = "sk-1234567890abcdef";
    private static final String JWT_SECRET = "mySecretJwtKey2024";

    public DataSource dataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:postgresql://localhost:5432/testdb");
        config.setUsername("admin");
        config.setPassword(DB_PASSWORD);
        return new HikariDataSource(config);
    }
}
""")

        # Thymeleaf template
        user_template = src_resources / "user-form.html"
        user_template.write_text("""
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>User Management</title>
    <link rel="stylesheet" type="text/css" th:href="@{/css/bootstrap.min.css}"/>
</head>
<body>
    <div class="container">
        <h1>User Management</h1>

        <div th:fragment="userForm">
            <form th:object="${user}" th:action="@{/users}" method="post">
                <div class="form-group">
                    <label for="name">Name:</label>
                    <input type="text"
                           th:field="*{name}"
                           id="name"
                           class="form-control"
                           required>
                    <span th:if="${#fields.hasErrors('name')}"
                          th:errors="*{name}"
                          class="text-danger"></span>
                </div>

                <div class="form-group">
                    <label for="email">Email:</label>
                    <input type="email"
                           th:field="*{email}"
                           id="email"
                           class="form-control"
                           required>
                    <span th:if="${#fields.hasErrors('email')}"
                          th:errors="*{email}"
                          class="text-danger"></span>
                </div>

                <button type="submit" class="btn btn-primary">Save User</button>
                <a th:href="@{/users}" class="btn btn-secondary">Cancel</a>
            </form>
        </div>

        <div th:fragment="userList">
            <table class="table table-striped" th:if="${users}">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Email</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr th:each="user : ${users}">
                        <td th:text="${user.id}">1</td>
                        <td th:text="${user.name}">John Doe</td>
                        <td th:text="${user.email}">john@example.com</td>
                        <td>
                            <a th:href="@{/users/{id}/edit(id=${user.id})}"
                               class="btn btn-sm btn-warning">Edit</a>
                            <a th:href="@{/users/{id}/delete(id=${user.id})}"
                               class="btn btn-sm btn-danger"
                               onclick="return confirm('Are you sure?')">Delete</a>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
""")

        return project_dir

    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, integration_configs, sample_spring_boot_project):
        """Test the complete pipeline from parsing to storage"""
        # Create pipeline
        pipeline = EmbeddingPipeline(
            parser_config=integration_configs["parser"],
            security_config=integration_configs["security"],
            embedding_config=integration_configs["embedding"]
        )

        # Mock external services
        with patch('src.embeddings.jina_client.JinaEmbeddingClient') as mock_client_class:
            # Mock embedding client
            mock_client = AsyncMock()
            mock_client.generate_embeddings_batch.return_value = [
                Mock(
                    request_id=f"chunk_{i}",
                    vector=[0.1 * i, 0.2 * i, 0.3 * i],
                    status="completed",
                    processing_time=0.5
                )
                for i in range(10)  # Assume 10 chunks
            ]
            mock_client_class.return_value = mock_client

            # Mock vector store
            with patch('src.database.chroma_store.ChromaVectorStore') as mock_store_class:
                mock_store = Mock()
                mock_store.connect.return_value = True
                mock_store.store_chunks.return_value = True
                mock_store_class.return_value = mock_store

                # Execute pipeline
                result = await pipeline.process_repository(str(sample_spring_boot_project))

                # Verify results
                assert result["status"] == "success"
                assert "processing_summary" in result
                assert result["processing_summary"]["files_processed"] >= 4  # At least 4 Java files + HTML
                assert result["processing_summary"]["chunks_created"] > 0
                assert result["processing_summary"]["secrets_detected"] > 0  # Config file has secrets

                # Verify pipeline steps were called
                mock_client.generate_embeddings_batch.assert_called()
                mock_store.store_chunks.assert_called()

    @pytest.mark.asyncio
    async def test_security_scanning_integration(self, integration_configs, sample_spring_boot_project):
        """Test security scanning integration in the pipeline"""
        from src.security.security_scanner import SecurityScanner

        scanner = SecurityScanner(integration_configs["security"])

        # Parse and scan the config file with secrets
        from src.code_parser.java_parser import JavaParser
        parser = JavaParser(integration_configs["parser"])

        config_file = sample_spring_boot_project / "src" / "main" / "java" / "com" / "example" / "DatabaseConfig.java"
        parsed_file = parser.parse_file(config_file)

        # Scan chunks for secrets
        scanned_chunks = scanner.scan_chunks(parsed_file.chunks)

        # Verify secrets were detected and masked
        security_chunks = [c for c in scanned_chunks if "security" in c.metadata]
        assert len(security_chunks) > 0

        # Check that secrets were masked
        for chunk in security_chunks:
            security_meta = chunk.metadata["security"]
            if security_meta["secrets_masked"] > 0:
                # Content should not contain original secrets
                assert "secretPassword123" not in chunk.content
                assert "sk-1234567890abcdef" not in chunk.content
                # Should contain masked placeholders
                assert "[MASKED_" in chunk.content

    @pytest.mark.asyncio
    async def test_chunking_and_embedding_flow(self, integration_configs, sample_spring_boot_project):
        """Test code chunking and embedding generation flow"""
        from src.code_parser.code_parser import CodeParser
        from src.embeddings.embedding_service import EmbeddingService

        # Parse repository
        parser = CodeParser(integration_configs["parser"])
        parsed_files = await parser.parse_repository_async(str(sample_spring_boot_project))

        assert len(parsed_files) >= 5  # At least 5 files (4 Java + 1 HTML)

        # Get chunks for embedding
        chunks = parser.get_chunks_for_embedding(parsed_files)
        assert len(chunks) > 0

        # Check chunk properties
        for chunk in chunks:
            assert chunk.token_count >= integration_configs["parser"].min_tokens
            assert chunk.token_count <= integration_configs["parser"].max_tokens
            assert chunk.content.strip()  # Not empty

        # Mock embedding service
        with patch('src.embeddings.jina_client.JinaEmbeddingClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_embeddings_batch.return_value = [
                Mock(
                    request_id=chunk.chunk_id,
                    vector=[0.1, 0.2, 0.3] * 128,  # 384-dimensional vector
                    status="completed",
                    processing_time=0.5
                )
                for chunk in chunks
            ]
            mock_client_class.return_value = mock_client

            # Generate embeddings
            service = EmbeddingService(integration_configs["embedding"])
            await service.start()

            embedded_chunks = await service.generate_chunk_embeddings(chunks)

            assert len(embedded_chunks) == len(chunks)
            for chunk in embedded_chunks:
                assert "embedding" in chunk.metadata
                embedding_meta = chunk.metadata["embedding"]
                assert "vector" in embedding_meta
                assert len(embedding_meta["vector"]) > 0

            await service.stop()

    @pytest.mark.asyncio
    async def test_incremental_updates_flow(self, integration_configs, sample_spring_boot_project):
        """Test incremental update flow with Git monitoring"""
        from src.updates.update_manager import UpdateManager
        from src.updates.models import FileChange, ChangeType

        # Mock services
        mock_embedding_service = AsyncMock()
        mock_vector_store = Mock()
        mock_vector_store.store_chunks.return_value = True
        mock_vector_store.delete_chunks.return_value = True

        # Create update manager
        update_manager = UpdateManager(
            repo_path=str(sample_spring_boot_project),
            config=integration_configs["update"],
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )

        # Simulate file changes
        changes = [
            FileChange(
                file_path="src/main/java/com/example/UserService.java",
                change_type=ChangeType.MODIFIED,
                old_content="old service code",
                new_content="updated service code with new method"
            ),
            FileChange(
                file_path="src/main/java/com/example/NewFeature.java",
                change_type=ChangeType.ADDED,
                new_content="new feature implementation"
            ),
            FileChange(
                file_path="src/main/java/com/example/OldClass.java",
                change_type=ChangeType.DELETED,
                old_content="deprecated class"
            )
        ]

        # Process changes
        with patch.object(update_manager, 'git_monitor') as mock_git:
            mock_git.check_for_updates.return_value = changes

            result = await update_manager.check_and_process_updates()

            assert result is True
            # Verify embedding service was called for new/modified files
            mock_embedding_service.generate_chunk_embeddings.assert_called()
            # Verify chunks were deleted for removed files
            mock_vector_store.delete_chunks.assert_called()

    @pytest.mark.asyncio
    async def test_search_functionality_integration(self, integration_configs):
        """Test end-to-end search functionality"""
        from src.embeddings.jina_client import JinaEmbeddingClient
        from src.database.chroma_store import ChromaVectorStore

        # Mock embedding client for query
        with patch('src.embeddings.jina_client.JinaEmbeddingClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_embedding.return_value = Mock(
                vector=[0.1, 0.2, 0.3] * 128,
                status="completed"
            )
            mock_client_class.return_value = mock_client

            # Mock vector store for search
            with patch('src.database.chroma_store.ChromaVectorStore') as mock_store_class:
                mock_store = Mock()
                mock_store.search_similar_chunks.return_value = [
                    {
                        "chunk_id": "user_service_create",
                        "content": "public User createUser(User user) { ... }",
                        "similarity": 0.95,
                        "metadata": {
                            "file": "UserService.java",
                            "class_name": "UserService",
                            "function_name": "createUser",
                            "layer_type": "service"
                        }
                    },
                    {
                        "chunk_id": "user_controller_post",
                        "content": "@PostMapping public ResponseEntity<User> createUser(...) { ... }",
                        "similarity": 0.87,
                        "metadata": {
                            "file": "UserController.java",
                            "class_name": "UserController",
                            "function_name": "createUser",
                            "layer_type": "controller"
                        }
                    }
                ]
                mock_store_class.return_value = mock_store

                # Create search components
                embedding_client = JinaEmbeddingClient(integration_configs["embedding"])
                vector_store = ChromaVectorStore(integration_configs["vector"])

                # Perform search
                query = "user creation and validation logic"

                # Generate query embedding
                async with embedding_client:
                    query_embedding = await embedding_client.generate_embedding(query, "search_query")

                # Search for similar chunks
                results = vector_store.search_similar_chunks(
                    query_embedding.vector,
                    limit=5,
                    similarity_threshold=0.7
                )

                # Verify search results
                assert len(results) == 2
                assert results[0]["similarity"] > results[1]["similarity"]  # Sorted by similarity
                assert "UserService" in results[0]["metadata"]["class_name"]
                assert "createUser" in results[0]["metadata"]["function_name"]

    def test_configuration_validation_integration(self, integration_configs):
        """Test configuration validation across all components"""
        # All configs should be valid
        for config_name, config in integration_configs.items():
            assert config.validate() is True, f"{config_name} config validation failed"

        # Test invalid configurations
        invalid_parser = ParserConfig(min_tokens=0)  # Invalid
        assert invalid_parser.validate() is False

        invalid_security = SecurityConfig(sensitivity_threshold=1.5)  # Invalid (> 1.0)
        assert invalid_security.validate() is False

        invalid_embedding = EmbeddingConfig(api_key="", batch_size=0)  # Invalid
        assert invalid_embedding.validate() is False

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integration_configs, sample_spring_boot_project):
        """Test error handling across the pipeline"""
        pipeline = EmbeddingPipeline(
            parser_config=integration_configs["parser"],
            security_config=integration_configs["security"],
            embedding_config=integration_configs["embedding"]
        )

        # Test with embedding service failure
        with patch('src.embeddings.jina_client.JinaEmbeddingClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_embeddings_batch.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client

            # Pipeline should handle error gracefully
            result = await pipeline.process_repository(str(sample_spring_boot_project))

            assert result["status"] == "error" or "errors" in result
            assert "error_message" in result or "processing_summary" in result

        # Test with invalid repository path
        result = await pipeline.process_repository("/nonexistent/path")
        assert result["status"] == "error"
        assert "error_message" in result

    def test_metrics_and_monitoring_integration(self, integration_configs):
        """Test metrics collection and monitoring integration"""
        from src.monitoring.metrics import MetricsCollector
        from src.monitoring.health_checker import HealthChecker

        # Create monitoring components
        metrics = MetricsCollector(integration_configs["monitoring"])
        health_checker = HealthChecker(integration_configs["monitoring"])

        # Test metrics collection
        metrics.record_embedding_success(1.5, 10, "test-model")
        metrics.record_database_operation("insert", 0.1, 50, True)
        metrics.increment_counter("api_requests", labels={"endpoint": "/search"})

        # Export metrics
        exported = metrics.export_metrics()
        assert "counters" in exported
        assert "histograms" in exported

        # Test health checking
        def mock_health_check():
            return {"status": "healthy", "response_time": 0.05}

        health_checker.register_component("test_service", mock_health_check)
        overall_health = health_checker.check_overall_health()

        assert overall_health["overall_status"] == "healthy"
        assert "test_service" in overall_health["components"]

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integration_configs, sample_spring_boot_project):
        """Test concurrent pipeline operations"""
        # Create multiple pipeline instances
        pipelines = [
            EmbeddingPipeline(
                parser_config=integration_configs["parser"],
                security_config=integration_configs["security"],
                embedding_config=integration_configs["embedding"]
            )
            for _ in range(3)
        ]

        # Mock external services
        with patch('src.embeddings.jina_client.JinaEmbeddingClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_embeddings_batch.return_value = [
                Mock(request_id=f"chunk_{i}", vector=[0.1], status="completed")
                for i in range(5)
            ]
            mock_client_class.return_value = mock_client

            with patch('src.database.chroma_store.ChromaVectorStore') as mock_store_class:
                mock_store = Mock()
                mock_store.connect.return_value = True
                mock_store.store_chunks.return_value = True
                mock_store_class.return_value = mock_store

                # Run pipelines concurrently
                tasks = [
                    pipeline.process_repository(str(sample_spring_boot_project))
                    for pipeline in pipelines
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # At least one should succeed
                successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
                assert len(successful_results) >= 1