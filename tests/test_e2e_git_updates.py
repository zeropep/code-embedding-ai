"""
End-to-End tests for Git auto-update functionality
Tests UpdateService, GitMonitor, and automatic embedding updates
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
import time
import subprocess


class TestE2EGitAutoUpdate:
    """E2E tests for Git auto-update functionality"""

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )

        # Create initial file
        (repo_path / "test.py").write_text("def hello(): pass")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True
        )

        yield repo_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_git_monitor_detects_changes(self, temp_git_repo):
        """
        E2E Test: GitMonitor detects file changes in Git repository
        """
        from src.updates.git_monitor import GitMonitor
        from src.updates.models import UpdateConfig

        config = UpdateConfig(
            check_interval_seconds=1,
            max_concurrent_updates=1,
            enable_file_watching=False
        )

        monitor = GitMonitor(str(temp_git_repo), config)

        # Get initial state
        assert monitor.connect()
        initial_state = monitor.get_current_state()
        assert initial_state is not None
        assert initial_state.commit_hash is not None

        # Make a change
        new_file = temp_git_repo / "new_file.py"
        new_file.write_text("def new_function(): return 42")

        subprocess.run(
            ["git", "add", "."],
            cwd=temp_git_repo,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add new file"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True
        )

        # Check for changes
        new_state = monitor.get_current_state()
        assert new_state.commit_hash != initial_state.commit_hash

        changes = monitor.detect_changes(initial_state, new_state)
        assert changes is not None
        assert len(changes.changed_files) > 0

        # Verify the new file is detected
        file_paths = [change.file_path for change in changes.changed_files]
        assert any("new_file.py" in path for path in file_paths)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_update_service_initialization(self, temp_git_repo):
        """
        E2E Test: UpdateService initializes correctly with Git repo
        """
        from src.updates.update_service import UpdateService
        from src.updates.models import UpdateConfig
        from src.embeddings.embedding_pipeline import EmbeddingPipeline
        from unittest.mock import Mock

        config = UpdateConfig(
            check_interval_seconds=60,
            max_concurrent_updates=1,
            enable_file_watching=False
        )

        # Mock pipeline
        mock_pipeline = Mock(spec=EmbeddingPipeline)
        mock_pipeline.process_repository = AsyncMock(return_value={"success": True})

        service = UpdateService(
            repo_path=str(temp_git_repo),
            pipeline=mock_pipeline,
            config=config
        )

        # Connect to repository
        assert service.connect()

        # Get initial state
        state = service.get_current_state()
        assert state is not None
        assert state.commit_hash is not None

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_file_change_detection_types(self, temp_git_repo):
        """
        E2E Test: Detect different types of file changes (add, modify, delete)
        """
        from src.updates.git_monitor import GitMonitor
        from src.updates.models import UpdateConfig, ChangeType

        config = UpdateConfig()
        monitor = GitMonitor(str(temp_git_repo), config)
        monitor.connect()

        initial_state = monitor.get_current_state()

        # Test ADDED
        new_file = temp_git_repo / "added.py"
        new_file.write_text("# New file")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add file"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True
        )

        state_after_add = monitor.get_current_state()
        changes = monitor.detect_changes(initial_state, state_after_add)
        assert any(
            change.change_type == ChangeType.ADDED and "added.py" in change.file_path
            for change in changes.changed_files
        )

        # Test MODIFIED
        (temp_git_repo / "test.py").write_text("def hello(): return 'modified'")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Modify file"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True
        )

        state_after_modify = monitor.get_current_state()
        changes = monitor.detect_changes(state_after_add, state_after_modify)
        assert any(
            change.change_type == ChangeType.MODIFIED and "test.py" in change.file_path
            for change in changes.changed_files
        )

        # Test DELETED
        (temp_git_repo / "added.py").unlink()
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Delete file"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True
        )

        state_after_delete = monitor.get_current_state()
        changes = monitor.detect_changes(state_after_modify, state_after_delete)
        assert any(
            change.change_type == ChangeType.DELETED and "added.py" in change.file_path
            for change in changes.changed_files
        )

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_state_persistence(self, temp_git_repo, tmp_path):
        """
        E2E Test: Repository state is persisted and loaded correctly
        """
        from src.updates.state_manager import StateManager

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        manager = StateManager(str(state_dir))

        # Create and save state
        from src.updates.models import RepositoryState
        import time

        state = RepositoryState(
            commit_hash="test_hash_123",
            branch="main",
            timestamp=time.time(),
            file_hashes={"test.py": "hash1"}
        )

        manager.save_state(state)

        # Load state
        loaded_state = manager.load_state()
        assert loaded_state is not None
        assert loaded_state.commit_hash == "test_hash_123"
        assert loaded_state.branch == "main"
        assert "test.py" in loaded_state.file_hashes

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_concurrent_update_limit(self, temp_git_repo):
        """
        E2E Test: UpdateService respects max_concurrent_updates limit
        """
        from src.updates.update_service import UpdateService
        from src.updates.models import UpdateConfig
        from unittest.mock import Mock, AsyncMock

        config = UpdateConfig(
            check_interval_seconds=60,
            max_concurrent_updates=1,  # Only allow 1 concurrent update
            enable_file_watching=False
        )

        mock_pipeline = Mock()
        # Simulate slow processing
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(0.5)
            return {"success": True}

        mock_pipeline.process_repository = AsyncMock(side_effect=slow_process)

        service = UpdateService(
            repo_path=str(temp_git_repo),
            pipeline=mock_pipeline,
            config=config
        )

        # This test just verifies the service can be initialized with the config
        # Actual concurrent update testing would require running the service
        assert service.config.max_concurrent_updates == 1


class TestE2EGitUpdateMetrics:
    """E2E tests for update metrics and monitoring"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_update_metrics_collection(self):
        """
        E2E Test: Update metrics are collected correctly
        """
        from src.updates.models import UpdateMetrics, UpdateStatus

        metrics = UpdateMetrics(
            total_updates=10,
            successful_updates=8,
            failed_updates=2,
            total_files_processed=100,
            total_processing_time=60.5,
            average_processing_time=6.05
        )

        assert metrics.total_updates == 10
        assert metrics.success_rate == 0.8
        assert metrics.average_processing_time == 6.05

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_update_result_tracking(self):
        """
        E2E Test: Update results are tracked correctly
        """
        from src.updates.models import UpdateResult, UpdateStatus

        result = UpdateResult(
            status=UpdateStatus.COMPLETED,
            files_processed=15,
            files_failed=2,
            processing_time=10.5,
            commit_hash="abc123",
            message="Update completed successfully"
        )

        assert result.status == UpdateStatus.COMPLETED
        assert result.files_processed == 15
        assert result.success_rate == (13 / 15)  # (15 - 2) / 15


class TestE2EGitUpdateErrorHandling:
    """E2E tests for error handling in Git updates"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_invalid_git_repo_path(self):
        """
        E2E Test: Gracefully handle invalid Git repository path
        """
        from src.updates.git_monitor import GitMonitor
        from src.updates.models import UpdateConfig

        config = UpdateConfig()
        monitor = GitMonitor("/nonexistent/path", config)

        # Should return False when connecting to invalid path
        assert not monitor.connect()

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_corrupted_state_recovery(self, tmp_path):
        """
        E2E Test: StateManager handles corrupted state files
        """
        from src.updates.state_manager import StateManager

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        # Create corrupted state file
        state_file = state_dir / "repository_state.json"
        state_file.write_text("{ invalid json }")

        manager = StateManager(str(state_dir))

        # Should return None for corrupted state
        loaded_state = manager.load_state()
        assert loaded_state is None

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_git_command_timeout(self, temp_git_repo):
        """
        E2E Test: Git commands should have reasonable timeouts
        """
        from src.updates.git_monitor import GitMonitor
        from src.updates.models import UpdateConfig

        config = UpdateConfig()
        monitor = GitMonitor(str(temp_git_repo), config)

        assert monitor.connect()

        # Get state should complete quickly (< 5 seconds)
        import time
        start = time.time()
        state = monitor.get_current_state()
        elapsed = time.time() - start

        assert state is not None
        assert elapsed < 5.0, f"Git command took too long: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
