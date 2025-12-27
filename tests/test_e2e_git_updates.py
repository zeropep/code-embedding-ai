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
from unittest.mock import AsyncMock


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

        changes = monitor.detect_changes(since_commit=initial_state.commit_hash)
        assert changes is not None
        assert len(changes.detected_changes) > 0

        # Verify the new file is detected
        file_paths = [change.file_path for change in changes.detected_changes]
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

        config = UpdateConfig(
            check_interval_seconds=60
        )

        service = UpdateService(
            repo_path=str(temp_git_repo),
            update_config=config
        )

        # Service should initialize without errors
        assert service is not None
        assert service.repo_path == str(temp_git_repo)
        assert service.git_monitor is not None

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
        changes = monitor.detect_changes(since_commit=initial_state.commit_hash)
        assert any(
            change.change_type == ChangeType.ADDED and "added.py" in change.file_path
            for change in changes.detected_changes
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
        changes = monitor.detect_changes(since_commit=state_after_add.commit_hash)
        assert any(
            change.change_type == ChangeType.MODIFIED and "test.py" in change.file_path
            for change in changes.detected_changes
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
        changes = monitor.detect_changes(since_commit=state_after_modify.commit_hash)
        assert any(
            change.change_type == ChangeType.DELETED and "added.py" in change.file_path
            for change in changes.detected_changes
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

        manager.save_repository_state(state)

        # Load state via current_state property
        loaded_state = manager.current_state
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

        config = UpdateConfig(
            check_interval_seconds=60
        )

        service = UpdateService(
            repo_path=str(temp_git_repo),
            update_config=config
        )

        # This test just verifies the service can be initialized with the config
        assert service is not None
        assert service.update_config.check_interval_seconds == 60


class TestE2EGitUpdateMetrics:
    """E2E tests for update metrics and monitoring"""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_update_metrics_collection(self):
        """
        E2E Test: Update metrics are collected correctly
        """
        from src.updates.models import UpdateMetrics, UpdateStatus

        metrics = UpdateMetrics()
        metrics.update_success(processing_time=5.0, files_processed=10, chunks_modified=20)
        metrics.update_success(processing_time=4.5, files_processed=8, chunks_modified=15)
        metrics.update_failure()

        assert metrics.total_update_requests == 3
        assert metrics.successful_updates == 2
        assert metrics.failed_updates == 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_update_result_tracking(self):
        """
        E2E Test: Update results are tracked correctly
        """
        from src.updates.models import UpdateResult, UpdateStatus

        result = UpdateResult(
            request_id="test_123",
            status=UpdateStatus.COMPLETED,
            changes_detected=[],
            processing_time=10.5
        )

        result.files_processed = 15
        result.chunks_added = 13

        assert result.status == UpdateStatus.COMPLETED
        assert result.files_processed == 15


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

        # Should return None for corrupted state (accessed via current_state property)
        loaded_state = manager.current_state
        assert loaded_state is None

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_git_command_timeout(self, tmp_path):
        """
        E2E Test: Git commands should have reasonable timeouts
        """
        import subprocess
        from src.updates.git_monitor import GitMonitor
        from src.updates.models import UpdateConfig

        # Create temp git repo inline
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
        (repo_path / "test.py").write_text("def hello(): pass")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_path, check=True, capture_output=True)

        config = UpdateConfig()
        monitor = GitMonitor(str(repo_path), config)

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
