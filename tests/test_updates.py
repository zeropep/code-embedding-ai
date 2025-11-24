"""
Tests for update monitoring and Git integration functionality
Based on actual implementation in src/updates/
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import time

from src.updates.models import (
    UpdateConfig, FileChange, ChangeType, UpdateStatus,
    RepositoryState, UpdateRequest, UpdateResult, GitInfo,
    ChangeDetectionResult, UpdateMetrics
)


class TestChangeType:
    """Test ChangeType enum"""

    def test_change_type_values(self):
        """Test change type enum values"""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.RENAMED.value == "renamed"


class TestUpdateStatus:
    """Test UpdateStatus enum"""

    def test_update_status_values(self):
        """Test update status enum values"""
        assert UpdateStatus.PENDING.value == "pending"
        assert UpdateStatus.PROCESSING.value == "processing"
        assert UpdateStatus.COMPLETED.value == "completed"
        assert UpdateStatus.FAILED.value == "failed"
        assert UpdateStatus.SKIPPED.value == "skipped"


class TestFileChange:
    """Test FileChange dataclass"""

    def test_file_change_creation(self):
        """Test FileChange model creation"""
        change = FileChange(
            file_path="src/main/java/User.java",
            change_type=ChangeType.MODIFIED,
            file_hash="abc123"
        )

        assert change.file_path == "src/main/java/User.java"
        assert change.change_type == ChangeType.MODIFIED
        assert change.file_hash == "abc123"
        assert change.last_modified > 0

    def test_file_change_defaults(self):
        """Test FileChange default values"""
        change = FileChange(
            file_path="test.java",
            change_type=ChangeType.ADDED
        )

        assert change.old_path is None
        assert change.size_bytes == 0
        assert change.last_modified is not None

    def test_file_change_renamed(self):
        """Test FileChange with renamed file"""
        change = FileChange(
            file_path="src/NewName.java",
            change_type=ChangeType.RENAMED,
            old_path="src/OldName.java"
        )

        assert change.change_type == ChangeType.RENAMED
        assert change.old_path == "src/OldName.java"


class TestRepositoryState:
    """Test RepositoryState dataclass"""

    def test_repository_state_creation(self):
        """Test creating repository state"""
        state = RepositoryState(
            commit_hash="abc123def456",
            branch="main",
            timestamp=time.time(),
            file_hashes={"file1.java": "hash1", "file2.java": "hash2"}
        )

        assert state.commit_hash == "abc123def456"
        assert state.branch == "main"
        assert state.total_files == 2

    def test_repository_state_auto_count(self):
        """Test that total_files is auto-calculated"""
        state = RepositoryState(
            commit_hash="test123",
            branch="develop",
            timestamp=time.time(),
            file_hashes={"a.java": "h1", "b.java": "h2", "c.java": "h3"}
        )

        assert state.total_files == 3


class TestUpdateRequest:
    """Test UpdateRequest dataclass"""

    def test_update_request_creation(self):
        """Test creating update request"""
        request = UpdateRequest(
            request_id="req-001",
            repo_path="/path/to/repo",
            force_full_update=True
        )

        assert request.request_id == "req-001"
        assert request.repo_path == "/path/to/repo"
        assert request.force_full_update is True
        assert request.created_at > 0

    def test_update_request_defaults(self):
        """Test update request default values"""
        request = UpdateRequest(
            request_id="req-002",
            repo_path="/test/repo"
        )

        assert request.target_branch == "main"
        assert request.force_full_update is False
        assert request.include_patterns is not None
        assert request.exclude_patterns is not None
        assert ".java" in str(request.include_patterns)


class TestUpdateResult:
    """Test UpdateResult dataclass"""

    def test_update_result_creation(self):
        """Test creating update result"""
        changes = [
            FileChange(file_path="a.java", change_type=ChangeType.MODIFIED),
            FileChange(file_path="b.java", change_type=ChangeType.ADDED)
        ]

        result = UpdateResult(
            request_id="req-001",
            status=UpdateStatus.COMPLETED,
            changes_detected=changes,
            files_processed=2,
            chunks_added=10,
            chunks_updated=5
        )

        assert result.request_id == "req-001"
        assert result.status == UpdateStatus.COMPLETED
        assert result.total_changes == 2
        assert result.files_processed == 2

    def test_update_result_success_rate(self):
        """Test success rate calculation"""
        changes = [FileChange(file_path=f"file{i}.java", change_type=ChangeType.MODIFIED) for i in range(10)]

        result = UpdateResult(
            request_id="req-002",
            status=UpdateStatus.COMPLETED,
            changes_detected=changes,
            files_processed=8
        )

        assert result.success_rate == 0.8

    def test_update_result_to_dict(self):
        """Test result to_dict method"""
        result = UpdateResult(
            request_id="req-003",
            status=UpdateStatus.COMPLETED,
            changes_detected=[],
            processing_time=5.5
        )

        data = result.to_dict()
        assert data["request_id"] == "req-003"
        assert data["status"] == "completed"
        assert data["processing_time"] == 5.5


class TestUpdateConfig:
    """Test UpdateConfig dataclass"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = UpdateConfig()

        assert config.check_interval_seconds == 300
        assert config.max_concurrent_updates == 3
        assert config.enable_file_watching is False
        assert config.batch_size == 50

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = UpdateConfig(
            check_interval_seconds=60,
            max_concurrent_updates=5,
            enable_backup=False,
            max_file_size_mb=20
        )

        assert config.check_interval_seconds == 60
        assert config.max_concurrent_updates == 5
        assert config.enable_backup is False
        assert config.max_file_size_mb == 20


class TestGitInfo:
    """Test GitInfo dataclass"""

    def test_git_info_creation(self):
        """Test creating git info"""
        info = GitInfo(
            repo_path="/path/to/repo",
            current_branch="main",
            current_commit="abc123",
            is_dirty=True
        )

        assert info.repo_path == "/path/to/repo"
        assert info.current_branch == "main"
        assert info.current_commit == "abc123"
        assert info.is_dirty is True

    def test_git_info_defaults(self):
        """Test git info default values"""
        info = GitInfo(
            repo_path="/test",
            current_branch="develop",
            current_commit="xyz789"
        )

        assert info.remote_url is None
        assert info.is_dirty is False
        assert info.untracked_files == []
        assert info.modified_files == []


class TestChangeDetectionResult:
    """Test ChangeDetectionResult dataclass"""

    def test_detection_result_creation(self):
        """Test creating change detection result"""
        repo_state = RepositoryState(
            commit_hash="abc123",
            branch="main",
            timestamp=time.time(),
            file_hashes={}
        )
        git_info = GitInfo(
            repo_path="/test",
            current_branch="main",
            current_commit="abc123"
        )

        result = ChangeDetectionResult(
            repo_state=repo_state,
            detected_changes=[
                FileChange(file_path="a.java", change_type=ChangeType.ADDED),
                FileChange(file_path="b.java", change_type=ChangeType.MODIFIED)
            ],
            git_info=git_info,
            detection_time=0.5
        )

        assert result.has_changes is True
        assert len(result.detected_changes) == 2

    def test_changes_by_type(self):
        """Test changes_by_type property"""
        repo_state = RepositoryState(
            commit_hash="test",
            branch="main",
            timestamp=time.time(),
            file_hashes={}
        )
        git_info = GitInfo(
            repo_path="/test",
            current_branch="main",
            current_commit="test"
        )

        result = ChangeDetectionResult(
            repo_state=repo_state,
            detected_changes=[
                FileChange(file_path="a.java", change_type=ChangeType.ADDED),
                FileChange(file_path="b.java", change_type=ChangeType.ADDED),
                FileChange(file_path="c.java", change_type=ChangeType.MODIFIED)
            ],
            git_info=git_info,
            detection_time=0.3
        )

        by_type = result.changes_by_type
        assert by_type["added"] == 2
        assert by_type["modified"] == 1


class TestUpdateMetrics:
    """Test UpdateMetrics dataclass"""

    def test_metrics_defaults(self):
        """Test default metrics values"""
        metrics = UpdateMetrics()

        assert metrics.total_update_requests == 0
        assert metrics.successful_updates == 0
        assert metrics.failed_updates == 0
        assert metrics.success_rate == 0.0

    def test_update_success(self):
        """Test updating metrics for success"""
        metrics = UpdateMetrics()
        metrics.update_success(
            processing_time=2.5,
            files_processed=10,
            chunks_modified=50
        )

        assert metrics.total_update_requests == 1
        assert metrics.successful_updates == 1
        assert metrics.total_files_processed == 10
        assert metrics.total_chunks_modified == 50
        assert metrics.avg_processing_time == 2.5

    def test_update_failure(self):
        """Test updating metrics for failure"""
        metrics = UpdateMetrics()
        metrics.update_failure()

        assert metrics.total_update_requests == 1
        assert metrics.failed_updates == 1
        assert metrics.successful_updates == 0

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = UpdateMetrics()
        metrics.update_success(1.0, 5, 10)
        metrics.update_success(1.0, 5, 10)
        metrics.update_failure()

        assert metrics.total_update_requests == 3
        assert metrics.success_rate == pytest.approx(0.666, rel=0.01)

    def test_metrics_to_dict(self):
        """Test metrics to_dict method"""
        metrics = UpdateMetrics()
        metrics.update_success(2.0, 10, 20)

        data = metrics.to_dict()
        assert data["total_update_requests"] == 1
        assert data["successful_updates"] == 1
        assert "success_rate" in data


class TestGitMonitorIntegration:
    """Test GitMonitor with mocked Git operations"""

    @patch('src.updates.git_monitor.git')
    def test_git_monitor_initialization(self, mock_git):
        """Test GitMonitor initialization"""
        from src.updates.git_monitor import GitMonitor

        mock_repo = MagicMock()
        mock_git.Repo.return_value = mock_repo

        config = UpdateConfig()
        monitor = GitMonitor("/test/repo", config)

        assert monitor.repo_path == Path("/test/repo")
        assert monitor.config == config

    @patch('src.updates.git_monitor.git')
    def test_git_monitor_connect(self, mock_git, tmp_path):
        """Test GitMonitor connect method"""
        from src.updates.git_monitor import GitMonitor

        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.bare = False
        mock_git.Repo.return_value = mock_repo

        config = UpdateConfig()
        # Use tmp_path which actually exists
        monitor = GitMonitor(str(tmp_path), config)
        result = monitor.connect()

        assert result is True
        assert monitor.repo is not None


@pytest.mark.skip(reason="FileWatcher is not implemented")
class TestFileWatcher:
    """Test file system watching functionality - SKIPPED"""
    pass
