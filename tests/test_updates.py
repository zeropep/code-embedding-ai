"""
Tests for update monitoring and Git integration functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import asyncio

from src.updates.models import UpdateConfig, FileChange, ChangeType
from src.updates.git_monitor import GitMonitor
from src.updates.update_manager import UpdateManager
from src.updates.file_watcher import FileWatcher


class TestUpdateModels:
    """Test update model classes"""

    def test_file_change_creation(self):
        """Test FileChange model creation"""
        change = FileChange(
            file_path="src/main/java/User.java",
            change_type=ChangeType.MODIFIED,
            old_content="old code",
            new_content="new code",
            timestamp=1234567890.0
        )

        assert change.file_path == "src/main/java/User.java"
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_content == "old code"
        assert change.new_content == "new code"

    def test_file_change_to_dict(self):
        """Test FileChange serialization"""
        change = FileChange(
            file_path="test.java",
            change_type=ChangeType.ADDED,
            new_content="test content"
        )

        change_dict = change.to_dict()

        assert "file_path" in change_dict
        assert "change_type" in change_dict
        assert change_dict["change_type"] == "added"

    def test_update_config_defaults(self):
        """Test UpdateConfig default values"""
        config = UpdateConfig()

        assert config.check_interval_seconds == 300  # 5 minutes
        assert config.max_concurrent_updates == 3
        assert config.enable_file_watching is True
        assert config.git_branch == "main"

    def test_update_config_validation(self):
        """Test UpdateConfig validation"""
        # Valid config
        valid_config = UpdateConfig(
            check_interval_seconds=60,
            max_concurrent_updates=1
        )
        assert valid_config.validate() is True

        # Invalid config - bad interval
        invalid_config = UpdateConfig(check_interval_seconds=0)
        assert invalid_config.validate() is False

        # Invalid config - bad concurrency
        invalid_config2 = UpdateConfig(max_concurrent_updates=0)
        assert invalid_config2.validate() is False


class TestGitMonitor:
    """Test Git repository monitoring"""

    @pytest.fixture
    def git_monitor(self, update_config, mock_git_repo):
        """Create git monitor for testing"""
        return GitMonitor(str(mock_git_repo), update_config)

    def test_git_monitor_initialization(self, update_config, mock_git_repo):
        """Test GitMonitor initialization"""
        monitor = GitMonitor(str(mock_git_repo), update_config)

        assert monitor.repo_path == str(mock_git_repo)
        assert monitor.config == update_config
        assert monitor.last_commit_hash is None

    @patch('subprocess.run')
    def test_get_current_commit_hash(self, mock_run, git_monitor):
        """Test getting current commit hash"""
        # Mock git rev-parse output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="abc123def456\n",
            stderr=""
        )

        commit_hash = git_monitor.get_current_commit_hash()

        assert commit_hash == "abc123def456"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_changed_files(self, mock_run, git_monitor):
        """Test getting changed files between commits"""
        # Mock git diff output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="M\tsrc/main/java/User.java\nA\tsrc/test/UserTest.java\nD\tREADME.md\n",
            stderr=""
        )

        changes = git_monitor.get_changed_files("old_hash", "new_hash")

        assert len(changes) == 3
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[1].change_type == ChangeType.ADDED
        assert changes[2].change_type == ChangeType.DELETED

    @patch('subprocess.run')
    def test_get_file_content_at_commit(self, mock_run, git_monitor):
        """Test getting file content at specific commit"""
        # Mock git show output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="public class User { }",
            stderr=""
        )

        content = git_monitor.get_file_content_at_commit("abc123", "src/User.java")

        assert content == "public class User { }"

    @patch('subprocess.run')
    def test_check_for_updates(self, mock_run, git_monitor):
        """Test checking for repository updates"""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse (current)
            Mock(returncode=0, stdout="new_hash\n", stderr=""),
            # git rev-parse (last known)
            Mock(returncode=0, stdout="old_hash\n", stderr=""),
            # git diff
            Mock(returncode=0, stdout="M\tsrc/User.java\n", stderr=""),
            # git show (old content)
            Mock(returncode=0, stdout="old content", stderr=""),
            # git show (new content)
            Mock(returncode=0, stdout="new content", stderr="")
        ]

        git_monitor.last_commit_hash = "old_hash"
        changes = git_monitor.check_for_updates()

        assert len(changes) >= 1
        if changes:
            assert changes[0].change_type == ChangeType.MODIFIED
            assert changes[0].file_path == "src/User.java"

    def test_is_supported_file(self, git_monitor):
        """Test file type support checking"""
        assert git_monitor.is_supported_file("src/User.java") is True
        assert git_monitor.is_supported_file("templates/user.html") is True
        assert git_monitor.is_supported_file("README.md") is False
        assert git_monitor.is_supported_file("config.xml") is False

    @patch('subprocess.run')
    def test_git_command_error_handling(self, mock_run, git_monitor):
        """Test error handling for Git commands"""
        # Mock git command failure
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository"
        )

        result = git_monitor.get_current_commit_hash()
        assert result is None

    def test_filter_supported_changes(self, git_monitor):
        """Test filtering changes to only supported file types"""
        changes = [
            FileChange("src/User.java", ChangeType.MODIFIED),
            FileChange("README.md", ChangeType.MODIFIED),
            FileChange("templates/user.html", ChangeType.ADDED),
            FileChange("pom.xml", ChangeType.DELETED)
        ]

        filtered = git_monitor.filter_supported_changes(changes)

        assert len(filtered) == 2
        file_paths = [change.file_path for change in filtered]
        assert "src/User.java" in file_paths
        assert "templates/user.html" in file_paths
        assert "README.md" not in file_paths
        assert "pom.xml" not in file_paths


class TestFileWatcher:
    """Test file system watching functionality"""

    @pytest.fixture
    def file_watcher(self, update_config, temp_dir):
        """Create file watcher for testing"""
        return FileWatcher(str(temp_dir), update_config)

    def test_file_watcher_initialization(self, update_config, temp_dir):
        """Test FileWatcher initialization"""
        watcher = FileWatcher(str(temp_dir), update_config)

        assert watcher.watch_path == str(temp_dir)
        assert watcher.config == update_config
        assert watcher._observer is None
        assert watcher._is_watching is False

    def test_start_stop_watching(self, file_watcher):
        """Test starting and stopping file watching"""
        # Mock watchdog observer
        with patch('watchdog.observers.Observer') as mock_observer_class:
            mock_observer = Mock()
            mock_observer_class.return_value = mock_observer

            # Start watching
            file_watcher.start_watching()
            assert file_watcher._is_watching is True
            mock_observer.start.assert_called_once()

            # Stop watching
            file_watcher.stop_watching()
            assert file_watcher._is_watching is False
            mock_observer.stop.assert_called_once()

    @patch('asyncio.Queue')
    def test_file_change_handler(self, mock_queue, file_watcher):
        """Test file change event handling"""
        from watchdog.events import FileModifiedEvent

        # Set up change queue
        file_watcher._change_queue = Mock()

        # Create file change event
        event = FileModifiedEvent("src/User.java")

        # Handle event
        handler = file_watcher._create_event_handler()
        handler.on_modified(event)

        # Verify change was queued
        file_watcher._change_queue.put_nowait.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pending_changes(self, file_watcher):
        """Test getting pending file changes"""
        # Mock change queue
        changes = [
            FileChange("file1.java", ChangeType.MODIFIED),
            FileChange("file2.java", ChangeType.ADDED)
        ]

        with patch('asyncio.Queue') as mock_queue_class:
            mock_queue = Mock()
            mock_queue_class.return_value = mock_queue

            # Mock queue.get to return changes then raise Empty
            mock_queue.get_nowait.side_effect = changes + [asyncio.QueueEmpty()]

            file_watcher._change_queue = mock_queue

            pending = await file_watcher.get_pending_changes()

            assert len(pending) == 2
            assert pending[0].file_path == "file1.java"
            assert pending[1].file_path == "file2.java"

    def test_debounce_changes(self, file_watcher):
        """Test change debouncing to avoid duplicate events"""
        import time

        changes = [
            FileChange("file1.java", ChangeType.MODIFIED, timestamp=time.time()),
            FileChange("file1.java", ChangeType.MODIFIED, timestamp=time.time()),  # Duplicate
            FileChange("file2.java", ChangeType.MODIFIED, timestamp=time.time())
        ]

        debounced = file_watcher.debounce_changes(changes)

        # Should remove duplicate
        assert len(debounced) == 2
        file_paths = [change.file_path for change in debounced]
        assert "file1.java" in file_paths
        assert "file2.java" in file_paths


class TestUpdateManager:
    """Test update management orchestration"""

    @pytest.fixture
    def update_manager(self, update_config, mock_git_repo, mock_embedding_service, mock_vector_store):
        """Create update manager for testing"""
        return UpdateManager(
            repo_path=str(mock_git_repo),
            config=update_config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )

    def test_update_manager_initialization(self, update_config, mock_git_repo, mock_embedding_service, mock_vector_store):
        """Test UpdateManager initialization"""
        manager = UpdateManager(
            repo_path=str(mock_git_repo),
            config=update_config,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )

        assert manager.repo_path == str(mock_git_repo)
        assert manager.config == update_config
        assert manager.embedding_service == mock_embedding_service
        assert manager.vector_store == mock_vector_store

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, update_manager):
        """Test starting and stopping update monitoring"""
        # Mock components
        with patch.object(update_manager, '_monitoring_task', new_callable=Mock) as mock_task:
            # Start monitoring
            await update_manager.start_monitoring()
            assert update_manager._is_monitoring is True

            # Stop monitoring
            await update_manager.stop_monitoring()
            assert update_manager._is_monitoring is False

    @pytest.mark.asyncio
    @patch('src.code_parser.code_parser.CodeParser')
    async def test_process_file_changes(self, mock_parser_class, update_manager, create_test_chunks):
        """Test processing file changes"""
        # Mock parser
        mock_parser = Mock()
        mock_chunks = create_test_chunks(2)
        mock_parser.parse_files.return_value = [Mock(chunks=mock_chunks)]
        mock_parser_class.return_value = mock_parser

        # Create test changes
        changes = [
            FileChange("src/User.java", ChangeType.MODIFIED, new_content="new code"),
            FileChange("src/Test.java", ChangeType.ADDED, new_content="test code")
        ]

        # Process changes
        result = await update_manager.process_file_changes(changes)

        assert result is True
        # Verify embedding service was called
        update_manager.embedding_service.generate_chunk_embeddings.assert_called()

    @pytest.mark.asyncio
    async def test_handle_file_deletion(self, update_manager):
        """Test handling file deletions"""
        # Create deletion change
        change = FileChange("src/DeletedUser.java", ChangeType.DELETED)

        # Process deletion
        result = await update_manager.handle_file_deletion(change)

        assert result is True
        # Verify chunks were deleted from vector store
        update_manager.vector_store.delete_chunks.assert_called()

    @pytest.mark.asyncio
    async def test_incremental_update_cycle(self, update_manager):
        """Test complete incremental update cycle"""
        # Mock git monitor
        with patch.object(update_manager, 'git_monitor') as mock_git:
            mock_git.check_for_updates.return_value = [
                FileChange("src/User.java", ChangeType.MODIFIED, new_content="updated code")
            ]

            # Mock file processing
            with patch.object(update_manager, 'process_file_changes', return_value=True):
                # Run update check
                result = await update_manager.check_and_process_updates()

                assert result is True
                mock_git.check_for_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_statistics(self, update_manager):
        """Test update statistics tracking"""
        # Initialize stats
        stats = update_manager.get_update_stats()

        assert "total_updates_processed" in stats
        assert stats["total_updates_processed"] == 0

        # Update stats manually for testing
        update_manager._update_stats = {
            "total_updates_processed": 5,
            "files_modified": 3,
            "files_added": 1,
            "files_deleted": 1,
            "last_update_time": 1234567890.0
        }

        updated_stats = update_manager.get_update_stats()
        assert updated_stats["total_updates_processed"] == 5
        assert updated_stats["files_modified"] == 3

    @pytest.mark.asyncio
    async def test_error_handling_in_updates(self, update_manager):
        """Test error handling during update processing"""
        # Mock git monitor to raise exception
        with patch.object(update_manager, 'git_monitor') as mock_git:
            mock_git.check_for_updates.side_effect = Exception("Git error")

            # Should handle error gracefully
            result = await update_manager.check_and_process_updates()
            assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_update_limiting(self, update_manager):
        """Test limiting concurrent updates"""
        # Set low concurrency limit
        update_manager.config.max_concurrent_updates = 1

        # Mock long-running update
        async def slow_update():
            await asyncio.sleep(0.1)
            return True

        with patch.object(update_manager, 'process_file_changes', side_effect=slow_update):
            # Start multiple updates
            tasks = [
                update_manager.check_and_process_updates()
                for _ in range(3)
            ]

            # Should handle concurrency properly
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # At least one should complete successfully
            successful = [r for r in results if r is True]
            assert len(successful) >= 1

    def test_health_check(self, update_manager):
        """Test update manager health check"""
        health = update_manager.health_check()

        assert "update_manager_status" in health
        assert "monitoring_active" in health
        assert "last_check_time" in health