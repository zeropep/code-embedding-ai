import git
import hashlib
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import structlog

from .models import (FileChange, ChangeType, RepositoryState, GitInfo,
                     ChangeDetectionResult, UpdateConfig)


logger = structlog.get_logger(__name__)


class GitMonitor:
    """Monitor Git repository for changes"""

    def __init__(self, repo_path: str, config: UpdateConfig = None):
        self.repo_path = Path(repo_path)
        self.config = config or UpdateConfig()
        self.repo: Optional[git.Repo] = None
        self.last_known_state: Optional[RepositoryState] = None

        logger.info("GitMonitor initialized", repo_path=str(self.repo_path))

    def connect(self) -> bool:
        """Connect to Git repository"""
        try:
            if not self.repo_path.exists():
                logger.error("Repository path does not exist", path=str(self.repo_path))
                return False

            # Try to open as Git repository
            self.repo = git.Repo(str(self.repo_path))

            if self.repo.bare:
                logger.error("Bare repositories are not supported")
                return False

            logger.info("Connected to Git repository",
                        branch=self.repo.active_branch.name,
                        commit=self.repo.head.commit.hexsha[:8])
            return True

        except git.InvalidGitRepositoryError:
            logger.error("Path is not a valid Git repository", path=str(self.repo_path))
            return False
        except Exception as e:
            logger.error("Failed to connect to Git repository", error=str(e))
            return False

    def get_current_state(self) -> Optional[RepositoryState]:
        """Get current repository state"""
        if not self._ensure_connected():
            return None

        try:
            current_commit = self.repo.head.commit.hexsha
            current_branch = self.repo.active_branch.name
            timestamp = time.time()

            # Get file hashes for tracked files
            file_hashes = self._get_file_hashes()

            state = RepositoryState(
                commit_hash=current_commit,
                branch=current_branch,
                timestamp=timestamp,
                file_hashes=file_hashes
            )

            logger.debug("Repository state captured",
                         commit=current_commit[:8],
                         branch=current_branch,
                         total_files=state.total_files)

            return state

        except Exception as e:
            logger.error("Failed to get repository state", error=str(e))
            return None

    def detect_changes(self, since_commit: Optional[str] = None) -> Optional[ChangeDetectionResult]:
        """Detect changes since the last known state or given commit"""
        if not self._ensure_connected():
            return None

        start_time = time.time()

        try:
            current_state = self.get_current_state()
            if not current_state:
                return None

            git_info = self._get_git_info()
            detected_changes = []

            if since_commit:
                # Detect changes since specific commit
                detected_changes = self._get_changes_since_commit(since_commit)
                is_full_scan = False
            elif self.last_known_state:
                # Detect changes since last known state
                detected_changes = self._get_changes_since_state(self.last_known_state, current_state)
                is_full_scan = False
            else:
                # First run - treat all files as new
                detected_changes = self._get_all_files_as_changes()
                is_full_scan = True

            detection_time = time.time() - start_time

            result = ChangeDetectionResult(
                repo_state=current_state,
                detected_changes=detected_changes,
                git_info=git_info,
                detection_time=detection_time,
                is_full_scan=is_full_scan
            )

            logger.info("Change detection completed",
                        changes_count=len(detected_changes),
                        detection_time=detection_time,
                        is_full_scan=is_full_scan)

            # Update last known state
            self.last_known_state = current_state

            return result

        except Exception as e:
            logger.error("Change detection failed", error=str(e))
            return None

    def _get_changes_since_commit(self, since_commit: str) -> List[FileChange]:
        """Get changes since a specific commit"""
        changes = []

        try:
            # Get diff between commits
            commit_obj = self.repo.commit(since_commit)
            current_commit = self.repo.head.commit

            diff_index = commit_obj.diff(current_commit)

            for diff_item in diff_index:
                change = self._process_diff_item(diff_item)
                if change and self._should_include_file(change.file_path):
                    changes.append(change)

            logger.debug("Changes detected since commit",
                         since_commit=since_commit[:8],
                         changes_count=len(changes))

        except Exception as e:
            logger.error("Failed to get changes since commit",
                         since_commit=since_commit,
                         error=str(e))

        return changes

    def _get_changes_since_state(self, old_state: RepositoryState,
                                 current_state: RepositoryState) -> List[FileChange]:
        """Compare two repository states to find changes"""
        changes = []

        old_files = set(old_state.file_hashes.keys())
        current_files = set(current_state.file_hashes.keys())

        # Find added files
        added_files = current_files - old_files
        for file_path in added_files:
            if self._should_include_file(file_path):
                change = FileChange(
                    file_path=file_path,
                    change_type=ChangeType.ADDED,
                    file_hash=current_state.file_hashes[file_path]
                )
                changes.append(change)

        # Find deleted files
        deleted_files = old_files - current_files
        for file_path in deleted_files:
            if self._should_include_file(file_path):
                change = FileChange(
                    file_path=file_path,
                    change_type=ChangeType.DELETED,
                    file_hash=old_state.file_hashes[file_path]
                )
                changes.append(change)

        # Find modified files
        common_files = old_files & current_files
        for file_path in common_files:
            if self._should_include_file(file_path):
                old_hash = old_state.file_hashes[file_path]
                current_hash = current_state.file_hashes[file_path]

                if old_hash != current_hash:
                    change = FileChange(
                        file_path=file_path,
                        change_type=ChangeType.MODIFIED,
                        file_hash=current_hash
                    )
                    changes.append(change)

        logger.debug("Changes detected between states",
                     added=len([c for c in changes if c.change_type == ChangeType.ADDED]),
                     modified=len([c for c in changes if c.change_type == ChangeType.MODIFIED]),
                     deleted=len([c for c in changes if c.change_type == ChangeType.DELETED]))

        return changes

    def _get_all_files_as_changes(self) -> List[FileChange]:
        """Get all tracked files as new changes (for initial scan)"""
        changes = []

        try:
            # Get all tracked files
            for file_path, file_hash in self._get_file_hashes().items():
                if self._should_include_file(file_path):
                    file_full_path = self.repo_path / file_path

                    change = FileChange(
                        file_path=file_path,
                        change_type=ChangeType.ADDED,
                        file_hash=file_hash,
                        last_modified=os.path.getmtime(file_full_path) if file_full_path.exists() else None,
                        size_bytes=os.path.getsize(file_full_path) if file_full_path.exists() else 0
                    )
                    changes.append(change)

        except Exception as e:
            logger.error("Failed to get all files", error=str(e))

        return changes

    def _process_diff_item(self, diff_item) -> Optional[FileChange]:
        """Process a GitPython diff item into a FileChange"""
        try:
            if diff_item.change_type == 'A':  # Added
                return FileChange(
                    file_path=diff_item.b_path,
                    change_type=ChangeType.ADDED,
                    file_hash=diff_item.b_blob.hexsha if diff_item.b_blob else None
                )
            elif diff_item.change_type == 'D':  # Deleted
                return FileChange(
                    file_path=diff_item.a_path,
                    change_type=ChangeType.DELETED,
                    file_hash=diff_item.a_blob.hexsha if diff_item.a_blob else None
                )
            elif diff_item.change_type == 'M':  # Modified
                return FileChange(
                    file_path=diff_item.b_path,
                    change_type=ChangeType.MODIFIED,
                    file_hash=diff_item.b_blob.hexsha if diff_item.b_blob else None
                )
            elif diff_item.change_type == 'R':  # Renamed
                return FileChange(
                    file_path=diff_item.b_path,
                    change_type=ChangeType.RENAMED,
                    old_path=diff_item.a_path,
                    file_hash=diff_item.b_blob.hexsha if diff_item.b_blob else None
                )

        except Exception as e:
            logger.warning("Failed to process diff item", error=str(e))

        return None

    def _get_file_hashes(self) -> Dict[str, str]:
        """Get hash for all tracked files"""
        file_hashes = {}

        try:
            # Get all tracked files from the index
            for file_path in self.repo.git.ls_files().splitlines():
                try:
                    # Get blob hash for the file
                    blob = self.repo.head.commit.tree[file_path]
                    file_hashes[file_path] = blob.hexsha
                except (KeyError, AttributeError):
                    # File might be new or deleted, calculate hash from filesystem
                    full_path = self.repo_path / file_path
                    if full_path.exists():
                        file_hashes[file_path] = self._calculate_file_hash(full_path)

        except Exception as e:
            logger.error("Failed to get file hashes", error=str(e))

        return file_hashes

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-1 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha1(content).hexdigest()
        except Exception as e:
            logger.warning("Failed to calculate file hash", file=str(file_path), error=str(e))
            return ""

    def _get_git_info(self) -> GitInfo:
        """Get current Git repository information"""
        try:
            return GitInfo(
                repo_path=str(self.repo_path),
                current_branch=self.repo.active_branch.name,
                current_commit=self.repo.head.commit.hexsha,
                remote_url=self._get_remote_url(),
                is_dirty=self.repo.is_dirty(),
                untracked_files=self.repo.untracked_files,
                modified_files=[item.a_path for item in self.repo.index.diff(None)]
            )
        except Exception as e:
            logger.error("Failed to get Git info", error=str(e))
            return GitInfo(
                repo_path=str(self.repo_path),
                current_branch="unknown",
                current_commit="unknown"
            )

    def _get_remote_url(self) -> Optional[str]:
        """Get remote origin URL"""
        try:
            if self.repo.remotes:
                return self.repo.remotes.origin.url
        except Exception:
            pass
        return None

    def _should_include_file(self, file_path: str) -> bool:
        """Check if file should be included based on include/exclude patterns"""
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if self._matches_pattern(file_path, pattern):
                return False

        # Check include patterns
        for pattern in self.config.include_patterns:
            if self._matches_pattern(file_path, pattern):
                return True

        return False

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a glob pattern"""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern) or pattern in file_path

    def _ensure_connected(self) -> bool:
        """Ensure Git repository connection"""
        if not self.repo:
            return self.connect()
        return True

    def get_commit_info(self, commit_hash: str) -> Dict[str, Any]:
        """Get information about a specific commit"""
        if not self._ensure_connected():
            return {}

        try:
            commit = self.repo.commit(commit_hash)
            return {
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:8],
                "message": commit.message.strip(),
                "author": str(commit.author),
                "authored_date": commit.authored_date,
                "committed_date": commit.committed_date,
                "stats": commit.stats.total
            }
        except Exception as e:
            logger.error("Failed to get commit info", commit=commit_hash, error=str(e))
            return {}

    def reset_to_commit(self, commit_hash: str) -> bool:
        """Reset repository to a specific commit (careful!)"""
        if not self._ensure_connected():
            return False

        try:
            logger.warning("Resetting repository to commit", commit=commit_hash)
            self.repo.git.reset('--hard', commit_hash)
            return True
        except Exception as e:
            logger.error("Failed to reset repository", commit=commit_hash, error=str(e))
            return False
