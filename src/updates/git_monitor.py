import git
import hashlib
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import structlog

from .models import (FileChange, ChangeType, RepositoryState, GitInfo,
                     ChangeDetectionResult, UpdateConfig)


logger = structlog.get_logger(__name__)


class GitMonitor:
    """Monitor Git repository for changes"""

    def __init__(self, repo_path: str, config: UpdateConfig = None, branch: str = "main"):
        self.repo_path = Path(repo_path)
        self.config = config or UpdateConfig()
        self.branch = branch  # 프로젝트에서 전달받은 브랜치
        self.repo: Optional[git.Repo] = None
        self.last_known_state: Optional[RepositoryState] = None

        logger.info("GitMonitor initialized", repo_path=str(self.repo_path), branch=self.branch)

    def _resolve_branch(self) -> str:
        """Resolve branch name with main/master compatibility"""
        if not self.repo:
            return self.branch

        # 지정된 브랜치가 존재하는지 확인
        try:
            self.repo.refs[self.branch]
            return self.branch
        except (IndexError, KeyError):
            pass

        # main/master 호환성 처리
        if self.branch == "main":
            try:
                self.repo.refs["master"]
                logger.info("Branch 'main' not found, using 'master'")
                return "master"
            except (IndexError, KeyError):
                pass
        elif self.branch == "master":
            try:
                self.repo.refs["main"]
                logger.info("Branch 'master' not found, using 'main'")
                return "main"
            except (IndexError, KeyError):
                pass

        # 둘 다 없으면 원래 브랜치 반환
        return self.branch

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

            # 안전하게 브랜치 및 커밋 정보 가져오기
            resolved_branch = self._resolve_branch()
            commit_sha = "unknown"
            try:
                if self.repo.head.is_valid():
                    commit_sha = self.repo.head.commit.hexsha[:8]
            except (ValueError, TypeError):
                # 빈 저장소이거나 Detached HEAD 상태
                logger.warning("Cannot read commit SHA (empty repo or detached HEAD)")

            logger.info("Connected to Git repository",
                        branch=resolved_branch,
                        commit=commit_sha)
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
            # 안전하게 커밋 SHA 가져오기
            try:
                current_commit = self.repo.head.commit.hexsha
            except (ValueError, TypeError):
                # 빈 저장소 - 커밋이 없음
                logger.warning("No commits in repository, cannot get state")
                return None

            current_branch = self._resolve_branch()
            timestamp = time.time()

            # Get file hashes for tracked files
            try:
                file_hashes = self._get_file_hashes()
            except OSError as e:
                logger.error("Failed to get file hashes (OS error)", error=str(e))
                file_hashes = {}

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

        except OSError as e:
            logger.error("Failed to get repository state (OS error)", error=str(e))
            return None
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
        """Get changes since a specific commit (checks final state of changed files)"""
        changes = []
        changed_files = set()  # Track all files that changed

        try:
            # Get commit timestamp
            commit_obj = self.repo.commit(since_commit)
            commit_timestamp = commit_obj.committed_date

            logger.debug("Finding changes since commit",
                         since_commit=since_commit[:8],
                         commit_date=time.strftime('%Y-%m-%d %H:%M:%S',
                                                   time.localtime(commit_timestamp)))

            # Get all commits between since_commit and HEAD
            commits = list(self.repo.iter_commits(f'{since_commit}..HEAD'))

            if len(commits) == 0:
                logger.info("No new commits since last check",
                            last_commit=since_commit[:8],
                            current_commit=self.repo.head.commit.hexsha[:8])
            else:
                logger.info("New commits detected",
                            new_commits=len(commits),
                            last_commit=since_commit[:8],
                            current_commit=self.repo.head.commit.hexsha[:8])

            logger.debug("Processing commits", commit_count=len(commits))

            # Collect all files that changed in any commit
            for commit in commits:
                for parent in commit.parents:
                    diff_index = parent.diff(commit)

                    for diff_item in diff_index:
                        file_path = diff_item.b_path or diff_item.a_path
                        
                        if self._should_include_file(file_path):
                            changed_files.add(file_path)

            logger.debug("Changed files collected", total_files=len(changed_files))

            # Now check final state of each changed file
            current_commit = self.repo.head.commit
            
            for file_path in changed_files:
                try:
                    # Check if file exists in HEAD
                    file_exists_in_head = False
                    file_hash = None
                    
                    try:
                        blob = current_commit.tree[file_path]
                        file_exists_in_head = True
                        file_hash = blob.hexsha
                    except (KeyError, AttributeError):
                        file_exists_in_head = False
                    
                    # Check if file existed in since_commit
                    file_existed_before = False
                    try:
                        commit_obj.tree[file_path]
                        file_existed_before = True
                    except (KeyError, AttributeError):
                        file_existed_before = False
                    
                    # Determine change type based on final state
                    if file_exists_in_head and not file_existed_before:
                        # File was added
                        change = FileChange(
                            file_path=file_path,
                            change_type=ChangeType.ADDED,
                            file_hash=file_hash
                        )
                        changes.append(change)
                    elif file_exists_in_head and file_existed_before:
                        # File was modified
                        change = FileChange(
                            file_path=file_path,
                            change_type=ChangeType.MODIFIED,
                            file_hash=file_hash
                        )
                        changes.append(change)
                    elif not file_exists_in_head and file_existed_before:
                        # File was deleted
                        change = FileChange(
                            file_path=file_path,
                            change_type=ChangeType.DELETED,
                            file_hash=None
                        )
                        changes.append(change)
                    
                except Exception as e:
                    logger.warning("Failed to process file", file_path=file_path, error=str(e))
                    continue

            logger.info("Changes detected since commit",
                        since_commit=since_commit[:8],
                        commits_processed=len(commits),
                        files_changed=len(changes),
                        added=len([c for c in changes if c.change_type == ChangeType.ADDED]),
                        modified=len([c for c in changes if c.change_type == ChangeType.MODIFIED]),
                        deleted=len([c for c in changes if c.change_type == ChangeType.DELETED]))

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
            # HEAD commit이 있는지 먼저 확인
            head_commit = None
            try:
                head_commit = self.repo.head.commit
            except (ValueError, TypeError):
                # 빈 저장소 - 커밋이 없음
                pass

            # Get all tracked files from the index
            for file_path in self.repo.git.ls_files().splitlines():
                try:
                    if head_commit:
                        # Get blob hash for the file from commit tree
                        blob = head_commit.tree[file_path]
                        file_hashes[file_path] = blob.hexsha
                    else:
                        # 빈 저장소 - 파일시스템에서 직접 해시 계산
                        full_path = self.repo_path / file_path
                        if full_path.exists():
                            file_hashes[file_path] = self._calculate_file_hash(full_path)
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
            # 기본값 설정
            current_commit = "unknown"
            is_dirty = False
            untracked_files = []
            modified_files = []

            # 커밋 SHA 가져오기
            try:
                current_commit = self.repo.head.commit.hexsha
            except (ValueError, TypeError) as e:
                logger.warning("Cannot get commit SHA", error=str(e))

            # is_dirty 확인 (Windows에서 Errno 22 발생 가능)
            try:
                is_dirty = self.repo.is_dirty()
            except OSError as e:
                logger.warning("Cannot check is_dirty (Windows compatibility issue)", error=str(e))

            # untracked_files 가져오기
            try:
                untracked_files = self.repo.untracked_files
            except OSError as e:
                logger.warning("Cannot get untracked_files", error=str(e))

            # modified_files 가져오기 (index.diff가 Windows에서 문제 발생 가능)
            try:
                modified_files = [item.a_path for item in self.repo.index.diff(None)]
            except OSError as e:
                logger.warning("Cannot get modified_files (Windows compatibility issue)", error=str(e))

            return GitInfo(
                repo_path=str(self.repo_path),
                current_branch=self._resolve_branch(),
                current_commit=current_commit,
                remote_url=self._get_remote_url(),
                is_dirty=is_dirty,
                untracked_files=untracked_files,
                modified_files=modified_files
            )
        except Exception as e:
            logger.error("Failed to get Git info", error=str(e))
            return GitInfo(
                repo_path=str(self.repo_path),
                current_branch=self.branch,
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

    def pull_latest(self, remote_name: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull latest changes from remote repository"""
        if not self._ensure_connected():
            logger.error("Cannot pull: not connected to repository")
            return False

        try:
            # Check if remote exists
            if not self.repo.remotes:
                logger.error("No remote repositories configured")
                return False

            # Get the specified remote or origin
            try:
                remote = self.repo.remote(remote_name)
            except ValueError:
                logger.error("Remote not found", remote_name=remote_name)
                return False

            # Use configured branch if not specified
            if not branch:
                branch = self._resolve_branch()

            logger.info("Pulling latest changes from remote",
                       remote=remote_name,
                       branch=branch)

            # Fetch latest changes
            fetch_info = remote.fetch()
            logger.debug("Fetch completed", info=str(fetch_info))

            # Pull changes
            pull_info = remote.pull(branch)

            # Check for conflicts or errors
            for info in pull_info:
                if info.flags & info.ERROR:
                    logger.error("Pull error detected", info=str(info))
                    return False

            logger.info("Successfully pulled latest changes",
                       remote=remote_name,
                       branch=branch)
            return True

        except git.GitCommandError as e:
            logger.error("Git command failed during pull",
                        error=str(e),
                        stderr=e.stderr if hasattr(e, 'stderr') else None)
            return False
        except Exception as e:
            logger.error("Failed to pull latest changes", error=str(e))
            return False

    def add_remote(self, name: str, url: str) -> bool:
        """Add a remote repository"""
        if not self._ensure_connected():
            logger.error("Cannot add remote: not connected to repository")
            return False

        try:
            # Check if remote already exists
            try:
                existing_remote = self.repo.remote(name)
                logger.warning("Remote already exists, updating URL",
                             name=name,
                             old_url=existing_remote.url,
                             new_url=url)
                existing_remote.set_url(url)
            except ValueError:
                # Remote doesn't exist, create it
                self.repo.create_remote(name, url)
                logger.info("Remote added", name=name, url=url)

            return True

        except Exception as e:
            logger.error("Failed to add remote",
                        name=name,
                        url=url,
                        error=str(e))
            return False

    def get_remote_latest_commit(self, remote_url: str, branch: str = "main") -> Optional[str]:
        """
        Get latest commit ID from remote repository without local clone
        Uses git ls-remote command

        Args:
            remote_url: Remote repository URL
            branch: Branch name (default: main)

        Returns:
            Commit hash (SHA) or None if failed
        """
        try:
            import subprocess

            result = subprocess.run(
                ["git", "ls-remote", remote_url, f"refs/heads/{branch}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout:
                commit_hash = result.stdout.split()[0]
                logger.info("Retrieved remote commit",
                           remote_url=remote_url,
                           branch=branch,
                           commit=commit_hash[:8])
                return commit_hash
            else:
                logger.warning("Failed to get remote commit",
                              remote_url=remote_url,
                              branch=branch,
                              stderr=result.stderr)
                return None

        except subprocess.TimeoutExpired:
            logger.error("Timeout getting remote commit",
                        remote_url=remote_url,
                        branch=branch)
            return None
        except Exception as e:
            logger.error("Failed to get remote commit",
                        remote_url=remote_url,
                        branch=branch,
                        error=str(e))
            return None

    def get_commits_since(self, since_commit: str) -> List[str]:
        """
        Get list of commit hashes between since_commit and HEAD (oldest first)

        Args:
            since_commit: Starting commit hash (exclusive)

        Returns:
            List of commit hashes in chronological order (oldest first)
        """
        try:
            commits = list(self.repo.iter_commits(f'{since_commit}..HEAD'))
            # iter_commits returns newest first, reverse for chronological order
            return [c.hexsha for c in reversed(commits)]
        except Exception as e:
            logger.error("Failed to get commits since", since_commit=since_commit[:8], error=str(e))
            return []

    def detect_changes_for_commit(self, commit_hash: str) -> Optional[ChangeDetectionResult]:
        """
        Detect changes introduced by a single commit

        Args:
            commit_hash: The commit to analyze

        Returns:
            ChangeDetectionResult with changes from this specific commit only
        """
        start_time = time.time()

        try:
            commit = self.repo.commit(commit_hash)
            parent = commit.parents[0] if commit.parents else None

            changed_files: Set[Path] = set()

            if parent:
                # Compare with parent commit
                diff = parent.diff(commit)
            else:
                # First commit - all files are new
                diff = commit.diff(None)  # Compare with empty tree

            for diff_item in diff:
                # Get the file path (a_path for deleted, b_path for added/modified)
                file_path = diff_item.b_path if diff_item.b_path else diff_item.a_path
                if file_path:
                    full_path = self.repo_path / file_path
                    # Only include if file exists and matches our extensions
                    if full_path.exists() and self._should_process_file(full_path):
                        changed_files.add(full_path)

            detection_time = time.time() - start_time

            logger.info("Change detection for commit completed",
                       commit=commit_hash[:8],
                       changes_count=len(changed_files),
                       detection_time=detection_time)

            return ChangeDetectionResult(
                changed_files=changed_files,
                is_full_scan=False,
                detection_time=detection_time
            )

        except Exception as e:
            logger.error("Failed to detect changes for commit",
                        commit=commit_hash[:8],
                        error=str(e))
            return None

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on include/exclude patterns"""
        file_str = str(file_path)

        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if self._matches_pattern(file_str, pattern):
                return False

        # Check include patterns
        for pattern in self.config.include_patterns:
            if self._matches_pattern(file_str, pattern):
                return True

        return False
