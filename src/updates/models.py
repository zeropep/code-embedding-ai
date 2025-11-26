from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class UpdateStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileChange:
    """Represents a change to a file in the repository"""
    file_path: str
    change_type: ChangeType
    old_path: Optional[str] = None  # For renamed files
    file_hash: Optional[str] = None
    last_modified: Optional[float] = None
    size_bytes: int = 0

    def __post_init__(self):
        if self.last_modified is None:
            self.last_modified = time.time()


@dataclass
class RepositoryState:
    """Snapshot of repository state"""
    commit_hash: str
    branch: str
    timestamp: float
    file_hashes: Dict[str, str]  # file_path -> hash
    total_files: int = 0

    def __post_init__(self):
        self.total_files = len(self.file_hashes)


@dataclass
class UpdateRequest:
    """Request for incremental update"""
    request_id: str
    repo_path: str
    target_commit: Optional[str] = None
    target_branch: str = "main"
    force_full_update: bool = False
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    project_id: Optional[str] = None  # Project ID for multi-project support
    project_name: Optional[str] = None  # Project name for multi-project support
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.include_patterns is None:
            self.include_patterns = ["*.java", "*.kt", "*.html", "*.xml", "*.yml", "*.yaml", "*.properties"]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["*.class", "*.jar", "target/", "build/", ".git/"]


@dataclass
class UpdateResult:
    """Result of an incremental update operation"""
    request_id: str
    status: UpdateStatus
    changes_detected: List[FileChange]
    files_processed: int = 0
    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def total_changes(self) -> int:
        return len(self.changes_detected)

    @property
    def success_rate(self) -> float:
        if self.total_changes == 0:
            return 1.0
        return self.files_processed / self.total_changes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "total_changes": self.total_changes,
            "files_processed": self.files_processed,
            "chunks_added": self.chunks_added,
            "chunks_updated": self.chunks_updated,
            "chunks_deleted": self.chunks_deleted,
            "processing_time": self.processing_time,
            "success_rate": self.success_rate,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


@dataclass
class UpdateConfig:
    """Configuration for incremental updates"""
    check_interval_seconds: int = 300  # 5 minutes
    max_concurrent_updates: int = 3
    enable_file_watching: bool = False
    watch_extensions: List[str] = None
    git_diff_timeout: int = 30
    force_update_threshold_hours: int = 24

    # Backup settings
    enable_backup: bool = True
    backup_retention_days: int = 7

    # Performance settings
    batch_size: int = 50
    max_file_size_mb: int = 10

    def __post_init__(self):
        if self.watch_extensions is None:
            self.watch_extensions = [".java", ".kt", ".html", ".xml", ".yml", ".yaml", ".properties"]


@dataclass
class GitInfo:
    """Git repository information"""
    repo_path: str
    current_branch: str
    current_commit: str
    remote_url: Optional[str] = None
    is_dirty: bool = False
    untracked_files: List[str] = None
    modified_files: List[str] = None

    def __post_init__(self):
        if self.untracked_files is None:
            self.untracked_files = []
        if self.modified_files is None:
            self.modified_files = []


@dataclass
class ChangeDetectionResult:
    """Result of change detection operation"""
    repo_state: RepositoryState
    detected_changes: List[FileChange]
    git_info: GitInfo
    detection_time: float
    is_full_scan: bool = False

    @property
    def has_changes(self) -> bool:
        return len(self.detected_changes) > 0

    @property
    def changes_by_type(self) -> Dict[str, int]:
        counts = {}
        for change in self.detected_changes:
            change_type = change.change_type.value
            counts[change_type] = counts.get(change_type, 0) + 1
        return counts


@dataclass
class StateSnapshot:
    """Snapshot for tracking repository state over time"""
    snapshot_id: str
    timestamp: float
    repo_state: RepositoryState
    total_chunks: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UpdateMetrics:
    """Metrics for update operations"""
    total_update_requests: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    avg_processing_time: float = 0.0
    total_files_processed: int = 0
    total_chunks_modified: int = 0
    last_update_time: Optional[float] = None

    def update_success(self, processing_time: float, files_processed: int, chunks_modified: int):
        """Update metrics for successful operation"""
        self.total_update_requests += 1
        self.successful_updates += 1
        self.total_files_processed += files_processed
        self.total_chunks_modified += chunks_modified
        self.last_update_time = time.time()

        # Update average processing time
        total_successful = self.successful_updates
        if total_successful == 1:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = (
                (self.avg_processing_time * (total_successful - 1) + processing_time) / total_successful
            )

    def update_failure(self):
        """Update metrics for failed operation"""
        self.total_update_requests += 1
        self.failed_updates += 1

    @property
    def success_rate(self) -> float:
        if self.total_update_requests == 0:
            return 0.0
        return self.successful_updates / self.total_update_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_update_requests": self.total_update_requests,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "success_rate": self.success_rate,
            "avg_processing_time": self.avg_processing_time,
            "total_files_processed": self.total_files_processed,
            "total_chunks_modified": self.total_chunks_modified,
            "last_update_time": self.last_update_time
        }
