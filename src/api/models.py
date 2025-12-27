from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class RequestStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    PROCESSING = "processing"


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    METADATA = "metadata"
    HYBRID = "hybrid"


class ProcessingMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    AUTO = "auto"


# Request Models
class ProcessRepositoryRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the repository to process")
    mode: ProcessingMode = Field(default=ProcessingMode.AUTO, description="Processing mode")
    force_update: bool = Field(default=False, description="Force full update even if no changes detected")
    include_patterns: Optional[List[str]] = Field(default=None, description="File patterns to include")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="File patterns to exclude")
    project_id: Optional[str] = Field(default=None, description="Project ID for organizing chunks")
    project_name: Optional[str] = Field(default=None, description="Project name for organizing chunks")

    @validator('repo_path')
    def validate_repo_path(cls, v):
        if not v.strip():
            raise ValueError("Repository path cannot be empty")
        return v.strip()


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    search_type: SearchType = Field(default=SearchType.SEMANTIC, description="Type of search to perform")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    project_id: Optional[str] = Field(default=None, description="Filter by specific project ID")
    include_content: bool = Field(default=True, description="Include code content in results")
    include_embeddings: bool = Field(default=False, description="Include embedding vectors in results")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class UpdateConfigRequest(BaseModel):
    check_interval_seconds: Optional[int] = Field(None, ge=60, description="Update check interval in seconds")
    max_concurrent_updates: Optional[int] = Field(None, ge=1, le=10, description="Maximum concurrent updates")
    enable_file_watching: Optional[bool] = Field(None, description="Enable file system watching")
    force_update_threshold_hours: Optional[int] = Field(None, ge=1, description="Hours before forcing full update")


# Response Models
class BaseResponse(BaseModel):
    status: RequestStatus
    message: str
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    status: RequestStatus = RequestStatus.ERROR
    error_details: Optional[Dict[str, Any]] = None


class ProcessingResponse(BaseResponse):
    status: RequestStatus = RequestStatus.PROCESSING
    progress: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseResponse):
    status: RequestStatus = RequestStatus.SUCCESS
    data: Optional[Dict[str, Any]] = None


class ProcessRepositoryResponse(BaseResponse):
    request_id: str
    processing_summary: Optional[Dict[str, Any]] = None
    parsing_stats: Optional[Dict[str, Any]] = None
    security_stats: Optional[Dict[str, Any]] = None
    embedding_stats: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class SearchResult(BaseModel):
    chunk_id: str
    similarity_score: float
    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    layer_type: str
    start_line: int
    end_line: int
    content: Optional[str] = None
    embedding_vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseResponse):
    results: List[SearchResult]
    total_results: int
    query_info: Dict[str, Any]
    search_time_ms: float


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: float = Field(default_factory=time.time)
    components: Dict[str, Any]
    uptime_seconds: float
    version: Optional[str] = None


class SystemStatusResponse(BaseModel):
    service_status: Dict[str, Any]
    git_info: Optional[Dict[str, Any]] = None
    database_stats: Dict[str, Any]
    update_metrics: Dict[str, Any]
    last_update_time: Optional[float] = None


class MetricsResponse(BaseModel):
    embedding_metrics: Dict[str, Any]
    update_metrics: Dict[str, Any]
    vector_store_stats: Dict[str, Any]
    system_metrics: Dict[str, Any]


class ConfigResponse(BaseModel):
    current_config: Dict[str, Any]
    config_updated: bool = False
    restart_required: bool = False


# File and Repository Models
class FileInfo(BaseModel):
    file_path: str
    language: str
    chunk_count: int
    total_tokens: int
    file_hash: str
    last_modified: float
    sensitivity_level: str


class RepositoryInfo(BaseModel):
    repo_path: str
    current_branch: str
    current_commit: str
    total_files: int
    supported_files: int
    last_scan_time: Optional[float] = None
    is_git_repo: bool


class UpdateStatus(BaseModel):
    update_id: str
    status: str
    started_at: float
    completed_at: Optional[float] = None
    files_processed: int
    chunks_added: int
    chunks_updated: int
    chunks_deleted: int
    errors: List[str] = []


# Statistics Models
class ProcessingStats(BaseModel):
    total_files_processed: int
    total_chunks_created: int
    successful_embeddings: int
    failed_embeddings: int
    processing_time_seconds: float
    avg_tokens_per_chunk: float


class SecurityStats(BaseModel):
    total_secrets_found: int
    secrets_masked: int
    high_sensitivity_files: int
    security_scan_time_seconds: float
    sensitivity_distribution: Dict[str, int]


class DatabaseStats(BaseModel):
    total_chunks: int
    total_files: int
    collection_size_mb: float
    avg_search_time_ms: float
    language_distribution: Dict[str, int]
    layer_distribution: Dict[str, int]


# Batch Operations
class BatchProcessRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to process")
    update_existing: bool = Field(default=True, description="Update existing chunks")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    project_id: Optional[str] = Field(default=None, description="Project ID for organizing chunks")
    project_name: Optional[str] = Field(default=None, description="Project name for organizing chunks")


class BatchDeleteRequest(BaseModel):
    file_paths: Optional[List[str]] = Field(None, description="Delete chunks from specific files")
    filters: Optional[Dict[str, Any]] = Field(None, description="Delete chunks matching filters")
    confirm: bool = Field(default=False, description="Confirmation flag for deletion")

    @validator('confirm')
    def validate_confirmation(cls, v, values):
        if not v:
            raise ValueError("Deletion must be confirmed by setting 'confirm' to true")
        return v


# Query Models for Advanced Search
class AdvancedSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="Multiple search queries")
    search_type: SearchType = Field(default=SearchType.SEMANTIC)
    combine_results: bool = Field(default=True, description="Combine results from multiple queries")
    top_k_per_query: int = Field(default=5, ge=1, le=50)
    overall_top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class SimilarCodeRequest(BaseModel):
    code_snippet: str = Field(..., description="Code snippet to find similar code for")
    language: Optional[str] = Field(None, description="Programming language of the snippet")
    top_k: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    include_same_file: bool = Field(default=False, description="Include results from the same file")
    project_id: Optional[str] = Field(default=None, description="Filter by specific project ID")


# Administrative Models
class BackupRequest(BaseModel):
    backup_name: Optional[str] = Field(None, description="Custom backup name")
    include_embeddings: bool = Field(default=True, description="Include embedding vectors in backup")
    compress: bool = Field(default=True, description="Compress backup files")


class RestoreRequest(BaseModel):
    backup_name: str = Field(..., description="Name of backup to restore")
    confirm_restore: bool = Field(default=False, description="Confirmation for restore operation")

    @validator('confirm_restore')
    def validate_restore_confirmation(cls, v):
        if not v:
            raise ValueError("Restore operation must be confirmed")
        return v


# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)


class ProgressUpdate(BaseModel):
    operation_id: str
    progress_percent: float
    status: str
    current_step: str
    total_steps: int
    current_step_num: int
    estimated_time_remaining: Optional[float] = None


# Project Management Models
class ProjectInfo(BaseModel):
    id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    chunk_count: int = Field(..., description="Number of chunks in this project")


class ProjectListResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    message: str
    projects: List[ProjectInfo]
    total_projects: int
    timestamp: float = Field(default_factory=time.time)


class ProjectStatsResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    project_id: str
    project_name: str
    total_chunks: int
    total_files: int
    total_tokens: int
    avg_tokens_per_chunk: float
    languages: Dict[str, int] = Field(..., description="Distribution of programming languages")
    layer_types: Dict[str, int] = Field(..., description="Distribution of layer types")
    last_updated: float
    timestamp: float = Field(default_factory=time.time)


# Project CRUD Models
class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Project name", min_length=1, max_length=200)
    repository_path: str = Field(..., description="Repository path for git monitoring")
    description: Optional[str] = Field(None, description="Project description", max_length=1000)
    git_remote_url: Optional[str] = Field(None, description="Git remote repository URL")
    git_branch: str = Field(default="main", description="Git branch to monitor")

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()

    @validator('repository_path')
    def validate_path(cls, v):
        if not v.strip():
            raise ValueError("Repository path cannot be empty")
        return v.strip()


class ProjectDetailResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    project_id: str
    name: str
    repository_path: str
    description: Optional[str] = None
    git_remote_url: Optional[str] = None
    git_branch: str = "main"
    project_status: str = Field(..., description="Project status (active, archived, etc.)")
    created_at: str
    updated_at: str
    total_chunks: int
    total_files: int
    last_indexed_at: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class ProjectCreateResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    message: str
    project_id: str
    name: str
    repository_path: str
    git_remote_url: Optional[str] = None
    git_branch: str = "main"
    created_at: str
    timestamp: float = Field(default_factory=time.time)


class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    repository_path: Optional[str] = Field(None)
    description: Optional[str] = Field(None, max_length=1000)
    git_remote_url: Optional[str] = Field(None, description="Git remote repository URL")
    git_branch: Optional[str] = Field(None, description="Git branch to monitor")
    status: Optional[str] = Field(None, description="Project status (active, archived)")

    @validator('name')
    def validate_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip() if v else None

    @validator('status')
    def validate_status(cls, v):
        if v is not None and v not in ['active', 'archived', 'initializing']:
            raise ValueError("Invalid status. Must be 'active', 'archived', or 'initializing'")
        return v


class ProjectUpdateResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    message: str
    project: ProjectDetailResponse
    timestamp: float = Field(default_factory=time.time)


class ProjectDeleteResponse(BaseModel):
    status: RequestStatus = RequestStatus.SUCCESS
    message: str
    project_id: str
    chunks_deleted: int = 0
    timestamp: float = Field(default_factory=time.time)
