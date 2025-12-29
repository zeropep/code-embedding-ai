import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from .git_monitor import GitMonitor
from .state_manager import StateManager
from .models import (UpdateRequest, UpdateResult, UpdateStatus, UpdateConfig,
                     ChangeType, ChangeDetectionResult)

from ..code_parser.code_parser import CodeParser
from ..code_parser.models import ParserConfig
from ..security.security_scanner import SecurityScanner
from ..security.models import SecurityConfig
from ..embeddings.embedding_service import EmbeddingService
from ..embeddings.models import EmbeddingConfig
from ..database.vector_store import VectorStore
from ..database.models import VectorDBConfig


logger = structlog.get_logger(__name__)


class UpdateService:
    """Service for managing incremental updates to the embedding database"""

    def __init__(self,
                 repo_path: Optional[str] = None,
                 state_dir: str = "./update_state",
                 parser_config: ParserConfig = None,
                 security_config: SecurityConfig = None,
                 embedding_config: EmbeddingConfig = None,
                 vector_config: VectorDBConfig = None,
                 update_config: UpdateConfig = None):

        self.repo_path = repo_path
        self.update_config = update_config or UpdateConfig()

        # Initialize components
        self.git_monitor = GitMonitor(repo_path, self.update_config) if repo_path else None
        self.state_manager = StateManager(state_dir, self.update_config)
        self.code_parser = CodeParser(parser_config or ParserConfig())
        self.security_scanner = SecurityScanner(security_config or SecurityConfig())
        self.embedding_service = EmbeddingService(embedding_config or EmbeddingConfig())
        self.vector_store = VectorStore(vector_config or VectorDBConfig())

        self._is_running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("UpdateService initialized",
                    repo_path=repo_path,
                    state_dir=state_dir)

    async def start(self) -> bool:
        """Start the update service"""
        try:
            # Connect to Git repository if repo_path is provided
            if self.repo_path and self.git_monitor:
                if self.git_monitor.connect():
                    logger.info("Connected to default repository", repo_path=self.repo_path)
                else:
                    logger.warning("Failed to connect to default repository, will use project repositories")

            # Connect to vector store
            if not self.vector_store.connect():
                logger.error("Failed to connect to vector store")
                return False

            # Start embedding service
            await self.embedding_service.start()

            self._is_running = True

            # Start periodic update task if configured (repo_path 없어도 시작)
            if self.update_config.check_interval_seconds > 0:
                self._update_task = asyncio.create_task(self._periodic_update_loop())
                logger.info("Periodic update loop started")

            logger.info("UpdateService started")
            return True

        except Exception as e:
            logger.error("Failed to start UpdateService", error=str(e))
            return False

    async def stop(self):
        """Stop the update service"""
        self._is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        await self.embedding_service.stop()
        self.vector_store.disconnect()

        logger.info("UpdateService stopped")

    async def request_update(self, request: UpdateRequest) -> UpdateResult:
        """Process an update request"""
        logger.info("Processing update request",
                    request_id=request.request_id,
                    repo_path=request.repo_path,
                    force_full=request.force_full_update)

        start_time = time.time()

        try:
            # Detect changes
            detection_result = await self._detect_changes(request)
            if not detection_result:
                return UpdateResult(
                    request_id=request.request_id,
                    status=UpdateStatus.FAILED,
                    changes_detected=[],
                    error_message="Failed to detect changes"
                )

            # Check if any changes were detected
            if not detection_result.has_changes and not request.force_full_update:
                logger.info("No new commits detected",
                            current_commit=detection_result.repo_state.commit_hash[:8])
                return UpdateResult(
                    request_id=request.request_id,
                    status=UpdateStatus.COMPLETED,
                    changes_detected=[],
                    processing_time=time.time() - start_time
                )

            # Process changes
            result = await self._process_changes(detection_result, request=request)

            # Save repository state
            self.state_manager.save_repository_state(detection_result.repo_state)

            # Create snapshot
            stats = self.vector_store.get_statistics()
            self.state_manager.create_snapshot(
                repo_state=detection_result.repo_state,
                total_chunks=stats.total_chunks,
                metadata={
                    "request_id": request.request_id,
                    "changes_processed": len(detection_result.detected_changes),
                    "is_full_scan": detection_result.is_full_scan
                }
            )

            # Save result
            result.processing_time = time.time() - start_time
            self.state_manager.save_update_result(result)

            logger.info("Update request completed",
                        request_id=request.request_id,
                        status=result.status.value,
                        processing_time=result.processing_time,
                        files_processed=result.files_processed)

            return result

        except Exception as e:
            logger.error("Update request failed", request_id=request.request_id, error=str(e))
            result = UpdateResult(
                request_id=request.request_id,
                status=UpdateStatus.FAILED,
                changes_detected=[],
                error_message=str(e),
                processing_time=time.time() - start_time
            )
            self.state_manager.save_update_result(result)
            return result

    async def quick_update(self, repo_path: str = None, project_id: str = None, project_name: str = None) -> UpdateResult:
        """Perform a quick update check"""
        repo_path = repo_path or self.repo_path

        request = UpdateRequest(
            request_id=f"quick_update_{int(time.time())}",
            repo_path=repo_path,
            force_full_update=False,
            project_id=project_id,
            project_name=project_name
        )

        return await self.request_update(request)

    async def force_full_update(self, repo_path: str = None, project_id: str = None, project_name: str = None) -> UpdateResult:
        """Force a complete repository scan and update"""
        repo_path = repo_path or self.repo_path

        request = UpdateRequest(
            request_id=f"full_update_{int(time.time())}",
            repo_path=repo_path,
            force_full_update=True,
            project_id=project_id,
            project_name=project_name
        )

        return await self.request_update(request)

    async def _detect_changes(self, request: UpdateRequest) -> Optional[ChangeDetectionResult]:
        """Detect changes in the repository"""
        try:
            # 프로젝트 정보 조회 (브랜치 정보 포함)
            git_branch = "main"  # 기본값
            git_remote_url = None

            if request.project_id:
                try:
                    from ..database.project_repository import ProjectRepository
                    project_repo = ProjectRepository()
                    project = project_repo.get(request.project_id)

                    if project:
                        git_branch = project.git_branch or "main"
                        git_remote_url = project.git_remote_url
                except Exception as e:
                    logger.warning("Failed to get project info, using defaults", error=str(e))

            # 프로젝트별 GitMonitor 생성 (브랜치 정보 전달)
            git_monitor = GitMonitor(
                repo_path=request.repo_path,
                config=self.update_config,
                branch=git_branch
            )

            if not git_monitor.connect():
                logger.error("Failed to connect to repository", repo_path=request.repo_path)
                return None

            # Pull latest changes from remote if available
            if git_remote_url:
                logger.info("Pulling latest changes from remote",
                           project_id=request.project_id,
                           remote_url=git_remote_url,
                           branch=git_branch)

                # Add or update remote
                if git_monitor.add_remote("origin", git_remote_url):
                    # Pull latest changes
                    if git_monitor.pull_latest("origin", git_branch):
                        logger.info("Successfully pulled latest changes")
                    else:
                        logger.warning("Failed to pull latest changes, continuing with local state")
                else:
                    logger.warning("Failed to add remote, continuing with local state")

            if request.force_full_update or self.state_manager.should_force_full_update():
                # Force full scan
                logger.info("Performing full repository scan")
                return git_monitor.detect_changes(since_commit=None)
            else:
                # Incremental update
                current_state = self.state_manager.current_state
                if current_state:
                    logger.info("Performing incremental update",
                                last_commit=current_state.commit_hash[:8])
                    # Use Git diff to detect changes since last known state
                    return git_monitor.detect_changes(since_commit=current_state.commit_hash)
                else:
                    logger.info("No previous state found, performing full scan")
                    return git_monitor.detect_changes(since_commit=None)

        except Exception as e:
            logger.error("Change detection failed", error=str(e))
            return None

    async def _process_changes(self,
                               changes: ChangeDetectionResult,
                               project_id: str = None,
                               project_name: str = None,
                               request: UpdateRequest = None) -> UpdateResult:
        """Process detected changes - flexible signature for both old and new usage"""
        # Support both old and new calling patterns
        if request is not None:
            # Old pattern: called with request and detection_result
            request_id = request.request_id
            detected_changes = changes.detected_changes if hasattr(changes, 'detected_changes') else []
            proj_id = request.project_id
            proj_name = request.project_name
        else:
            # New pattern: called with changes, project_id, project_name
            request_id = f"changes_{int(time.time())}"
            detected_changes = changes.changed_files if hasattr(changes, 'changed_files') else []
            proj_id = project_id
            proj_name = project_name

        result = UpdateResult(
            request_id=request_id,
            status=UpdateStatus.PROCESSING,
            changes_detected=detected_changes
        )

        try:
            # Group changes by type
            files_to_delete = []
            files_to_process = []

            # Handle both FileChange objects and Path objects
            for change in detected_changes:
                if hasattr(change, 'change_type'):
                    # FileChange object
                    if change.change_type == ChangeType.DELETED:
                        files_to_delete.append(change.file_path)
                    elif change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED, ChangeType.RENAMED]:
                        # Modified files need to delete old chunks first
                        if change.change_type in [ChangeType.MODIFIED, ChangeType.RENAMED]:
                            files_to_delete.append(change.file_path)
                        files_to_process.append(change.file_path)
                else:
                    # Path object (from changed_files set)
                    files_to_process.append(str(change))

            # Delete removed/modified files from vector store (to avoid duplicates)
            if files_to_delete:
                await self._delete_files_from_store(files_to_delete, result)

            # Process new/modified files
            if files_to_process:
                # Create a minimal request object for _process_files
                if request is not None:
                    await self._process_files(files_to_process, result, request)
                else:
                    # Create temporary request object
                    temp_request = UpdateRequest(
                        request_id=request_id,
                        repo_path=str(Path(files_to_process[0]).parent) if files_to_process else None,
                        project_id=proj_id,
                        project_name=proj_name
                    )
                    await self._process_files(files_to_process, result, temp_request)

            result.status = UpdateStatus.COMPLETED

        except Exception as e:
            logger.error("Failed to process changes", error=str(e))
            result.status = UpdateStatus.FAILED
            result.error_message = str(e)

        return result

    async def _delete_files_from_store(self, file_paths: List[str], result: UpdateResult):
        """Delete files from vector store"""
        logger.info("Deleting files from vector store", count=len(file_paths))

        for file_path in file_paths:
            try:
                delete_result = self.vector_store.delete_chunks_by_file(file_path)
                result.chunks_deleted += delete_result.successful_items

                logger.debug("File deleted from vector store",
                             file_path=file_path,
                             chunks_deleted=delete_result.successful_items)

            except Exception as e:
                logger.error("Failed to delete file from vector store",
                             file_path=file_path,
                             error=str(e))
                result.warnings.append(f"Failed to delete {file_path}: {str(e)}")

    async def _process_files(self, file_paths: List[str], result: UpdateResult, request: UpdateRequest = None):
        """Process new/modified files"""
        logger.info("Processing files", count=len(file_paths))

        # Convert relative paths to absolute paths
        # Use request.repo_path or fallback to self.repo_path
        base_repo_path = request.repo_path if request else self.repo_path
        if not base_repo_path:
            logger.error("No repository path available")
            raise ValueError("Repository path is required for processing files")

        full_file_paths = []
        for file_path in file_paths:
            full_path = str(Path(base_repo_path) / file_path)
            if Path(full_path).exists():
                full_file_paths.append(full_path)
            else:
                logger.warning("File not found", file_path=full_path)
                result.warnings.append(f"File not found: {file_path}")

        if not full_file_paths:
            logger.warning("No valid files to process")
            return

        try:
            # Parse files
            parsed_files = self.code_parser.parse_files(full_file_paths)
            if not parsed_files:
                logger.warning("No files were successfully parsed")
                return

            # Extract chunks
            all_chunks = []
            for parsed_file in parsed_files:
                all_chunks.extend(parsed_file.chunks)

            if not all_chunks:
                logger.warning("No chunks extracted from files")
                return

            # Set project metadata on all chunks if provided
            if request and (request.project_id or request.project_name):
                for chunk in all_chunks:
                    chunk.project_id = request.project_id
                    chunk.project_name = request.project_name
                logger.info("Project metadata set on chunks",
                           project_id=request.project_id,
                           project_name=request.project_name)

            logger.info("Files parsed", files_count=len(parsed_files), chunks_count=len(all_chunks))

            # Security scanning
            secured_chunks = self.security_scanner.scan_chunks(all_chunks)

            # Generate embeddings
            embedded_chunks = await self.embedding_service.generate_chunk_embeddings(secured_chunks)

            # Store in vector database
            store_result = self.vector_store.store_chunks(embedded_chunks)

            # Update result statistics
            result.files_processed = len(parsed_files)
            result.chunks_added += store_result.successful_items

            logger.info("Files processed successfully",
                        files_processed=len(parsed_files),
                        chunks_stored=store_result.successful_items)

        except Exception as e:
            logger.error("Failed to process files", error=str(e))
            raise

    async def _process_single_commit(
        self,
        project: Any,
        commit_hash: str,
        git_monitor: GitMonitor
    ) -> UpdateResult:
        """
        Process changes from a single commit

        Args:
            project: Project object
            commit_hash: Commit hash to process
            git_monitor: GitMonitor instance

        Returns:
            UpdateResult with processing status
        """
        logger.info("Processing single commit",
                   project_id=project.project_id,
                   commit=commit_hash[:8])

        # Detect changes for this specific commit
        changes = git_monitor.detect_changes_for_commit(commit_hash)

        if changes is None:
            return UpdateResult(
                request_id=f"commit_{commit_hash[:8]}",
                status=UpdateStatus.FAILED,
                changes_detected=[],
                error_message=f"Failed to detect changes for commit {commit_hash[:8]}",
                files_processed=0
            )

        if not changes.changed_files:
            logger.info("No relevant changes in commit",
                       commit=commit_hash[:8])
            return UpdateResult(
                request_id=f"commit_{commit_hash[:8]}",
                status=UpdateStatus.COMPLETED,
                changes_detected=[],
                files_processed=0
            )

        # Process the changes using existing logic
        # Create temporary request with repo_path
        temp_request = UpdateRequest(
            request_id=f"commit_{commit_hash[:8]}_{int(time.time())}",
            repo_path=project.repository_path,
            project_id=project.project_id,
            project_name=project.name
        )
        return await self._process_changes(
            changes=changes,
            request=temp_request
        )

    async def _periodic_update_loop(self):
        """Periodic update loop - 등록된 모든 프로젝트 순회"""
        logger.info("Starting periodic update loop",
                    interval_seconds=self.update_config.check_interval_seconds)

        while self._is_running:
            try:
                await asyncio.sleep(self.update_config.check_interval_seconds)

                if not self._is_running:
                    break

                logger.info("Periodic update check started")

                # 등록된 모든 active 프로젝트 조회
                from ..database.project_repository import ProjectRepository
                project_repo = ProjectRepository()
                projects = project_repo.get_all(status="active")

                if not projects:
                    logger.debug("No active projects to monitor")
                    continue

                logger.info("Checking projects for updates", project_count=len(projects))

                for project in projects:
                    if not project.repository_path:
                        logger.debug("Skipping project without repository_path",
                                    project_id=project.project_id)
                        continue

                    try:
                        logger.info("Checking project for updates",
                                   project_id=project.project_id,
                                   project_name=project.name,
                                   repo_path=project.repository_path)

                        # 프로젝트별 GitMonitor 생성 (브랜치 정보 전달)
                        project_git_monitor = GitMonitor(
                            repo_path=project.repository_path,
                            config=self.update_config,
                            branch=project.git_branch or "main"
                        )

                        if not project_git_monitor.connect():
                            logger.warning("Failed to connect to project repository",
                                          project_id=project.project_id,
                                          repo_path=project.repository_path)
                            continue

                        # Check if we need to update (compare remote commit with last processed)
                        should_update = True
                        if project.git_remote_url and project.last_processed_commit:
                            try:
                                remote_commit = project_git_monitor.get_remote_latest_commit(
                                    remote_url=project.git_remote_url,
                                    branch=project.git_branch or "main"
                                )

                                if remote_commit:
                                    if remote_commit == project.last_processed_commit:
                                        logger.info("No changes in remote repository, skipping update",
                                                   project_id=project.project_id,
                                                   remote_commit=remote_commit[:8],
                                                   last_processed=project.last_processed_commit[:8])
                                        should_update = False
                                    else:
                                        logger.info("Remote repository has new commits",
                                                   project_id=project.project_id,
                                                   remote_commit=remote_commit[:8],
                                                   last_processed=project.last_processed_commit[:8])
                                else:
                                    logger.warning("Failed to get remote commit, will attempt update anyway",
                                                  project_id=project.project_id)
                            except Exception as e:
                                logger.warning("Error checking remote commit, will attempt update anyway",
                                              project_id=project.project_id,
                                              error=str(e))

                        if not should_update:
                            continue

                        # remote가 있으면 pull
                        if project.git_remote_url:
                            if project_git_monitor.add_remote("origin", project.git_remote_url):
                                if project_git_monitor.pull_latest("origin", project.git_branch or "main"):
                                    logger.info("Successfully pulled latest changes",
                                               project_id=project.project_id)
                                else:
                                    logger.debug("No changes to pull",
                                                project_id=project.project_id)

                        # Get list of new commits
                        commits = project_git_monitor.get_commits_since(project.last_processed_commit or "")

                        if not commits:
                            logger.debug("No new commits to process",
                                        project_id=project.project_id)
                            continue

                        logger.info("Found new commits to process",
                                   project_id=project.project_id,
                                   commit_count=len(commits))

                        # Process each commit sequentially
                        for commit_hash in commits:
                            result = await self._process_single_commit(
                                project=project,
                                commit_hash=commit_hash,
                                git_monitor=project_git_monitor
                            )

                            if result.status == UpdateStatus.COMPLETED:
                                # Update last_processed_commit only on full success
                                from ..database.project_repository import ProjectRepository
                                project_repo = ProjectRepository()
                                project_repo.update(project.project_id, {
                                    "last_processed_commit": commit_hash
                                })
                                logger.info("Commit processed successfully",
                                           project_id=project.project_id,
                                           commit=commit_hash[:8])
                            else:
                                # Stop processing on failure, retry this commit next time
                                logger.warning("Commit processing failed, stopping batch",
                                              project_id=project.project_id,
                                              commit=commit_hash[:8],
                                              status=result.status.value,
                                              error_message=result.error_message)
                                break

                    except Exception as e:
                        logger.error("Error checking project",
                                    project_id=project.project_id,
                                    error=str(e))
                        continue

            except asyncio.CancelledError:
                logger.info("Periodic update loop cancelled")
                break
            except Exception as e:
                logger.error("Error in periodic update loop", error=str(e))
                await asyncio.sleep(60)  # 에러 시 1분 후 재시도

        logger.info("Periodic update loop stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current update service status"""
        git_info = None
        if self.git_monitor and self.git_monitor.repo:
            git_info = self.git_monitor._get_git_info()

        return {
            "service_running": self._is_running,
            "periodic_updates_enabled": self.update_config.check_interval_seconds > 0,
            "git_connected": self.git_monitor.repo if self.git_monitor else False,
            "vector_store_connected": self.vector_store._is_connected,
            "embedding_service_running": self.embedding_service._is_running,
            "current_git_info": git_info.repo_path if git_info else None,
            "state_summary": self.state_manager.get_state_summary(),
            "vector_store_stats": self.vector_store.get_statistics().to_dict()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            "status": "healthy",
            "components": {}
        }

        try:
            # Check Git repository
            if self.git_monitor and self.git_monitor.repo:
                health["components"]["git"] = {
                    "status": "healthy",
                    "connected": True,
                    "current_branch": self.git_monitor._resolve_branch()
                }
            else:
                health["components"]["git"] = {
                    "status": "not_configured",
                    "connected": False,
                    "message": "No default repository configured, using project repositories"
                }

            # Check vector store
            vector_health = self.vector_store.health_check()
            health["components"]["vector_store"] = vector_health

            # Check embedding service
            embedding_health = await self.embedding_service.health_check()
            health["components"]["embedding_service"] = embedding_health

            # Check state manager
            health["components"]["state_manager"] = {
                "status": "healthy",
                "has_current_state": self.state_manager.current_state is not None
            }

            # Determine overall status
            component_statuses = [comp.get("status") for comp in health["components"].values()]
            if "unhealthy" in component_statuses:
                health["status"] = "unhealthy"

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            logger.error("Health check failed", error=str(e))

        return health
