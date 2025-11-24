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
                 repo_path: str,
                 state_dir: str = "./update_state",
                 parser_config: ParserConfig = None,
                 security_config: SecurityConfig = None,
                 embedding_config: EmbeddingConfig = None,
                 vector_config: VectorDBConfig = None,
                 update_config: UpdateConfig = None):

        self.repo_path = repo_path
        self.update_config = update_config or UpdateConfig()

        # Initialize components
        self.git_monitor = GitMonitor(repo_path, self.update_config)
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
            # Connect to Git repository
            if not self.git_monitor.connect():
                logger.error("Failed to connect to Git repository")
                return False

            # Connect to vector store
            if not self.vector_store.connect():
                logger.error("Failed to connect to vector store")
                return False

            # Start embedding service
            await self.embedding_service.start()

            self._is_running = True

            # Start periodic update task if configured
            if self.update_config.check_interval_seconds > 0:
                self._update_task = asyncio.create_task(self._periodic_update_loop())

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
                logger.info("No changes detected", request_id=request.request_id)
                return UpdateResult(
                    request_id=request.request_id,
                    status=UpdateStatus.COMPLETED,
                    changes_detected=[],
                    processing_time=time.time() - start_time
                )

            # Process changes
            result = await self._process_changes(request, detection_result)

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

    async def quick_update(self, repo_path: str = None) -> UpdateResult:
        """Perform a quick update check"""
        repo_path = repo_path or self.repo_path

        request = UpdateRequest(
            request_id=f"quick_update_{int(time.time())}",
            repo_path=repo_path,
            force_full_update=False
        )

        return await self.request_update(request)

    async def force_full_update(self, repo_path: str = None) -> UpdateResult:
        """Force a complete repository scan and update"""
        repo_path = repo_path or self.repo_path

        request = UpdateRequest(
            request_id=f"full_update_{int(time.time())}",
            repo_path=repo_path,
            force_full_update=True
        )

        return await self.request_update(request)

    async def _detect_changes(self, request: UpdateRequest) -> Optional[ChangeDetectionResult]:
        """Detect changes in the repository"""
        try:
            if request.force_full_update or self.state_manager.should_force_full_update():
                # Force full scan
                logger.info("Performing full repository scan")
                return self.git_monitor.detect_changes(since_commit=None)
            else:
                # Incremental update
                current_state = self.state_manager.current_state
                if current_state:
                    logger.info("Performing incremental update",
                                last_commit=current_state.commit_hash[:8])
                    # Use Git diff to detect changes since last known state
                    return self.git_monitor.detect_changes(since_commit=current_state.commit_hash)
                else:
                    logger.info("No previous state found, performing full scan")
                    return self.git_monitor.detect_changes(since_commit=None)

        except Exception as e:
            logger.error("Change detection failed", error=str(e))
            return None

    async def _process_changes(self, request: UpdateRequest,
                               detection_result: ChangeDetectionResult) -> UpdateResult:
        """Process detected changes"""
        result = UpdateResult(
            request_id=request.request_id,
            status=UpdateStatus.PROCESSING,
            changes_detected=detection_result.detected_changes
        )

        try:
            # Group changes by type
            files_to_delete = []
            files_to_process = []

            for change in detection_result.detected_changes:
                if change.change_type == ChangeType.DELETED:
                    files_to_delete.append(change.file_path)
                elif change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED, ChangeType.RENAMED]:
                    files_to_process.append(change.file_path)

            # Delete removed files from vector store
            if files_to_delete:
                await self._delete_files_from_store(files_to_delete, result)

            # Process new/modified files
            if files_to_process:
                await self._process_files(files_to_process, result)

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

    async def _process_files(self, file_paths: List[str], result: UpdateResult):
        """Process new/modified files"""
        logger.info("Processing files", count=len(file_paths))

        # Convert relative paths to absolute paths
        full_file_paths = []
        for file_path in file_paths:
            full_path = str(Path(self.repo_path) / file_path)
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

    async def _periodic_update_loop(self):
        """Periodic update loop"""
        logger.info("Starting periodic update loop",
                    interval_seconds=self.update_config.check_interval_seconds)

        while self._is_running:
            try:
                await asyncio.sleep(self.update_config.check_interval_seconds)

                if not self._is_running:
                    break

                logger.debug("Performing periodic update check")
                await self.quick_update()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic update loop", error=str(e))
                # Continue running even if one update fails

        logger.info("Periodic update loop stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current update service status"""
        git_info = self.git_monitor._get_git_info() if self.git_monitor.repo else None

        return {
            "service_running": self._is_running,
            "periodic_updates_enabled": self.update_config.check_interval_seconds > 0,
            "git_connected": self.git_monitor.repo is not None,
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
            if self.git_monitor.repo:
                health["components"]["git"] = {
                    "status": "healthy",
                    "connected": True,
                    "current_branch": self.git_monitor.repo.active_branch.name
                }
            else:
                health["components"]["git"] = {
                    "status": "unhealthy",
                    "connected": False
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
