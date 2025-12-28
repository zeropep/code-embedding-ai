import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
import structlog
import shutil

from .models import (RepositoryState, StateSnapshot, UpdateConfig,
                     UpdateMetrics, UpdateResult, UpdateStatus)
from .state_repository import StateRepository


logger = structlog.get_logger(__name__)


class StateManager:
    """Manages repository state persistence and tracking"""

    def __init__(self, state_dir: str, config: UpdateConfig = None, project_id: Optional[str] = None):
        self.state_dir = Path(state_dir)
        self.config = config or UpdateConfig()
        self.project_id = project_id

        # Legacy JSON file paths (for migration)
        self.state_file = self.state_dir / "repository_state.json"
        self.snapshots_file = self.state_dir / "state_snapshots.json"
        self.metrics_file = self.state_dir / "update_metrics.json"

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize repository
        self.repository = StateRepository()

        self.current_state: Optional[RepositoryState] = None
        self.snapshots: List[StateSnapshot] = []
        self.metrics: UpdateMetrics = UpdateMetrics()

        # Migrate from JSON if needed
        self._migrate_from_json()

        # Load existing state
        self._load_state()

        logger.info("StateManager initialized", state_dir=str(self.state_dir), project_id=project_id)

    def save_repository_state(self, state: RepositoryState) -> bool:
        """Save current repository state"""
        try:
            success = self.repository.save_repository_state(state, self.project_id)
            if success:
                self.current_state = state
            return success

        except Exception as e:
            logger.error("Failed to save repository state", error=str(e))
            return False

    def load_repository_state(self) -> Optional[RepositoryState]:
        """Load saved repository state"""
        try:
            state = self.repository.load_repository_state(self.project_id)
            if state:
                self.current_state = state
                logger.info("Repository state loaded",
                            commit=state.commit_hash[:8],
                            branch=state.branch,
                            age_hours=(time.time() - state.timestamp) / 3600)
            return state

        except Exception as e:
            logger.error("Failed to load repository state", error=str(e))
            return None

    def create_snapshot(self, repo_state: RepositoryState, total_chunks: int,
                        metadata: Dict[str, Any] = None) -> StateSnapshot:
        """Create a state snapshot"""
        snapshot = StateSnapshot(
            snapshot_id=f"snapshot_{int(time.time())}",
            timestamp=time.time(),
            repo_state=repo_state,
            total_chunks=total_chunks,
            metadata=metadata or {}
        )

        # Save to database
        self.repository.save_snapshot(snapshot, self.project_id)

        # Keep in memory (for compatibility)
        self.snapshots.append(snapshot)

        # Keep only recent snapshots (configurable retention)
        self._cleanup_old_snapshots()

        logger.info("State snapshot created",
                    snapshot_id=snapshot.snapshot_id,
                    total_chunks=total_chunks)

        return snapshot

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent state snapshot"""
        # Get from database
        snapshot = self.repository.get_latest_snapshot(self.project_id)
        if snapshot and snapshot not in self.snapshots:
            self.snapshots.append(snapshot)
        return snapshot

    def get_snapshots_since(self, since_timestamp: float) -> List[StateSnapshot]:
        """Get snapshots since a specific timestamp"""
        # Get from database
        return self.repository.get_snapshots_since(since_timestamp, self.project_id)

    def save_update_result(self, result: UpdateResult) -> bool:
        """Save update operation result and update metrics"""
        try:
            # Update metrics
            if result.status == UpdateStatus.COMPLETED:
                total_chunks_modified = result.chunks_added + result.chunks_updated + result.chunks_deleted
                self.metrics.update_success(
                    processing_time=result.processing_time,
                    files_processed=result.files_processed,
                    chunks_modified=total_chunks_modified
                )
            else:
                self.metrics.update_failure()

            # Save metrics to database
            self.repository.save_metrics(self.metrics, self.project_id)

            # Save individual result to database
            self.repository.save_update_result(result, self.project_id)

            logger.info("Update result saved",
                        request_id=result.request_id,
                        status=result.status.value,
                        files_processed=result.files_processed)

            return True

        except Exception as e:
            logger.error("Failed to save update result", error=str(e))
            return False

    def get_metrics(self) -> UpdateMetrics:
        """Get current update metrics"""
        return self.metrics

    def reset_metrics(self) -> bool:
        """Reset update metrics"""
        try:
            self.metrics = UpdateMetrics()
            self.repository.save_metrics(self.metrics, self.project_id)
            logger.info("Update metrics reset")
            return True
        except Exception as e:
            logger.error("Failed to reset metrics", error=str(e))
            return False

    def should_force_full_update(self) -> bool:
        """Check if a full update should be forced based on time threshold"""
        if not self.current_state:
            return True

        hours_since_last = (time.time() - self.current_state.timestamp) / 3600
        should_force = hours_since_last >= self.config.force_update_threshold_hours

        if should_force:
            logger.info("Force full update triggered",
                        hours_since_last=hours_since_last,
                        threshold=self.config.force_update_threshold_hours)

        return should_force

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state and statistics"""
        total_snapshots = self.repository.count_snapshots(self.project_id)

        summary = {
            "has_current_state": self.current_state is not None,
            "total_snapshots": total_snapshots,
            "metrics": self.metrics.to_dict()
        }

        if self.current_state:
            summary.update({
                "current_commit": self.current_state.commit_hash[:8],
                "current_branch": self.current_state.branch,
                "total_files": self.current_state.total_files,
                "state_age_hours": (time.time() - self.current_state.timestamp) / 3600
            })

        latest_snapshot = self.get_latest_snapshot()
        if latest_snapshot:
            summary.update({
                "latest_snapshot_id": latest_snapshot.snapshot_id,
                "latest_snapshot_chunks": latest_snapshot.total_chunks,
                "snapshot_age_hours": (time.time() - latest_snapshot.timestamp) / 3600
            })

        return summary

    def cleanup_old_data(self, retention_days: int = None) -> int:
        """Clean up old state data"""
        if retention_days is None:
            retention_days = self.config.backup_retention_days

        cutoff_time = time.time() - (retention_days * 24 * 3600)
        removed_count = 0

        # Remove old snapshots from database
        removed_count += self.repository.delete_snapshots_before(cutoff_time, self.project_id)

        # Remove old update results from database
        removed_count += self.repository.delete_results_before(cutoff_time, self.project_id)

        # Clear in-memory snapshots
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        logger.info("Cleaned up old data",
                    removed_count=removed_count,
                    retention_days=retention_days)

        return removed_count

    def _load_state(self):
        """Load all persistent state data"""
        self.current_state = self.load_repository_state()
        self._load_metrics()


    def _load_metrics(self):
        """Load update metrics"""
        try:
            self.metrics = self.repository.load_metrics(self.project_id)
            logger.debug("Metrics loaded", success_rate=self.metrics.success_rate)
        except Exception as e:
            logger.error("Failed to load metrics", error=str(e))


    def _cleanup_old_snapshots(self):
        """Clean up old snapshots beyond retention period"""
        cutoff_time = time.time() - (self.config.backup_retention_days * 24 * 3600)
        # Delete from database
        self.repository.delete_snapshots_before(cutoff_time, self.project_id)
        # Clear in-memory
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

    def _migrate_from_json(self):
        """Migrate existing JSON files to SQLite"""
        try:
            migrated_count = 0

            # 1. Migrate repository state
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                state = RepositoryState(
                    commit_hash=state_data["commit_hash"],
                    branch=state_data["branch"],
                    timestamp=state_data["timestamp"],
                    file_hashes=state_data["file_hashes"],
                    total_files=state_data.get("total_files", len(state_data["file_hashes"]))
                )

                if self.repository.save_repository_state(state, self.project_id):
                    # Backup JSON file
                    backup_path = self.state_file.with_suffix('.json.backup')
                    shutil.move(str(self.state_file), str(backup_path))
                    migrated_count += 1
                    logger.info("Repository state migrated from JSON", backup=str(backup_path))

            # 2. Migrate snapshots
            if self.snapshots_file.exists():
                with open(self.snapshots_file, 'r', encoding='utf-8') as f:
                    snapshots_data = json.load(f)

                for data in snapshots_data:
                    repo_state = RepositoryState(
                        commit_hash=data["repo_state"]["commit_hash"],
                        branch=data["repo_state"]["branch"],
                        timestamp=data["repo_state"]["timestamp"],
                        file_hashes={},  # Not migrated to save space
                        total_files=data["repo_state"]["total_files"]
                    )

                    snapshot = StateSnapshot(
                        snapshot_id=data["snapshot_id"],
                        timestamp=data["timestamp"],
                        repo_state=repo_state,
                        total_chunks=data["total_chunks"],
                        metadata=data.get("metadata", {})
                    )

                    self.repository.save_snapshot(snapshot, self.project_id)
                    migrated_count += 1

                # Backup JSON file
                backup_path = self.snapshots_file.with_suffix('.json.backup')
                shutil.move(str(self.snapshots_file), str(backup_path))
                logger.info("Snapshots migrated from JSON", count=len(snapshots_data), backup=str(backup_path))

            # 3. Migrate metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)

                metrics = UpdateMetrics(
                    total_update_requests=metrics_data.get("total_update_requests", 0),
                    successful_updates=metrics_data.get("successful_updates", 0),
                    failed_updates=metrics_data.get("failed_updates", 0),
                    avg_processing_time=metrics_data.get("avg_processing_time", 0.0),
                    total_files_processed=metrics_data.get("total_files_processed", 0),
                    total_chunks_modified=metrics_data.get("total_chunks_modified", 0),
                    last_update_time=metrics_data.get("last_update_time")
                )

                if self.repository.save_metrics(metrics, self.project_id):
                    # Backup JSON file
                    backup_path = self.metrics_file.with_suffix('.json.backup')
                    shutil.move(str(self.metrics_file), str(backup_path))
                    migrated_count += 1
                    logger.info("Metrics migrated from JSON", backup=str(backup_path))

            # 4. Migrate update results from results/ directory
            results_dir = self.state_dir / "results"
            if results_dir.exists():
                result_files = list(results_dir.glob("*.json"))
                for result_file in result_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)

                        # Reconstruct UpdateResult (simplified, status only)
                        result = UpdateResult(
                            request_id=result_data.get("request_id", result_file.stem),
                            status=UpdateStatus(result_data.get("status", "completed")),
                            changes_detected=[],  # Not migrated
                            files_processed=result_data.get("files_processed", 0),
                            chunks_added=result_data.get("chunks_added", 0),
                            chunks_updated=result_data.get("chunks_updated", 0),
                            chunks_deleted=result_data.get("chunks_deleted", 0),
                            processing_time=result_data.get("processing_time", 0.0),
                            error_message=result_data.get("error_message"),
                            warnings=result_data.get("warnings", [])
                        )

                        self.repository.save_update_result(result, self.project_id)
                        migrated_count += 1
                    except Exception as e:
                        logger.warning("Failed to migrate result file", file=str(result_file), error=str(e))

                # Backup results directory
                if result_files:
                    backup_dir = self.state_dir / "results.backup"
                    shutil.move(str(results_dir), str(backup_dir))
                    logger.info("Update results migrated from JSON", count=len(result_files), backup=str(backup_dir))

            if migrated_count > 0:
                logger.info("JSON to SQLite migration completed", migrated_items=migrated_count)

        except Exception as e:
            logger.error("Failed to migrate from JSON", error=str(e))
