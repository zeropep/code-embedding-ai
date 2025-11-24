import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
import structlog

from .models import (RepositoryState, StateSnapshot, UpdateConfig,
                     UpdateMetrics, UpdateResult, UpdateStatus)


logger = structlog.get_logger(__name__)


class StateManager:
    """Manages repository state persistence and tracking"""

    def __init__(self, state_dir: str, config: UpdateConfig = None):
        self.state_dir = Path(state_dir)
        self.config = config or UpdateConfig()
        self.state_file = self.state_dir / "repository_state.json"
        self.snapshots_file = self.state_dir / "state_snapshots.json"
        self.metrics_file = self.state_dir / "update_metrics.json"

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.current_state: Optional[RepositoryState] = None
        self.snapshots: List[StateSnapshot] = []
        self.metrics: UpdateMetrics = UpdateMetrics()

        # Load existing state
        self._load_state()

        logger.info("StateManager initialized", state_dir=str(self.state_dir))

    def save_repository_state(self, state: RepositoryState) -> bool:
        """Save current repository state"""
        try:
            state_data = {
                "commit_hash": state.commit_hash,
                "branch": state.branch,
                "timestamp": state.timestamp,
                "file_hashes": state.file_hashes,
                "total_files": state.total_files
            }

            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            self.current_state = state

            logger.debug("Repository state saved",
                         commit=state.commit_hash[:8],
                         branch=state.branch,
                         total_files=state.total_files)

            return True

        except Exception as e:
            logger.error("Failed to save repository state", error=str(e))
            return False

    def load_repository_state(self) -> Optional[RepositoryState]:
        """Load saved repository state"""
        try:
            if not self.state_file.exists():
                logger.info("No saved repository state found")
                return None

            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            state = RepositoryState(
                commit_hash=state_data["commit_hash"],
                branch=state_data["branch"],
                timestamp=state_data["timestamp"],
                file_hashes=state_data["file_hashes"],
                total_files=state_data.get("total_files", len(state_data["file_hashes"]))
            )

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

        self.snapshots.append(snapshot)

        # Keep only recent snapshots (configurable retention)
        self._cleanup_old_snapshots()

        # Save snapshots
        self._save_snapshots()

        logger.info("State snapshot created",
                    snapshot_id=snapshot.snapshot_id,
                    total_chunks=total_chunks)

        return snapshot

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        """Get the most recent state snapshot"""
        if not self.snapshots:
            return None

        return max(self.snapshots, key=lambda s: s.timestamp)

    def get_snapshots_since(self, since_timestamp: float) -> List[StateSnapshot]:
        """Get snapshots since a specific timestamp"""
        return [s for s in self.snapshots if s.timestamp >= since_timestamp]

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

            # Save metrics
            self._save_metrics()

            # Save individual result (optional detailed logging)
            self._save_update_result_detail(result)

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
            self._save_metrics()
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
        summary = {
            "has_current_state": self.current_state is not None,
            "total_snapshots": len(self.snapshots),
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

        # Remove old snapshots
        old_snapshots = [s for s in self.snapshots if s.timestamp < cutoff_time]
        for snapshot in old_snapshots:
            self.snapshots.remove(snapshot)
            removed_count += 1

        if removed_count > 0:
            self._save_snapshots()
            logger.info("Cleaned up old snapshots",
                        removed_count=removed_count,
                        retention_days=retention_days)

        # Clean up old result files
        results_dir = self.state_dir / "results"
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                try:
                    file_age = time.time() - result_file.stat().st_mtime
                    if file_age > (retention_days * 24 * 3600):
                        result_file.unlink()
                        removed_count += 1
                except Exception as e:
                    logger.warning("Failed to remove old result file",
                                   file=str(result_file), error=str(e))

        return removed_count

    def _load_state(self):
        """Load all persistent state data"""
        self.current_state = self.load_repository_state()
        self._load_snapshots()
        self._load_metrics()

    def _save_snapshots(self) -> bool:
        """Save state snapshots"""
        try:
            snapshots_data = []
            for snapshot in self.snapshots:
                snapshot_data = {
                    "snapshot_id": snapshot.snapshot_id,
                    "timestamp": snapshot.timestamp,
                    "total_chunks": snapshot.total_chunks,
                    "metadata": snapshot.metadata,
                    "repo_state": {
                        "commit_hash": snapshot.repo_state.commit_hash,
                        "branch": snapshot.repo_state.branch,
                        "timestamp": snapshot.repo_state.timestamp,
                        "total_files": snapshot.repo_state.total_files
                        # Note: Not saving file_hashes to reduce size
                    }
                }
                snapshots_data.append(snapshot_data)

            with open(self.snapshots_file, 'w', encoding='utf-8') as f:
                json.dump(snapshots_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error("Failed to save snapshots", error=str(e))
            return False

    def _load_snapshots(self):
        """Load state snapshots"""
        try:
            if not self.snapshots_file.exists():
                return

            with open(self.snapshots_file, 'r', encoding='utf-8') as f:
                snapshots_data = json.load(f)

            self.snapshots = []
            for data in snapshots_data:
                repo_state = RepositoryState(
                    commit_hash=data["repo_state"]["commit_hash"],
                    branch=data["repo_state"]["branch"],
                    timestamp=data["repo_state"]["timestamp"],
                    file_hashes={},  # Not loaded to save memory
                    total_files=data["repo_state"]["total_files"]
                )

                snapshot = StateSnapshot(
                    snapshot_id=data["snapshot_id"],
                    timestamp=data["timestamp"],
                    repo_state=repo_state,
                    total_chunks=data["total_chunks"],
                    metadata=data.get("metadata", {})
                )
                self.snapshots.append(snapshot)

            logger.debug("Snapshots loaded", count=len(self.snapshots))

        except Exception as e:
            logger.error("Failed to load snapshots", error=str(e))

    def _save_metrics(self) -> bool:
        """Save update metrics"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))
            return False

    def _load_metrics(self):
        """Load update metrics"""
        try:
            if not self.metrics_file.exists():
                return

            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)

            self.metrics = UpdateMetrics(
                total_update_requests=metrics_data.get("total_update_requests", 0),
                successful_updates=metrics_data.get("successful_updates", 0),
                failed_updates=metrics_data.get("failed_updates", 0),
                avg_processing_time=metrics_data.get("avg_processing_time", 0.0),
                total_files_processed=metrics_data.get("total_files_processed", 0),
                total_chunks_modified=metrics_data.get("total_chunks_modified", 0),
                last_update_time=metrics_data.get("last_update_time")
            )

            logger.debug("Metrics loaded", success_rate=self.metrics.success_rate)

        except Exception as e:
            logger.error("Failed to load metrics", error=str(e))

    def _save_update_result_detail(self, result: UpdateResult):
        """Save detailed update result for debugging"""
        try:
            results_dir = self.state_dir / "results"
            results_dir.mkdir(exist_ok=True)

            result_file = results_dir / f"{result.request_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning("Failed to save detailed result", error=str(e))

    def _cleanup_old_snapshots(self):
        """Clean up old snapshots beyond retention period"""
        cutoff_time = time.time() - (self.config.backup_retention_days * 24 * 3600)
        self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
