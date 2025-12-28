"""
State repository for storing and managing update state in SQLite
"""
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import structlog

from .models import (
    RepositoryState, StateSnapshot, UpdateMetrics, UpdateResult, UpdateStatus
)


logger = structlog.get_logger(__name__)


class StateRepository:
    """Repository for managing update state in SQLite"""

    def __init__(self, db_path: str = "./chroma_db/projects.db"):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. repository_states
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repository_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    commit_hash TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT 'main',
                    timestamp REAL NOT NULL,
                    total_files INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id)
                )
            """)

            # 2. file_hashes (1:N relationship)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    FOREIGN KEY (state_id) REFERENCES repository_states(id) ON DELETE CASCADE,
                    UNIQUE(state_id, file_path)
                )
            """)

            # Index for file_hashes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hashes_state_id
                ON file_hashes(state_id)
            """)

            # 3. state_snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT UNIQUE NOT NULL,
                    project_id TEXT,
                    timestamp REAL NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    metadata TEXT,
                    commit_hash TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    repo_timestamp REAL NOT NULL,
                    total_files INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # Index for snapshots
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON state_snapshots(timestamp)
            """)

            # 4. update_metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS update_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE,
                    total_update_requests INTEGER DEFAULT 0,
                    successful_updates INTEGER DEFAULT 0,
                    failed_updates INTEGER DEFAULT 0,
                    avg_processing_time REAL DEFAULT 0.0,
                    total_files_processed INTEGER DEFAULT 0,
                    total_chunks_modified INTEGER DEFAULT 0,
                    last_update_time REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 5. update_results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS update_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    project_id TEXT,
                    status TEXT NOT NULL,
                    total_changes INTEGER DEFAULT 0,
                    files_processed INTEGER DEFAULT 0,
                    chunks_added INTEGER DEFAULT 0,
                    chunks_updated INTEGER DEFAULT 0,
                    chunks_deleted INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 0.0,
                    error_message TEXT,
                    warnings TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Index for update_results
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_update_results_created_at
                ON update_results(created_at)
            """)

            conn.commit()
            conn.close()
            logger.info("State database initialized", db_path=self.db_path)

        except Exception as e:
            logger.error("Failed to initialize state database", error=str(e))
            raise

    # ========== Repository State Methods ==========

    def save_repository_state(self, state: RepositoryState, project_id: Optional[str] = None) -> bool:
        """Save or update repository state"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            # Check if state exists
            cursor.execute("""
                SELECT id FROM repository_states WHERE project_id = ?
            """, (project_id,))
            existing = cursor.fetchone()

            if existing:
                state_id = existing[0]
                # Update existing state
                cursor.execute("""
                    UPDATE repository_states SET
                        commit_hash = ?,
                        branch = ?,
                        timestamp = ?,
                        total_files = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    state.commit_hash,
                    state.branch,
                    state.timestamp,
                    state.total_files,
                    now,
                    state_id
                ))

                # Delete old file hashes
                cursor.execute("""
                    DELETE FROM file_hashes WHERE state_id = ?
                """, (state_id,))
            else:
                # Insert new state
                cursor.execute("""
                    INSERT INTO repository_states (
                        project_id, commit_hash, branch, timestamp, total_files, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    state.commit_hash,
                    state.branch,
                    state.timestamp,
                    state.total_files,
                    now,
                    now
                ))
                state_id = cursor.lastrowid

            # Insert file hashes
            if state.file_hashes:
                for file_path, file_hash in state.file_hashes.items():
                    cursor.execute("""
                        INSERT INTO file_hashes (state_id, file_path, file_hash)
                        VALUES (?, ?, ?)
                    """, (state_id, file_path, file_hash))

            conn.commit()
            conn.close()

            logger.debug("Repository state saved",
                         project_id=project_id,
                         commit=state.commit_hash[:8],
                         total_files=state.total_files)
            return True

        except Exception as e:
            logger.error("Failed to save repository state", error=str(e))
            return False

    def load_repository_state(self, project_id: Optional[str] = None) -> Optional[RepositoryState]:
        """Load repository state"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get state
            cursor.execute("""
                SELECT * FROM repository_states WHERE project_id = ?
            """, (project_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                logger.info("No saved repository state found", project_id=project_id)
                return None

            state_id = row["id"]

            # Get file hashes
            cursor.execute("""
                SELECT file_path, file_hash FROM file_hashes WHERE state_id = ?
            """, (state_id,))
            file_hash_rows = cursor.fetchall()

            file_hashes = {row["file_path"]: row["file_hash"] for row in file_hash_rows}

            conn.close()

            state = RepositoryState(
                commit_hash=row["commit_hash"],
                branch=row["branch"],
                timestamp=row["timestamp"],
                file_hashes=file_hashes,
                total_files=row["total_files"]
            )

            logger.info("Repository state loaded",
                        project_id=project_id,
                        commit=state.commit_hash[:8],
                        total_files=state.total_files)

            return state

        except Exception as e:
            logger.error("Failed to load repository state", error=str(e))
            return None

    # ========== State Snapshot Methods ==========

    def save_snapshot(self, snapshot: StateSnapshot, project_id: Optional[str] = None) -> bool:
        """Save a state snapshot"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO state_snapshots (
                    snapshot_id, project_id, timestamp, total_chunks, metadata,
                    commit_hash, branch, repo_timestamp, total_files, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id,
                project_id,
                snapshot.timestamp,
                snapshot.total_chunks,
                json.dumps(snapshot.metadata) if snapshot.metadata else None,
                snapshot.repo_state.commit_hash,
                snapshot.repo_state.branch,
                snapshot.repo_state.timestamp,
                snapshot.repo_state.total_files,
                now
            ))

            conn.commit()
            conn.close()

            logger.info("State snapshot saved",
                        snapshot_id=snapshot.snapshot_id,
                        total_chunks=snapshot.total_chunks)
            return True

        except Exception as e:
            logger.error("Failed to save snapshot", error=str(e))
            return False

    def get_latest_snapshot(self, project_id: Optional[str] = None) -> Optional[StateSnapshot]:
        """Get the most recent snapshot"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM state_snapshots
                WHERE project_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (project_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            return self._row_to_snapshot(row)

        except Exception as e:
            logger.error("Failed to get latest snapshot", error=str(e))
            return None

    def get_snapshots_since(self, since_timestamp: float, project_id: Optional[str] = None) -> List[StateSnapshot]:
        """Get snapshots since a specific timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM state_snapshots
                WHERE project_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (project_id, since_timestamp))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_snapshot(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get snapshots since timestamp", error=str(e))
            return []

    def delete_snapshots_before(self, cutoff_timestamp: float, project_id: Optional[str] = None) -> int:
        """Delete snapshots older than cutoff timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM state_snapshots
                WHERE project_id = ? AND timestamp < ?
            """, (project_id, cutoff_timestamp))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info("Old snapshots deleted", deleted_count=deleted)
            return deleted

        except Exception as e:
            logger.error("Failed to delete old snapshots", error=str(e))
            return 0

    def count_snapshots(self, project_id: Optional[str] = None) -> int:
        """Count total snapshots"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM state_snapshots WHERE project_id = ?
            """, (project_id,))

            count = cursor.fetchone()[0]
            conn.close()

            return count

        except Exception as e:
            logger.error("Failed to count snapshots", error=str(e))
            return 0

    # ========== Update Metrics Methods ==========

    def save_metrics(self, metrics: UpdateMetrics, project_id: Optional[str] = None) -> bool:
        """Save or update metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            # Check if metrics exist
            cursor.execute("""
                SELECT id FROM update_metrics WHERE project_id = ?
            """, (project_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE update_metrics SET
                        total_update_requests = ?,
                        successful_updates = ?,
                        failed_updates = ?,
                        avg_processing_time = ?,
                        total_files_processed = ?,
                        total_chunks_modified = ?,
                        last_update_time = ?,
                        updated_at = ?
                    WHERE project_id = ?
                """, (
                    metrics.total_update_requests,
                    metrics.successful_updates,
                    metrics.failed_updates,
                    metrics.avg_processing_time,
                    metrics.total_files_processed,
                    metrics.total_chunks_modified,
                    metrics.last_update_time,
                    now,
                    project_id
                ))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO update_metrics (
                        project_id, total_update_requests, successful_updates, failed_updates,
                        avg_processing_time, total_files_processed, total_chunks_modified,
                        last_update_time, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    metrics.total_update_requests,
                    metrics.successful_updates,
                    metrics.failed_updates,
                    metrics.avg_processing_time,
                    metrics.total_files_processed,
                    metrics.total_chunks_modified,
                    metrics.last_update_time,
                    now,
                    now
                ))

            conn.commit()
            conn.close()

            logger.debug("Metrics saved", project_id=project_id)
            return True

        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))
            return False

    def load_metrics(self, project_id: Optional[str] = None) -> UpdateMetrics:
        """Load metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM update_metrics WHERE project_id = ?
            """, (project_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                logger.debug("No metrics found, returning default", project_id=project_id)
                return UpdateMetrics()

            return UpdateMetrics(
                total_update_requests=row["total_update_requests"],
                successful_updates=row["successful_updates"],
                failed_updates=row["failed_updates"],
                avg_processing_time=row["avg_processing_time"],
                total_files_processed=row["total_files_processed"],
                total_chunks_modified=row["total_chunks_modified"],
                last_update_time=row["last_update_time"]
            )

        except Exception as e:
            logger.error("Failed to load metrics", error=str(e))
            return UpdateMetrics()

    # ========== Update Result Methods ==========

    def save_update_result(self, result: UpdateResult, project_id: Optional[str] = None) -> bool:
        """Save update result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO update_results (
                    request_id, project_id, status, total_changes, files_processed,
                    chunks_added, chunks_updated, chunks_deleted, processing_time,
                    success_rate, error_message, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.request_id,
                project_id,
                result.status.value,
                result.total_changes,
                result.files_processed,
                result.chunks_added,
                result.chunks_updated,
                result.chunks_deleted,
                result.processing_time,
                result.success_rate,
                result.error_message,
                json.dumps(result.warnings) if result.warnings else None,
                now
            ))

            conn.commit()
            conn.close()

            logger.info("Update result saved",
                        request_id=result.request_id,
                        status=result.status.value)
            return True

        except Exception as e:
            logger.error("Failed to save update result", error=str(e))
            return False

    def delete_results_before(self, cutoff_timestamp: float, project_id: Optional[str] = None) -> int:
        """Delete update results older than cutoff timestamp (created_at as ISO string)"""
        try:
            # Convert timestamp to ISO format for comparison
            cutoff_datetime = datetime.fromtimestamp(cutoff_timestamp).isoformat()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if project_id:
                cursor.execute("""
                    DELETE FROM update_results
                    WHERE project_id = ? AND created_at < ?
                """, (project_id, cutoff_datetime))
            else:
                cursor.execute("""
                    DELETE FROM update_results
                    WHERE created_at < ?
                """, (cutoff_datetime,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info("Old update results deleted", deleted_count=deleted)
            return deleted

        except Exception as e:
            logger.error("Failed to delete old update results", error=str(e))
            return 0

    # ========== Helper Methods ==========

    def reset_all(self) -> bool:
        """Delete all state data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete in order due to foreign key constraints
            cursor.execute("DELETE FROM file_hashes")
            cursor.execute("DELETE FROM repository_states")
            cursor.execute("DELETE FROM state_snapshots")
            cursor.execute("DELETE FROM update_metrics")
            cursor.execute("DELETE FROM update_results")

            conn.commit()
            conn.close()

            logger.warning("All state data deleted from database")
            return True

        except Exception as e:
            logger.error("Failed to reset state data", error=str(e))
            return False

    def _row_to_snapshot(self, row: sqlite3.Row) -> StateSnapshot:
        """Convert database row to StateSnapshot"""
        repo_state = RepositoryState(
            commit_hash=row["commit_hash"],
            branch=row["branch"],
            timestamp=row["repo_timestamp"],
            file_hashes={},  # Not loaded to save memory
            total_files=row["total_files"]
        )

        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return StateSnapshot(
            snapshot_id=row["snapshot_id"],
            timestamp=row["timestamp"],
            repo_state=repo_state,
            total_chunks=row["total_chunks"],
            metadata=metadata
        )
