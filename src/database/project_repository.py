"""
Project repository for storing and managing project metadata
"""
import sqlite3
import os
from typing import List, Optional
from datetime import datetime
import structlog

from .project_models import Project, ProjectStatus


logger = structlog.get_logger(__name__)


class ProjectRepository:
    """Repository for managing project metadata in SQLite"""

    def __init__(self, db_path: str = "./chroma_db/projects.db"):
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    description TEXT,
                    git_remote_url TEXT,
                    git_branch TEXT DEFAULT 'main',
                    primary_language TEXT DEFAULT 'java',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    last_indexed_at TEXT
                )
            """)

            # Create index on status for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_status
                ON projects(status)
            """)

            conn.commit()
            conn.close()
            logger.info("Project database initialized", db_path=self.db_path)

        except Exception as e:
            logger.error("Failed to initialize project database", error=str(e))
            raise

    def create(self, project: Project) -> Project:
        """Create a new project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO projects (
                    project_id, name, repository_path, description, git_remote_url, git_branch, primary_language, status,
                    created_at, updated_at, total_chunks, total_files, last_indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project.project_id,
                project.name,
                project.repository_path,
                project.description,
                project.git_remote_url,
                project.git_branch,
                project.primary_language,
                project.status.value,
                project.created_at.isoformat(),
                project.updated_at.isoformat(),
                project.total_chunks,
                project.total_files,
                project.last_indexed_at.isoformat() if project.last_indexed_at else None
            ))

            conn.commit()
            conn.close()

            logger.info("Project created", project_id=project.project_id, name=project.name)
            return project

        except sqlite3.IntegrityError as e:
            logger.error("Project already exists", project_id=project.project_id)
            raise ValueError(f"Project with ID {project.project_id} already exists")
        except Exception as e:
            logger.error("Failed to create project", error=str(e))
            raise

    def get(self, project_id: str) -> Optional[Project]:
        """Get a project by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM projects WHERE project_id = ?
            """, (project_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_project(row)
            return None

        except Exception as e:
            logger.error("Failed to get project", project_id=project_id, error=str(e))
            raise

    def get_all(self, status: Optional[str] = None) -> List[Project]:
        """Get all projects, optionally filtered by status"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if status:
                cursor.execute("""
                    SELECT * FROM projects WHERE status = ?
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT * FROM projects
                    ORDER BY created_at DESC
                """)

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_project(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get all projects", error=str(e))
            raise

    def update(self, project_id: str, updates: dict) -> Optional[Project]:
        """Update a project"""
        try:
            # Get current project
            project = self.get(project_id)
            if not project:
                return None

            # Update fields
            if "name" in updates:
                project.name = updates["name"]
            if "repository_path" in updates:
                project.repository_path = updates["repository_path"]
            if "description" in updates:
                project.description = updates["description"]
            if "git_remote_url" in updates:
                project.git_remote_url = updates["git_remote_url"]
            if "git_branch" in updates:
                project.git_branch = updates["git_branch"]
            if "primary_language" in updates:
                project.primary_language = updates["primary_language"]
            if "status" in updates:
                project.status = ProjectStatus(updates["status"])
            if "total_chunks" in updates:
                project.total_chunks = updates["total_chunks"]
            if "total_files" in updates:
                project.total_files = updates["total_files"]
            if "last_indexed_at" in updates:
                project.last_indexed_at = updates["last_indexed_at"]

            project.updated_at = datetime.now()

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE projects SET
                    name = ?,
                    repository_path = ?,
                    description = ?,
                    git_remote_url = ?,
                    git_branch = ?,
                    primary_language = ?,
                    status = ?,
                    updated_at = ?,
                    total_chunks = ?,
                    total_files = ?,
                    last_indexed_at = ?
                WHERE project_id = ?
            """, (
                project.name,
                project.repository_path,
                project.description,
                project.git_remote_url,
                project.git_branch,
                project.primary_language,
                project.status.value,
                project.updated_at.isoformat(),
                project.total_chunks,
                project.total_files,
                project.last_indexed_at.isoformat() if project.last_indexed_at else None,
                project_id
            ))

            conn.commit()
            conn.close()

            logger.info("Project updated", project_id=project_id)
            return project

        except Exception as e:
            logger.error("Failed to update project", project_id=project_id, error=str(e))
            raise

    def delete(self, project_id: str) -> bool:
        """Delete a project"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM projects WHERE project_id = ?
            """, (project_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            if deleted:
                logger.info("Project deleted", project_id=project_id)
            else:
                logger.warning("Project not found for deletion", project_id=project_id)

            return deleted

        except Exception as e:
            logger.error("Failed to delete project", project_id=project_id, error=str(e))
            raise

    def exists(self, project_id: str) -> bool:
        """Check if a project exists"""
        return self.get(project_id) is not None

    def reset_all(self) -> bool:
        """Delete all projects from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM projects")
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.warning("All projects deleted from database", deleted_count=deleted_count)
            return True

        except Exception as e:
            logger.error("Failed to reset projects database", error=str(e))
            return False

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        """Convert database row to Project object"""
        return Project(
            project_id=row["project_id"],
            name=row["name"],
            repository_path=row["repository_path"],
            description=row["description"],
            git_remote_url=row["git_remote_url"] if "git_remote_url" in row.keys() else None,
            git_branch=row["git_branch"] if "git_branch" in row.keys() else "main",
            primary_language=row["primary_language"] if "primary_language" in row.keys() else "java",
            status=ProjectStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            total_chunks=row["total_chunks"],
            total_files=row["total_files"],
            last_indexed_at=datetime.fromisoformat(row["last_indexed_at"])
                if row["last_indexed_at"] else None
        )
