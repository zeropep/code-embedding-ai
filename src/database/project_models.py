"""
Project metadata models for multi-project support
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class ProjectStatus(Enum):
    """Project status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    INITIALIZING = "initializing"


@dataclass
class Project:
    """Project metadata"""
    project_id: str
    name: str
    repository_path: str
    description: Optional[str] = None
    git_remote_url: Optional[str] = None  # Remote Git repository URL
    git_branch: str = "main"              # Git branch to monitor
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Statistics (optional, calculated on-the-fly)
    total_chunks: int = 0
    total_files: int = 0
    last_indexed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "repository_path": self.repository_path,
            "description": self.description,
            "git_remote_url": self.git_remote_url,
            "git_branch": self.git_branch,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_chunks": self.total_chunks,
            "total_files": self.total_files,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Create from dictionary"""
        return cls(
            project_id=data["project_id"],
            name=data["name"],
            repository_path=data["repository_path"],
            description=data.get("description"),
            git_remote_url=data.get("git_remote_url"),
            git_branch=data.get("git_branch", "main"),
            status=ProjectStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            total_chunks=data.get("total_chunks", 0),
            total_files=data.get("total_files", 0),
            last_indexed_at=datetime.fromisoformat(data["last_indexed_at"])
                if data.get("last_indexed_at") else None
        )

    @staticmethod
    def generate_id() -> str:
        """Generate unique project ID"""
        return f"proj_{uuid.uuid4().hex[:12]}"


@dataclass
class ProjectCreateRequest:
    """Request model for creating a project"""
    name: str
    repository_path: str
    description: Optional[str] = None
    git_remote_url: Optional[str] = None
    git_branch: str = "main"


@dataclass
class ProjectUpdateRequest:
    """Request model for updating a project"""
    name: Optional[str] = None
    repository_path: Optional[str] = None
    description: Optional[str] = None
    git_remote_url: Optional[str] = None
    git_branch: Optional[str] = None
    status: Optional[str] = None
