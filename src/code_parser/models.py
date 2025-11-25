from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class CodeLanguage(Enum):
    JAVA = "java"
    KOTLIN = "kotlin"
    HTML = "html"
    XML = "xml"
    YAML = "yaml"
    PROPERTIES = "properties"
    PYTHON = "python"


class LayerType(Enum):
    # Common layers
    CONTROLLER = "Controller"
    SERVICE = "Service"
    REPOSITORY = "Repository"
    ENTITY = "Entity"
    CONFIG = "Config"
    UTIL = "Util"
    TEST = "Test"
    TEMPLATE = "Template"
    UNKNOWN = "Unknown"
    # Python-specific layers
    VIEW = "View"                    # Django views
    MIDDLEWARE = "Middleware"        # Django/Flask middleware
    SERIALIZER = "Serializer"        # DRF serializers
    ROUTER = "Router"                # FastAPI routers
    SCHEMA = "Schema"                # Pydantic models/schemas
    FORM = "Form"                    # Django forms
    ADMIN = "Admin"                  # Django admin
    COMMAND = "Command"              # Django management commands
    TASK = "Task"                    # Celery tasks
    SIGNAL = "Signal"                # Django signals
    MIGRATION = "Migration"          # Database migrations


@dataclass
class CodeChunk:
    """Represents a parsed code chunk with metadata"""
    content: str
    file_path: str
    language: CodeLanguage
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    layer_type: LayerType = LayerType.UNKNOWN
    token_count: int = 0
    metadata: Dict[str, Any] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.token_count == 0:
            self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(self.content) // 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "language": self.language.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "layer_type": self.layer_type.value,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "project_id": self.project_id,
            "project_name": self.project_name
        }


@dataclass
class ParsedFile:
    """Represents a parsed source file"""
    file_path: str
    language: CodeLanguage
    chunks: List[CodeChunk]
    total_lines: int
    file_hash: str
    last_modified: float

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)


@dataclass
class ParserConfig:
    """Configuration for code parsing"""
    min_tokens: int = 50
    max_tokens: int = 500
    overlap_tokens: int = 20
    include_comments: bool = False
    extract_methods: bool = True
    extract_classes: bool = True
    extract_interfaces: bool = True
    supported_extensions: List[str] = None

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                ".java", ".kt", ".kts", ".html", ".htm",
                ".xml", ".yml", ".yaml", ".properties",
                ".py", ".pyw"  # Python files
            ]
