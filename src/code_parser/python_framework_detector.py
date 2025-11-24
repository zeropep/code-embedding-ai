"""
Python Framework Detector for Django, Flask, and FastAPI.
Detects framework-specific patterns and layer types.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from .models import LayerType


class PythonFramework(Enum):
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    UNKNOWN = "unknown"


@dataclass
class FrameworkDetectionResult:
    """Result of framework detection"""
    framework: PythonFramework
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    layer_type: LayerType
    metadata: Dict


class PythonFrameworkDetector:
    """Detects Python frameworks and layer types from code analysis"""

    # Django-specific patterns
    DJANGO_IMPORTS = {
        'django', 'rest_framework', 'django.db', 'django.views',
        'django.contrib', 'django.http', 'django.forms', 'django.admin'
    }

    DJANGO_BASE_CLASSES = {
        # Models
        'models.Model': LayerType.ENTITY,
        'Model': LayerType.ENTITY,
        # Views
        'View': LayerType.VIEW,
        'TemplateView': LayerType.VIEW,
        'ListView': LayerType.VIEW,
        'DetailView': LayerType.VIEW,
        'CreateView': LayerType.VIEW,
        'UpdateView': LayerType.VIEW,
        'DeleteView': LayerType.VIEW,
        'FormView': LayerType.VIEW,
        # DRF Views
        'APIView': LayerType.VIEW,
        'ViewSet': LayerType.VIEW,
        'ModelViewSet': LayerType.VIEW,
        'GenericViewSet': LayerType.VIEW,
        'ReadOnlyModelViewSet': LayerType.VIEW,
        'GenericAPIView': LayerType.VIEW,
        # Serializers
        'Serializer': LayerType.SERIALIZER,
        'ModelSerializer': LayerType.SERIALIZER,
        'HyperlinkedModelSerializer': LayerType.SERIALIZER,
        # Forms
        'Form': LayerType.FORM,
        'ModelForm': LayerType.FORM,
        # Admin
        'ModelAdmin': LayerType.ADMIN,
        'TabularInline': LayerType.ADMIN,
        'StackedInline': LayerType.ADMIN,
        # Middleware
        'MiddlewareMixin': LayerType.MIDDLEWARE,
    }

    DJANGO_DECORATORS = {
        'api_view': LayerType.VIEW,
        'action': LayerType.VIEW,
        'permission_classes': LayerType.VIEW,
        'authentication_classes': LayerType.VIEW,
        'login_required': LayerType.VIEW,
        'permission_required': LayerType.VIEW,
        'csrf_exempt': LayerType.VIEW,
        'require_http_methods': LayerType.VIEW,
        'require_GET': LayerType.VIEW,
        'require_POST': LayerType.VIEW,
        'receiver': LayerType.SIGNAL,
        'admin.register': LayerType.ADMIN,
        'register': LayerType.ADMIN,
        'shared_task': LayerType.TASK,
        'task': LayerType.TASK,
    }

    # Flask-specific patterns
    FLASK_IMPORTS = {'flask', 'flask_restful', 'flask_login', 'flask_sqlalchemy'}

    FLASK_BASE_CLASSES = {
        'Resource': LayerType.CONTROLLER,
        'MethodView': LayerType.CONTROLLER,
        'FlaskForm': LayerType.FORM,
        'db.Model': LayerType.ENTITY,
    }

    FLASK_DECORATORS = {
        'route': LayerType.CONTROLLER,
        'get': LayerType.CONTROLLER,
        'post': LayerType.CONTROLLER,
        'put': LayerType.CONTROLLER,
        'delete': LayerType.CONTROLLER,
        'patch': LayerType.CONTROLLER,
        'before_request': LayerType.MIDDLEWARE,
        'after_request': LayerType.MIDDLEWARE,
        'before_first_request': LayerType.MIDDLEWARE,
        'errorhandler': LayerType.CONTROLLER,
        'login_required': LayerType.VIEW,
    }

    # FastAPI-specific patterns
    FASTAPI_IMPORTS = {'fastapi', 'pydantic', 'starlette'}

    FASTAPI_BASE_CLASSES = {
        'BaseModel': LayerType.SCHEMA,
        'BaseSettings': LayerType.CONFIG,
    }

    FASTAPI_DECORATORS = {
        'get': LayerType.CONTROLLER,
        'post': LayerType.CONTROLLER,
        'put': LayerType.CONTROLLER,
        'delete': LayerType.CONTROLLER,
        'patch': LayerType.CONTROLLER,
        'options': LayerType.CONTROLLER,
        'head': LayerType.CONTROLLER,
        'api_route': LayerType.CONTROLLER,
        'websocket': LayerType.CONTROLLER,
        'on_event': LayerType.MIDDLEWARE,
        'middleware': LayerType.MIDDLEWARE,
        'Depends': LayerType.SERVICE,
    }

    # File path patterns
    DJANGO_PATH_PATTERNS = {
        'models.py': LayerType.ENTITY,
        'views.py': LayerType.VIEW,
        'serializers.py': LayerType.SERIALIZER,
        'forms.py': LayerType.FORM,
        'admin.py': LayerType.ADMIN,
        'urls.py': LayerType.ROUTER,
        'signals.py': LayerType.SIGNAL,
        'tasks.py': LayerType.TASK,
        'middleware.py': LayerType.MIDDLEWARE,
        'management/commands': LayerType.COMMAND,
        'migrations': LayerType.MIGRATION,
        'tests.py': LayerType.TEST,
        'test_': LayerType.TEST,
    }

    def __init__(self):
        self._project_cache: Dict[str, PythonFramework] = {}

    def detect_framework_from_imports(self, imports: Dict[str, Set[str]]) -> Tuple[PythonFramework, float]:
        """Detect framework from import statements"""
        modules = imports.get('modules', set())

        django_score = len(modules & self.DJANGO_IMPORTS) / len(self.DJANGO_IMPORTS)
        flask_score = len(modules & self.FLASK_IMPORTS) / len(self.FLASK_IMPORTS)
        fastapi_score = len(modules & self.FASTAPI_IMPORTS) / len(self.FASTAPI_IMPORTS)

        if django_score > 0 and django_score >= max(flask_score, fastapi_score):
            return PythonFramework.DJANGO, min(django_score * 2, 1.0)
        elif flask_score > 0 and flask_score >= fastapi_score:
            return PythonFramework.FLASK, min(flask_score * 2, 1.0)
        elif fastapi_score > 0:
            return PythonFramework.FASTAPI, min(fastapi_score * 2, 1.0)

        return PythonFramework.UNKNOWN, 0.0

    def detect_framework_from_project(self, project_path: Path) -> PythonFramework:
        """Detect framework from project structure"""
        if str(project_path) in self._project_cache:
            return self._project_cache[str(project_path)]

        indicators = {
            PythonFramework.DJANGO: 0,
            PythonFramework.FLASK: 0,
            PythonFramework.FASTAPI: 0,
        }

        # Check for Django-specific files
        django_files = ['manage.py', 'settings.py', 'wsgi.py', 'asgi.py']
        for f in django_files:
            if list(project_path.glob(f'**/{f}')):
                indicators[PythonFramework.DJANGO] += 1

        # Check for Flask-specific patterns
        if list(project_path.glob('**/app.py')) or list(project_path.glob('**/application.py')):
            indicators[PythonFramework.FLASK] += 1

        # Check for FastAPI-specific patterns
        if list(project_path.glob('**/main.py')):
            indicators[PythonFramework.FASTAPI] += 1

        # Check requirements.txt or pyproject.toml
        req_files = list(project_path.glob('requirements*.txt')) + list(project_path.glob('pyproject.toml'))
        for req_file in req_files:
            try:
                content = req_file.read_text().lower()
                if 'django' in content:
                    indicators[PythonFramework.DJANGO] += 2
                if 'flask' in content:
                    indicators[PythonFramework.FLASK] += 2
                if 'fastapi' in content:
                    indicators[PythonFramework.FASTAPI] += 2
            except Exception:
                pass

        # Determine framework
        max_score = max(indicators.values())
        if max_score == 0:
            result = PythonFramework.UNKNOWN
        else:
            result = max(indicators, key=indicators.get)

        self._project_cache[str(project_path)] = result
        return result

    def detect_layer_from_class(self, class_name: str, base_classes: List[str],
                                decorators: List[str], framework: PythonFramework) -> LayerType:
        """Detect layer type from class definition"""
        # Check base classes
        base_class_layers = self._get_base_class_layers(framework)
        for base in base_classes:
            base_simple = base.split('.')[-1]
            if base in base_class_layers:
                return base_class_layers[base]
            if base_simple in base_class_layers:
                return base_class_layers[base_simple]

        # Check decorators
        decorator_layers = self._get_decorator_layers(framework)
        for dec in decorators:
            dec_simple = dec.split('.')[-1]
            if dec in decorator_layers:
                return decorator_layers[dec]
            if dec_simple in decorator_layers:
                return decorator_layers[dec_simple]

        # Check class name patterns
        class_lower = class_name.lower()
        name_patterns = {
            'service': LayerType.SERVICE,
            'repository': LayerType.REPOSITORY,
            'dao': LayerType.REPOSITORY,
            'controller': LayerType.CONTROLLER,
            'view': LayerType.VIEW,
            'serializer': LayerType.SERIALIZER,
            'form': LayerType.FORM,
            'model': LayerType.ENTITY,
            'schema': LayerType.SCHEMA,
            'middleware': LayerType.MIDDLEWARE,
            'admin': LayerType.ADMIN,
            'config': LayerType.CONFIG,
            'settings': LayerType.CONFIG,
            'test': LayerType.TEST,
            'util': LayerType.UTIL,
            'helper': LayerType.UTIL,
        }

        for pattern, layer in name_patterns.items():
            if pattern in class_lower:
                return layer

        return LayerType.UNKNOWN

    def detect_layer_from_function(self, func_name: str, decorators: List[str],
                                   framework: PythonFramework) -> LayerType:
        """Detect layer type from function definition"""
        decorator_layers = self._get_decorator_layers(framework)

        for dec in decorators:
            dec_simple = dec.split('.')[-1]
            if dec in decorator_layers:
                return decorator_layers[dec]
            if dec_simple in decorator_layers:
                return decorator_layers[dec_simple]

        # Check function name patterns
        func_lower = func_name.lower()
        if func_lower.startswith('test_') or func_lower.startswith('test'):
            return LayerType.TEST

        return LayerType.UNKNOWN

    def detect_layer_from_path(self, file_path: str, framework: PythonFramework) -> LayerType:
        """Detect layer type from file path"""
        path_lower = file_path.lower().replace('\\', '/')
        filename = Path(file_path).name.lower()

        # Check Django-specific path patterns
        for pattern, layer in self.DJANGO_PATH_PATTERNS.items():
            if pattern in path_lower or filename == pattern or filename.startswith(pattern):
                return layer

        # Common patterns
        common_patterns = {
            'services': LayerType.SERVICE,
            'repositories': LayerType.REPOSITORY,
            'controllers': LayerType.CONTROLLER,
            'routes': LayerType.ROUTER,
            'endpoints': LayerType.CONTROLLER,
            'schemas': LayerType.SCHEMA,
            'utils': LayerType.UTIL,
            'helpers': LayerType.UTIL,
            'config': LayerType.CONFIG,
            'settings': LayerType.CONFIG,
            'tests': LayerType.TEST,
        }

        for pattern, layer in common_patterns.items():
            if f'/{pattern}/' in path_lower or path_lower.endswith(f'/{pattern}'):
                return layer

        return LayerType.UNKNOWN

    def detect(self, file_path: str, content: str, imports: Dict[str, Set[str]],
               class_info: Optional[Dict] = None,
               function_info: Optional[Dict] = None) -> FrameworkDetectionResult:
        """
        Comprehensive detection of framework and layer type.

        Args:
            file_path: Path to the Python file
            content: File content
            imports: Extracted imports from AST
            class_info: Optional class information (name, bases, decorators)
            function_info: Optional function information (name, decorators)

        Returns:
            FrameworkDetectionResult with framework, confidence, and layer type
        """
        indicators = []

        # Detect framework from imports
        framework, confidence = self.detect_framework_from_imports(imports)
        if framework != PythonFramework.UNKNOWN:
            indicators.append(f"imports suggest {framework.value}")

        # Detect layer type
        layer_type = LayerType.UNKNOWN

        # Priority: class > function > path
        if class_info:
            layer_type = self.detect_layer_from_class(
                class_info.get('name', ''),
                class_info.get('bases', []),
                class_info.get('decorators', []),
                framework
            )
            if layer_type != LayerType.UNKNOWN:
                indicators.append(f"class pattern suggests {layer_type.value}")

        if layer_type == LayerType.UNKNOWN and function_info:
            layer_type = self.detect_layer_from_function(
                function_info.get('name', ''),
                function_info.get('decorators', []),
                framework
            )
            if layer_type != LayerType.UNKNOWN:
                indicators.append(f"function pattern suggests {layer_type.value}")

        if layer_type == LayerType.UNKNOWN:
            layer_type = self.detect_layer_from_path(file_path, framework)
            if layer_type != LayerType.UNKNOWN:
                indicators.append(f"path pattern suggests {layer_type.value}")

        return FrameworkDetectionResult(
            framework=framework,
            confidence=confidence,
            indicators=indicators,
            layer_type=layer_type,
            metadata={
                'file_path': file_path,
                'has_class': class_info is not None,
                'has_function': function_info is not None,
            }
        )

    def _get_base_class_layers(self, framework: PythonFramework) -> Dict[str, LayerType]:
        """Get base class to layer mapping for framework"""
        if framework == PythonFramework.DJANGO:
            return self.DJANGO_BASE_CLASSES
        elif framework == PythonFramework.FLASK:
            return self.FLASK_BASE_CLASSES
        elif framework == PythonFramework.FASTAPI:
            return self.FASTAPI_BASE_CLASSES
        return {}

    def _get_decorator_layers(self, framework: PythonFramework) -> Dict[str, LayerType]:
        """Get decorator to layer mapping for framework"""
        if framework == PythonFramework.DJANGO:
            return self.DJANGO_DECORATORS
        elif framework == PythonFramework.FLASK:
            return self.FLASK_DECORATORS
        elif framework == PythonFramework.FASTAPI:
            return self.FASTAPI_DECORATORS
        return {}

    def get_framework_info(self, framework: PythonFramework) -> Dict:
        """Get information about a framework"""
        info = {
            PythonFramework.DJANGO: {
                'name': 'Django',
                'description': 'High-level Python web framework',
                'typical_layers': ['Model', 'View', 'Serializer', 'Form', 'Admin', 'Signal', 'Task'],
                'file_patterns': ['models.py', 'views.py', 'serializers.py', 'forms.py', 'admin.py'],
            },
            PythonFramework.FLASK: {
                'name': 'Flask',
                'description': 'Lightweight WSGI web application framework',
                'typical_layers': ['Controller', 'Model', 'Form', 'Service'],
                'file_patterns': ['app.py', 'routes.py', 'models.py'],
            },
            PythonFramework.FASTAPI: {
                'name': 'FastAPI',
                'description': 'Modern, fast web framework for building APIs',
                'typical_layers': ['Controller', 'Schema', 'Service', 'Repository'],
                'file_patterns': ['main.py', 'routers/', 'schemas.py', 'models.py'],
            },
            PythonFramework.UNKNOWN: {
                'name': 'Unknown',
                'description': 'Framework not detected',
                'typical_layers': [],
                'file_patterns': [],
            },
        }
        return info.get(framework, info[PythonFramework.UNKNOWN])
