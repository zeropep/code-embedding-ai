"""
Python AST Parser for code embedding pipeline.
Supports Django, Flask, FastAPI frameworks with decorator analysis.
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from .base_parser import BaseParser
from .models import ParsedFile, CodeChunk, CodeLanguage, LayerType, ParserConfig


class PythonParser(BaseParser):
    """Parser for Python source code using AST"""

    # Decorator patterns for framework detection
    DJANGO_DECORATORS = {
        'api_view', 'action', 'permission_classes', 'authentication_classes',
        'login_required', 'permission_required', 'csrf_exempt', 'require_http_methods',
        'receiver', 'admin.register', 'register'
    }

    FLASK_DECORATORS = {
        'route', 'get', 'post', 'put', 'delete', 'patch',
        'before_request', 'after_request', 'errorhandler'
    }

    FASTAPI_DECORATORS = {
        'get', 'post', 'put', 'delete', 'patch', 'options', 'head',
        'api_route', 'websocket', 'on_event'
    }

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.py', '.pyw']

    def get_language(self) -> CodeLanguage:
        return CodeLanguage.PYTHON

    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Parse Python file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file stats
            total_lines, last_modified = self.get_file_stats(file_path)
            file_hash = self.calculate_file_hash(file_path)

            # Parse Python AST
            chunks = self._parse_python_ast(content, str(file_path))

            # If AST parsing fails or returns no chunks, fall back to simple chunking
            if not chunks:
                chunks = self.chunk_content(content, str(file_path), self.get_language())

            return ParsedFile(
                file_path=str(file_path),
                language=self.get_language(),
                chunks=chunks,
                total_lines=total_lines,
                file_hash=file_hash,
                last_modified=last_modified
            )

        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
            return None

    def _parse_python_ast(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python using AST to extract classes, functions, and methods"""
        chunks = []
        lines = content.split('\n')

        try:
            tree = ast.parse(content)

            # Extract imports for framework detection
            imports = self._extract_imports(tree)
            framework = self._detect_framework(imports)

            # Determine layer type from file path
            file_layer = self._determine_layer_from_path(file_path)

            # Extract module-level docstring
            module_docstring = ast.get_docstring(tree)

            # Process top-level nodes
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class and its methods
                    class_chunks = self._extract_class_chunks(
                        node, lines, file_path, imports, framework, file_layer
                    )
                    chunks.extend(class_chunks)

                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Extract standalone function
                    func_chunk = self._extract_function_chunk(
                        node, lines, file_path, None, imports, framework, file_layer
                    )
                    if func_chunk:
                        chunks.append(func_chunk)

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            # Fall back to regex-based parsing
            chunks = self._parse_python_regex(content, file_path)

        return chunks

    def _extract_imports(self, tree: ast.Module) -> Dict[str, Set[str]]:
        """Extract import information from AST"""
        imports = {
            'modules': set(),
            'from_imports': {},  # module -> [names]
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['modules'].add(alias.name.split('.')[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    imports['modules'].add(module)
                    if node.module not in imports['from_imports']:
                        imports['from_imports'][node.module] = set()
                    for alias in node.names:
                        imports['from_imports'][node.module].add(alias.name)

        return imports

    def _detect_framework(self, imports: Dict[str, Set[str]]) -> Optional[str]:
        """Detect Python framework from imports"""
        modules = imports['modules']

        if 'django' in modules or 'rest_framework' in modules:
            return 'django'
        elif 'flask' in modules:
            return 'flask'
        elif 'fastapi' in modules:
            return 'fastapi'

        return None

    def _determine_layer_from_path(self, file_path: str) -> LayerType:
        """Determine layer type from file path patterns"""
        path_lower = file_path.lower().replace('\\', '/')
        filename = Path(file_path).stem.lower()

        # Django patterns
        if 'models.py' in path_lower or filename == 'models':
            return LayerType.ENTITY
        if 'views.py' in path_lower or filename == 'views':
            return LayerType.VIEW
        if 'serializers.py' in path_lower or filename == 'serializers':
            return LayerType.SERIALIZER
        if 'forms.py' in path_lower or filename == 'forms':
            return LayerType.FORM
        if 'admin.py' in path_lower or filename == 'admin':
            return LayerType.ADMIN
        if 'urls.py' in path_lower or filename == 'urls':
            return LayerType.ROUTER
        if 'middleware' in path_lower:
            return LayerType.MIDDLEWARE
        if 'signals.py' in path_lower or filename == 'signals':
            return LayerType.SIGNAL
        if 'tasks.py' in path_lower or filename == 'tasks':
            return LayerType.TASK
        if '/management/commands/' in path_lower:
            return LayerType.COMMAND
        if '/migrations/' in path_lower:
            return LayerType.MIGRATION

        # Flask/FastAPI patterns
        if 'routes.py' in path_lower or 'endpoints.py' in path_lower:
            return LayerType.CONTROLLER
        if 'schemas.py' in path_lower or filename == 'schemas':
            return LayerType.SCHEMA

        # Common patterns
        if 'config' in path_lower or 'settings' in path_lower:
            return LayerType.CONFIG
        if 'test' in path_lower or filename.startswith('test_'):
            return LayerType.TEST
        if 'utils' in path_lower or 'helpers' in path_lower:
            return LayerType.UTIL
        if 'services' in path_lower or filename.endswith('_service'):
            return LayerType.SERVICE
        if 'repository' in path_lower or 'repositories' in path_lower:
            return LayerType.REPOSITORY

        return LayerType.UNKNOWN

    def _extract_class_chunks(self, node: ast.ClassDef, lines: List[str],
                             file_path: str, imports: Dict, framework: Optional[str],
                             file_layer: LayerType) -> List[CodeChunk]:
        """Extract class and its methods as chunks"""
        chunks = []

        # Get class content
        start_line = node.lineno
        end_line = self._get_node_end_line(node, lines)
        class_content = '\n'.join(lines[start_line - 1:end_line])

        # Determine layer type from class
        layer_type = self._determine_class_layer(node, class_content, framework, file_layer)

        # Extract decorators
        decorators = self._extract_decorators(node)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get base classes
        bases = [self._get_base_name(base) for base in node.bases]

        # Create class-level chunk
        class_chunk = CodeChunk(
            content=class_content,
            file_path=file_path,
            language=self.get_language(),
            start_line=start_line,
            end_line=end_line,
            class_name=node.name,
            layer_type=layer_type,
            metadata={
                'type': 'class',
                'decorators': decorators,
                'bases': bases,
                'docstring': docstring[:200] if docstring else None,
                'framework': framework,
                'is_async': False
            }
        )
        chunks.append(class_chunk)

        # Extract methods if class is large enough
        if self.config.extract_methods and len(lines[start_line - 1:end_line]) > 20:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = self._extract_function_chunk(
                        item, lines, file_path, node.name, imports, framework, layer_type
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

        return chunks

    def _extract_function_chunk(self, node, lines: List[str],
                               file_path: str, class_name: Optional[str],
                               imports: Dict, framework: Optional[str],
                               parent_layer: LayerType) -> Optional[CodeChunk]:
        """Extract function/method as a chunk"""
        start_line = node.lineno
        end_line = self._get_node_end_line(node, lines)

        # Skip very small functions
        if end_line - start_line < 2:
            return None

        content = '\n'.join(lines[start_line - 1:end_line])

        # Check token count
        token_count = len(content) // 4
        if token_count < self.config.min_tokens:
            return None

        # Extract decorators
        decorators = self._extract_decorators(node)

        # Determine layer from decorators
        layer_type = self._determine_function_layer(decorators, framework, parent_layer)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Extract function signature info
        args = self._extract_arguments(node.args)
        returns = self._get_return_annotation(node)

        is_async = isinstance(node, ast.AsyncFunctionDef)

        return CodeChunk(
            content=content,
            file_path=file_path,
            language=self.get_language(),
            start_line=start_line,
            end_line=end_line,
            function_name=node.name,
            class_name=class_name,
            layer_type=layer_type,
            token_count=token_count,
            metadata={
                'type': 'method' if class_name else 'function',
                'decorators': decorators,
                'arguments': args,
                'returns': returns,
                'docstring': docstring[:200] if docstring else None,
                'is_async': is_async,
                'is_private': node.name.startswith('_'),
                'is_dunder': node.name.startswith('__') and node.name.endswith('__'),
                'framework': framework
            }
        )

    def _extract_decorators(self, node) -> List[str]:
        """Extract decorator names from a node"""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{self._get_attribute_name(dec)}")
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attribute_name(dec.func))
        return decorators

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'app.route')"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))

    def _get_base_name(self, node) -> str:
        """Get base class name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            # Handle Generic types like List[str]
            if isinstance(node.value, ast.Name):
                return node.value.id
            elif isinstance(node.value, ast.Attribute):
                return self._get_attribute_name(node.value)
        return str(node)

    def _extract_arguments(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Extract function arguments with type hints"""
        arguments = []

        all_args = args.args + args.posonlyargs + args.kwonlyargs
        for arg in all_args:
            arg_info = {'name': arg.arg}
            if arg.annotation:
                arg_info['type'] = self._annotation_to_string(arg.annotation)
            arguments.append(arg_info)

        if args.vararg:
            arguments.append({'name': f'*{args.vararg.arg}', 'type': 'vararg'})
        if args.kwarg:
            arguments.append({'name': f'**{args.kwarg.arg}', 'type': 'kwarg'})

        return arguments

    def _get_return_annotation(self, node) -> Optional[str]:
        """Get return type annotation"""
        if node.returns:
            return self._annotation_to_string(node.returns)
        return None

    def _annotation_to_string(self, node) -> str:
        """Convert annotation AST node to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            base = self._annotation_to_string(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ', '.join(self._annotation_to_string(elt) for elt in node.slice.elts)
            else:
                args = self._annotation_to_string(node.slice)
            return f"{base}[{args}]"
        return 'Any'

    def _get_node_end_line(self, node, lines: List[str]) -> int:
        """Get the end line of an AST node"""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        # Fallback: find end by indentation
        return self._find_block_end(lines, node.lineno - 1)

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a Python block by indentation"""
        if start_idx >= len(lines):
            return start_idx + 1

        start_line = lines[start_idx]
        start_indent = len(start_line) - len(start_line.lstrip())

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            current_indent = len(line) - len(line.lstrip())

            # If we find a line with same or less indentation, previous line was end
            if current_indent <= start_indent and stripped:
                return i

        return len(lines)

    def _determine_class_layer(self, node: ast.ClassDef, content: str,
                              framework: Optional[str], file_layer: LayerType) -> LayerType:
        """Determine layer type for a class"""
        class_name = node.name.lower()
        content_lower = content.lower()
        decorators = self._extract_decorators(node)
        bases = [self._get_base_name(base).lower() for base in node.bases]

        # Django patterns
        if framework == 'django':
            if 'models.model' in bases or 'model' in bases:
                return LayerType.ENTITY
            if any(b in bases for b in ['serializer', 'modelserializer']):
                return LayerType.SERIALIZER
            if any(b in bases for b in ['viewset', 'apiview', 'genericapiview']):
                return LayerType.VIEW
            if any(b in bases for b in ['form', 'modelform']):
                return LayerType.FORM
            if any(b in bases for b in ['modeladmin', 'admin']):
                return LayerType.ADMIN
            if 'middleware' in class_name:
                return LayerType.MIDDLEWARE

        # Flask patterns
        elif framework == 'flask':
            if 'resource' in bases or 'methodview' in bases:
                return LayerType.CONTROLLER

        # FastAPI patterns
        elif framework == 'fastapi':
            if 'basemodel' in bases:
                return LayerType.SCHEMA

        # Common patterns
        if 'test' in class_name or any('test' in d.lower() for d in decorators):
            return LayerType.TEST
        if 'service' in class_name:
            return LayerType.SERVICE
        if 'repository' in class_name or 'dao' in class_name:
            return LayerType.REPOSITORY
        if 'config' in class_name or 'settings' in class_name:
            return LayerType.CONFIG
        if 'util' in class_name or 'helper' in class_name:
            return LayerType.UTIL

        # Use file-level layer as fallback
        if file_layer != LayerType.UNKNOWN:
            return file_layer

        return LayerType.UNKNOWN

    def _determine_function_layer(self, decorators: List[str],
                                  framework: Optional[str],
                                  parent_layer: LayerType) -> LayerType:
        """Determine layer type for a function based on decorators"""
        decorator_set = set(d.lower() for d in decorators)

        if framework == 'django':
            if decorator_set & {'api_view', 'action'}:
                return LayerType.VIEW
            if 'receiver' in decorator_set:
                return LayerType.SIGNAL
            if 'task' in decorator_set or 'shared_task' in decorator_set:
                return LayerType.TASK

        elif framework == 'flask':
            if any(d in decorator_set for d in ['route', 'get', 'post', 'put', 'delete']):
                return LayerType.CONTROLLER

        elif framework == 'fastapi':
            if any(d in decorator_set for d in ['get', 'post', 'put', 'delete', 'patch', 'api_route']):
                return LayerType.CONTROLLER

        return parent_layer

    def _parse_python_regex(self, content: str, file_path: str) -> List[CodeChunk]:
        """Fallback regex-based parsing when AST fails"""
        chunks = []
        lines = content.split('\n')

        # Patterns for class and function detection
        class_pattern = r'^class\s+(\w+)'
        func_pattern = r'^(async\s+)?def\s+(\w+)'

        i = 0
        current_class = None

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                i += 1
                continue

            # Check for class
            class_match = re.match(class_pattern, stripped)
            if class_match:
                current_class = class_match.group(1)
                end_idx = self._find_block_end(lines, i)
                chunk_content = '\n'.join(lines[i:end_idx])

                token_count = len(chunk_content) // 4
                if token_count >= self.config.min_tokens:
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=self.get_language(),
                        start_line=i + 1,
                        end_line=end_idx,
                        class_name=current_class,
                        layer_type=self._determine_layer_from_path(file_path),
                        token_count=token_count
                    )
                    chunks.append(chunk)

                i = end_idx
                continue

            # Check for function
            func_match = re.match(func_pattern, stripped)
            if func_match:
                is_async = func_match.group(1) is not None
                func_name = func_match.group(2)
                end_idx = self._find_block_end(lines, i)
                chunk_content = '\n'.join(lines[i:end_idx])

                token_count = len(chunk_content) // 4
                if token_count >= self.config.min_tokens:
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=self.get_language(),
                        start_line=i + 1,
                        end_line=end_idx,
                        function_name=func_name,
                        class_name=current_class,
                        layer_type=self._determine_layer_from_path(file_path),
                        token_count=token_count,
                        metadata={'is_async': is_async}
                    )
                    chunks.append(chunk)

                i = end_idx
                continue

            i += 1

        return chunks
