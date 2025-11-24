"""
Python metadata extraction for enhanced code analysis.
Extracts rich metadata from Python code including type hints,
complexity metrics, and code style information.
"""

import ast
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionSignature:
    """Represents a Python function signature"""
    name: str
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    decorators: List[str]
    is_async: bool
    is_classmethod: bool
    is_staticmethod: bool
    is_property: bool
    docstring: Optional[str]


@dataclass
class ImportInfo:
    """Information about imports in a Python file"""
    standard_library: Set[str]
    third_party: Set[str]
    local: Set[str]
    from_imports: Dict[str, List[str]]


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    lines_of_comments: int
    blank_lines: int
    number_of_functions: int
    number_of_classes: int
    max_nesting_depth: int


class PythonMetadataExtractor:
    """Extracts rich metadata from Python code"""

    # Python standard library modules (Python 3.8+)
    STDLIB_MODULES = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
        'asyncore', 'atexit', 'base64', 'bdb', 'binascii', 'binhex',
        'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk',
        'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
        'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
        'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
        'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
        'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
        'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
        'heapq', 'hmac', 'html', 'http', 'imaplib', 'imghdr', 'imp',
        'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
        'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma',
        'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap',
        'modulefinder', 'multiprocessing', 'netrc', 'nis', 'nntplib',
        'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib',
        'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
        'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile',
        'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue',
        'quopri', 'random', 're', 'readline', 'reprlib', 'resource',
        'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
        'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
        'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl',
        'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
        'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
        'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
        'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
        'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
        'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu',
        'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
        'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
        'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo',
        # typing extensions
        'typing_extensions',
    }

    def __init__(self):
        pass

    def extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from Python code"""
        metadata = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
        }

        try:
            tree = ast.parse(content)

            # Extract various metadata
            metadata['imports'] = self._extract_import_info(tree, file_path)
            metadata['functions'] = self._extract_functions(tree)
            metadata['classes'] = self._extract_classes(tree)
            metadata['complexity'] = self._calculate_complexity(tree, content)
            metadata['module_docstring'] = ast.get_docstring(tree)
            metadata['type_hints'] = self._analyze_type_hints(tree)
            metadata['style_info'] = self._analyze_code_style(content)

        except SyntaxError as e:
            metadata['parse_error'] = str(e)

        return metadata

    def _extract_import_info(self, tree: ast.Module, file_path: str) -> Dict[str, Any]:
        """Extract and categorize imports"""
        imports = ImportInfo(
            standard_library=set(),
            third_party=set(),
            local=set(),
            from_imports={}
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    self._categorize_import(module, imports, file_path)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    self._categorize_import(module, imports, file_path)

                    # Track from imports
                    if node.module not in imports.from_imports:
                        imports.from_imports[node.module] = []
                    for alias in node.names:
                        imports.from_imports[node.module].append(alias.name)

        return {
            'standard_library': list(imports.standard_library),
            'third_party': list(imports.third_party),
            'local': list(imports.local),
            'from_imports': imports.from_imports
        }

    def _categorize_import(self, module: str, imports: ImportInfo, file_path: str):
        """Categorize a module as stdlib, third-party, or local"""
        if module in self.STDLIB_MODULES:
            imports.standard_library.add(module)
        elif module.startswith('.') or self._is_local_import(module, file_path):
            imports.local.add(module)
        else:
            imports.third_party.add(module)

    def _is_local_import(self, module: str, file_path: str) -> bool:
        """Check if import is from local project"""
        # Simple heuristic: if the module name appears in the file path
        file_parts = Path(file_path).parts
        return module in file_parts

    def _extract_functions(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract function signatures and metadata"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (they'll be extracted with classes)
                if self._is_method(node, tree):
                    continue

                sig = self._extract_function_signature(node)
                functions.append(self._signature_to_dict(sig))

        return functions

    def _is_method(self, func_node, tree: ast.Module) -> bool:
        """Check if function is a method inside a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False

    def _extract_function_signature(self, node) -> FunctionSignature:
        """Extract function signature details"""
        decorators = []
        is_classmethod = False
        is_staticmethod = False
        is_property = False

        for dec in node.decorator_list:
            dec_name = self._get_decorator_name(dec)
            decorators.append(dec_name)
            if dec_name == 'classmethod':
                is_classmethod = True
            elif dec_name == 'staticmethod':
                is_staticmethod = True
            elif dec_name == 'property':
                is_property = True

        parameters = self._extract_parameters(node.args)
        return_type = self._annotation_to_string(node.returns) if node.returns else None
        docstring = ast.get_docstring(node)

        return FunctionSignature(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
            docstring=docstring[:200] if docstring else None
        )

    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Extract parameter information including type hints"""
        params = []

        # Positional-only args (Python 3.8+)
        for arg in args.posonlyargs:
            params.append(self._extract_arg(arg, 'positional_only'))

        # Regular args
        for arg in args.args:
            params.append(self._extract_arg(arg, 'positional_or_keyword'))

        # *args
        if args.vararg:
            params.append({
                'name': f'*{args.vararg.arg}',
                'type': self._annotation_to_string(args.vararg.annotation) if args.vararg.annotation else None,
                'kind': 'var_positional'
            })

        # Keyword-only args
        for arg in args.kwonlyargs:
            params.append(self._extract_arg(arg, 'keyword_only'))

        # **kwargs
        if args.kwarg:
            params.append({
                'name': f'**{args.kwarg.arg}',
                'type': self._annotation_to_string(args.kwarg.annotation) if args.kwarg.annotation else None,
                'kind': 'var_keyword'
            })

        return params

    def _extract_arg(self, arg: ast.arg, kind: str) -> Dict[str, Any]:
        """Extract single argument info"""
        return {
            'name': arg.arg,
            'type': self._annotation_to_string(arg.annotation) if arg.annotation else None,
            'kind': kind
        }

    def _extract_classes(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract class information"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [self._get_base_name(base) for base in node.bases],
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node)[:200] if ast.get_docstring(node) else None,
                    'methods': [],
                    'class_variables': [],
                    'instance_variables': []
                }

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sig = self._extract_function_signature(item)
                        class_info['methods'].append(self._signature_to_dict(sig))

                    # Class variables (simple assignment at class level)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        class_info['class_variables'].append({
                            'name': item.target.id,
                            'type': self._annotation_to_string(item.annotation) if item.annotation else None
                        })

                classes.append(class_info)

        return classes

    def _calculate_complexity(self, tree: ast.Module, content: str) -> Dict[str, int]:
        """Calculate code complexity metrics"""
        lines = content.split('\n')

        metrics = ComplexityMetrics(
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(tree),
            cognitive_complexity=self._calculate_cognitive_complexity(tree),
            lines_of_code=len([ln for ln in lines if ln.strip() and not ln.strip().startswith('#')]),
            lines_of_comments=len([ln for ln in lines if ln.strip().startswith('#')]),
            blank_lines=len([ln for ln in lines if not ln.strip()]),
            number_of_functions=sum(1 for _ in ast.walk(tree) if isinstance(
                _, (ast.FunctionDef, ast.AsyncFunctionDef))),
            number_of_classes=sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef)),
            max_nesting_depth=self._calculate_max_nesting(tree)
        )

        return {
            'cyclomatic_complexity': metrics.cyclomatic_complexity,
            'cognitive_complexity': metrics.cognitive_complexity,
            'lines_of_code': metrics.lines_of_code,
            'lines_of_comments': metrics.lines_of_comments,
            'blank_lines': metrics.blank_lines,
            'number_of_functions': metrics.number_of_functions,
            'number_of_classes': metrics.number_of_classes,
            'max_nesting_depth': metrics.max_nesting_depth,
            'comment_ratio': metrics.lines_of_comments / max(metrics.lines_of_code, 1)
        }

    def _calculate_cyclomatic_complexity(self, tree: ast.Module) -> int:
        """Calculate McCabe cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Assert, ast.comprehension)):
                complexity += 1

        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.Module) -> int:
        """Calculate cognitive complexity (simplified version)"""
        complexity = 0

        def visit_with_nesting(node, nesting_level=0):
            nonlocal complexity

            # Increment for structural complexity
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1 + nesting_level
                nesting_level += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + nesting_level
            elif isinstance(node, ast.BoolOp):
                complexity += 1

            for child in ast.iter_child_nodes(node):
                visit_with_nesting(child, nesting_level)

        visit_with_nesting(tree)
        return complexity

    def _calculate_max_nesting(self, tree: ast.Module) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0

        def visit(node, depth=0):
            nonlocal max_depth
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                depth += 1
                max_depth = max(max_depth, depth)

            for child in ast.iter_child_nodes(node):
                visit(child, depth)

        visit(tree)
        return max_depth

    def _analyze_type_hints(self, tree: ast.Module) -> Dict[str, Any]:
        """Analyze type hint coverage"""
        total_functions = 0
        functions_with_return_hint = 0
        total_parameters = 0
        parameters_with_hint = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                if node.returns:
                    functions_with_return_hint += 1

                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    total_parameters += 1
                    if arg.annotation:
                        parameters_with_hint += 1

        return {
            'total_functions': total_functions,
            'functions_with_return_hint': functions_with_return_hint,
            'return_type_coverage': functions_with_return_hint / max(total_functions, 1),
            'total_parameters': total_parameters,
            'parameters_with_hint': parameters_with_hint,
            'parameter_type_coverage': parameters_with_hint / max(total_parameters, 1)
        }

    def _analyze_code_style(self, content: str) -> Dict[str, Any]:
        """Analyze code style (simplified PEP8 checks)"""
        lines = content.split('\n')
        issues = []

        for i, line in enumerate(lines, 1):
            # Line length
            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} > 120)")

            # Trailing whitespace
            if line.rstrip() != line:
                issues.append(f"Line {i}: Trailing whitespace")

            # Mixed tabs and spaces
            if '\t' in line and '    ' in line:
                issues.append(f"Line {i}: Mixed tabs and spaces")

        return {
            'issues_count': len(issues),
            'issues': issues[:10],  # Limit to first 10 issues
            'max_line_length': max(len(ln) for ln in lines) if lines else 0,
            'uses_tabs': any('\t' in ln for ln in lines),
            'uses_spaces': any('    ' in ln for ln in lines)
        }

    def _get_decorator_name(self, node) -> str:
        """Get decorator name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return str(node)

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name"""
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
            return self._annotation_to_string(node)
        return str(node)

    def _annotation_to_string(self, node) -> str:
        """Convert annotation AST node to string"""
        if node is None:
            return 'None'
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            base = self._annotation_to_string(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ', '.join(self._annotation_to_string(elt) for elt in node.slice.elts)
            else:
                args = self._annotation_to_string(node.slice)
            return f"{base}[{args}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type (X | Y)
            left = self._annotation_to_string(node.left)
            right = self._annotation_to_string(node.right)
            return f"{left} | {right}"
        return 'Any'

    def _signature_to_dict(self, sig: FunctionSignature) -> Dict[str, Any]:
        """Convert FunctionSignature to dictionary"""
        return {
            'name': sig.name,
            'parameters': sig.parameters,
            'return_type': sig.return_type,
            'decorators': sig.decorators,
            'is_async': sig.is_async,
            'is_classmethod': sig.is_classmethod,
            'is_staticmethod': sig.is_staticmethod,
            'is_property': sig.is_property,
            'docstring': sig.docstring
        }
