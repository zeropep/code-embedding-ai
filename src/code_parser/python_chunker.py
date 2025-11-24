"""
Python-specific code chunking strategies.
Optimized for semantic splitting of Python code.
"""

import ast
from typing import List, Optional
from dataclasses import dataclass
from .models import CodeChunk, CodeLanguage, LayerType


@dataclass
class ChunkingConfig:
    """Configuration for Python code chunking"""
    min_tokens: int = 50
    max_tokens: int = 500
    overlap_tokens: int = 20
    include_docstrings: bool = True
    include_imports: bool = True
    split_large_functions: bool = True
    merge_small_classes: bool = True
    preserve_decorators: bool = True


class PythonChunker:
    """
    Intelligent chunking strategy for Python code.
    Respects Python's semantic structure (classes, functions, methods).
    """

    def __init__(self, config: ChunkingConfig = None):
        if config is None:
            config = ChunkingConfig()
        self.config = config

    def chunk_file(self, content: str, file_path: str,
                   layer_type: LayerType = LayerType.UNKNOWN) -> List[CodeChunk]:
        """
        Chunk a Python file into semantically meaningful pieces.

        Strategy:
        1. Parse AST to identify structure
        2. Extract module-level elements (imports, constants)
        3. Extract classes and their methods
        4. Extract standalone functions
        5. Apply size-based splitting/merging
        """
        chunks = []
        lines = content.split('\n')

        try:
            tree = ast.parse(content)

            # Extract module-level docstring
            ast.get_docstring(tree)

            # Extract imports section
            if self.config.include_imports:
                import_chunk = self._extract_imports_chunk(tree, lines, file_path)
                if import_chunk:
                    chunks.append(import_chunk)

            # Process top-level nodes
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_chunks = self._chunk_class(node, lines, file_path, layer_type)
                    chunks.extend(class_chunks)

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_chunks = self._chunk_function(node, lines, file_path, None, layer_type)
                    chunks.extend(func_chunks)

            # Merge small chunks if needed
            if self.config.merge_small_classes:
                chunks = self._merge_small_chunks(chunks)

        except SyntaxError:
            # Fall back to line-based chunking
            chunks = self._fallback_chunking(content, file_path, layer_type)

        return chunks

    def _extract_imports_chunk(self, tree: ast.Module, lines: List[str],
                               file_path: str) -> Optional[CodeChunk]:
        """Extract import statements as a single chunk"""
        first_import_line = None
        last_import_line = None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if first_import_line is None:
                    first_import_line = node.lineno
                last_import_line = self._get_node_end_line(node, lines)

        if first_import_line is None:
            return None

        content = '\n'.join(lines[first_import_line - 1:last_import_line])
        token_count = len(content) // 4

        if token_count < 10:  # Too small to be useful
            return None

        return CodeChunk(
            content=content,
            file_path=file_path,
            language=CodeLanguage.PYTHON,
            start_line=first_import_line,
            end_line=last_import_line,
            layer_type=LayerType.UNKNOWN,
            token_count=token_count,
            metadata={'type': 'imports'}
        )

    def _chunk_class(self, node: ast.ClassDef, lines: List[str],
                     file_path: str, layer_type: LayerType) -> List[CodeChunk]:
        """Chunk a class into appropriate pieces"""
        chunks = []

        start_line = self._get_decorator_start(node, lines)
        end_line = self._get_node_end_line(node, lines)
        class_content = '\n'.join(lines[start_line - 1:end_line])
        class_tokens = len(class_content) // 4

        # If class is small enough, keep it as one chunk
        if class_tokens <= self.config.max_tokens:
            chunks.append(CodeChunk(
                content=class_content,
                file_path=file_path,
                language=CodeLanguage.PYTHON,
                start_line=start_line,
                end_line=end_line,
                class_name=node.name,
                layer_type=layer_type,
                token_count=class_tokens,
                metadata={
                    'type': 'class',
                    'docstring': ast.get_docstring(node)[:200] if ast.get_docstring(node) else None
                }
            ))
            return chunks

        # Large class: extract class header + fields, then methods separately
        # Class header (up to first method or end of class body)
        header_end = self._find_class_header_end(node, lines)
        if header_end > start_line:
            header_content = '\n'.join(lines[start_line - 1:header_end])
            header_tokens = len(header_content) // 4

            if header_tokens >= self.config.min_tokens:
                chunks.append(CodeChunk(
                    content=header_content,
                    file_path=file_path,
                    language=CodeLanguage.PYTHON,
                    start_line=start_line,
                    end_line=header_end,
                    class_name=node.name,
                    layer_type=layer_type,
                    token_count=header_tokens,
                    metadata={'type': 'class_header'}
                ))

        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunks = self._chunk_function(item, lines, file_path, node.name, layer_type)
                chunks.extend(method_chunks)

        return chunks

    def _chunk_function(self, node, lines: List[str], file_path: str,
                        class_name: Optional[str], layer_type: LayerType) -> List[CodeChunk]:
        """Chunk a function/method into appropriate pieces"""
        chunks = []

        start_line = self._get_decorator_start(node, lines)
        end_line = self._get_node_end_line(node, lines)
        func_content = '\n'.join(lines[start_line - 1:end_line])
        func_tokens = len(func_content) // 4

        is_async = isinstance(node, ast.AsyncFunctionDef)

        # If function is small enough, keep it as one chunk
        if func_tokens <= self.config.max_tokens:
            if func_tokens >= self.config.min_tokens:
                chunks.append(CodeChunk(
                    content=func_content,
                    file_path=file_path,
                    language=CodeLanguage.PYTHON,
                    start_line=start_line,
                    end_line=end_line,
                    function_name=node.name,
                    class_name=class_name,
                    layer_type=layer_type,
                    token_count=func_tokens,
                    metadata={
                        'type': 'method' if class_name else 'function',
                        'is_async': is_async,
                        'docstring': ast.get_docstring(node)[:200] if ast.get_docstring(node) else None
                    }
                ))
            return chunks

        # Large function: split by logical blocks
        if self.config.split_large_functions:
            return self._split_large_function(node, lines, file_path, class_name, layer_type)
        else:
            # Just create one large chunk
            chunks.append(CodeChunk(
                content=func_content,
                file_path=file_path,
                language=CodeLanguage.PYTHON,
                start_line=start_line,
                end_line=end_line,
                function_name=node.name,
                class_name=class_name,
                layer_type=layer_type,
                token_count=func_tokens,
                metadata={'type': 'method' if class_name else 'function', 'is_async': is_async}
            ))

        return chunks

    def _split_large_function(self, node, lines: List[str], file_path: str,
                              class_name: Optional[str], layer_type: LayerType) -> List[CodeChunk]:
        """Split a large function into logical blocks"""
        chunks = []

        start_line = self._get_decorator_start(node, lines)
        end_line = self._get_node_end_line(node, lines)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Find logical split points (if/for/while/try/with blocks)
        split_points = [start_line]

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                if hasattr(child, 'lineno'):
                    split_points.append(child.lineno)

        split_points.append(end_line + 1)
        split_points = sorted(set(split_points))

        # Create chunks based on split points
        current_start = start_line
        current_content = []
        current_tokens = 0

        for i, line_num in enumerate(range(start_line, end_line + 1)):
            line = lines[line_num - 1]
            line_tokens = len(line) // 4
            current_content.append(line)
            current_tokens += line_tokens

            # Check if we should create a chunk
            is_split_point = (line_num + 1) in split_points
            exceeds_max = current_tokens >= self.config.max_tokens
            is_last_line = line_num == end_line

            if (is_split_point and current_tokens >= self.config.min_tokens) or exceeds_max or is_last_line:
                if current_tokens >= self.config.min_tokens:
                    chunk_content = '\n'.join(current_content)
                    chunk_num = len(chunks) + 1

                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=CodeLanguage.PYTHON,
                        start_line=current_start,
                        end_line=line_num,
                        function_name=f"{node.name}_part{chunk_num}",
                        class_name=class_name,
                        layer_type=layer_type,
                        token_count=current_tokens,
                        metadata={
                            'type': 'function_part',
                            'original_function': node.name,
                            'part_number': chunk_num,
                            'is_async': is_async
                        }
                    ))

                    current_start = line_num + 1
                    current_content = []
                    current_tokens = 0

        return chunks

    def _merge_small_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Merge adjacent small chunks that are below minimum token count"""
        if len(chunks) <= 1:
            return chunks

        merged = []
        current_group = []
        current_tokens = 0

        for chunk in chunks:
            # Don't merge different types
            if (current_group and
                    current_group[0].metadata.get('type') != chunk.metadata.get('type')):
                if current_tokens >= self.config.min_tokens:
                    merged.append(self._combine_chunks(current_group))
                else:
                    merged.extend(current_group)
                current_group = [chunk]
                current_tokens = chunk.token_count
            # Check if adding this chunk would exceed max
            elif current_tokens + chunk.token_count <= self.config.max_tokens:
                current_group.append(chunk)
                current_tokens += chunk.token_count
            else:
                if current_group:
                    merged.append(self._combine_chunks(current_group))
                current_group = [chunk]
                current_tokens = chunk.token_count

        # Handle remaining group
        if current_group:
            merged.append(self._combine_chunks(current_group))

        return merged

    def _combine_chunks(self, chunks: List[CodeChunk]) -> CodeChunk:
        """Combine multiple chunks into one"""
        if len(chunks) == 1:
            return chunks[0]

        combined_content = '\n\n'.join(c.content for c in chunks)
        total_tokens = sum(c.token_count for c in chunks)

        # Use the first chunk as base
        base = chunks[0]

        return CodeChunk(
            content=combined_content,
            file_path=base.file_path,
            language=base.language,
            start_line=chunks[0].start_line,
            end_line=chunks[-1].end_line,
            class_name=base.class_name,
            function_name=base.function_name,
            layer_type=base.layer_type,
            token_count=total_tokens,
            metadata={
                'type': 'merged',
                'original_chunks': len(chunks)
            }
        )

    def _get_decorator_start(self, node, lines: List[str]) -> int:
        """Get the line number where decorators start"""
        if node.decorator_list:
            return min(d.lineno for d in node.decorator_list)
        return node.lineno

    def _get_node_end_line(self, node, lines: List[str]) -> int:
        """Get the end line of an AST node"""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
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

            if not stripped or stripped.startswith('#'):
                continue

            current_indent = len(line) - len(line.lstrip())

            if current_indent <= start_indent and stripped:
                return i

        return len(lines)

    def _find_class_header_end(self, node: ast.ClassDef, lines: List[str]) -> int:
        """Find where class header ends (before first method)"""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Return line before first method
                return item.lineno - 1

        # No methods, return class end
        return self._get_node_end_line(node, lines)

    def _fallback_chunking(self, content: str, file_path: str,
                           layer_type: LayerType) -> List[CodeChunk]:
        """Fallback to simple line-based chunking when AST fails"""
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_tokens = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_tokens = len(line) // 4
            current_chunk.append(line)
            current_tokens += line_tokens

            # Create chunk if we exceed max tokens
            if current_tokens >= self.config.max_tokens:
                chunk_content = '\n'.join(current_chunk)
                if current_tokens >= self.config.min_tokens:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=CodeLanguage.PYTHON,
                        start_line=start_line,
                        end_line=i,
                        layer_type=layer_type,
                        token_count=current_tokens,
                        metadata={'type': 'fallback'}
                    ))

                start_line = i + 1
                current_chunk = []
                current_tokens = 0

        # Handle remaining content
        if current_chunk and current_tokens >= self.config.min_tokens:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                language=CodeLanguage.PYTHON,
                start_line=start_line,
                end_line=len(lines),
                layer_type=layer_type,
                token_count=current_tokens,
                metadata={'type': 'fallback'}
            ))

        return chunks
