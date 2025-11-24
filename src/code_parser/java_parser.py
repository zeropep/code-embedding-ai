import re
from pathlib import Path
from typing import List, Optional
import javalang
from .base_parser import BaseParser
from .models import ParsedFile, CodeChunk, CodeLanguage, LayerType


class JavaParser(BaseParser):
    """Parser for Java source code"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.java'

    def get_language(self) -> CodeLanguage:
        return CodeLanguage.JAVA

    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Parse Java file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file stats
            total_lines, last_modified = self.get_file_stats(file_path)
            file_hash = self.calculate_file_hash(file_path)

            # Parse Java AST
            chunks = self._parse_java_ast(content, str(file_path))

            # If AST parsing fails, fall back to simple chunking
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
            print(f"Error parsing Java file {file_path}: {e}")
            return None

    def _parse_java_ast(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Java using AST to extract methods and classes"""
        chunks = []
        lines = content.split('\n')

        try:
            # Parse with javalang
            tree = javalang.parse.parse(content)

            # Extract package and imports info
            package_name = tree.package.name if tree.package else ""
            [imp.path for imp in tree.imports] if tree.imports else []

            # Extract classes and their methods
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                class_chunk = self._extract_class_chunk(node, lines, file_path, package_name)
                if class_chunk:
                    chunks.append(class_chunk)

                # Extract methods within the class
                for method_path, method_node in node.filter(javalang.tree.MethodDeclaration):
                    method_chunk = self._extract_method_chunk(
                        method_node, lines, file_path, node.name, package_name
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

            # Extract interfaces
            for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
                interface_chunk = self._extract_interface_chunk(node, lines, file_path, package_name)
                if interface_chunk:
                    chunks.append(interface_chunk)

        except Exception as e:
            print(f"AST parsing failed for {file_path}: {e}")
            # Fall back to regex-based parsing
            chunks = self._parse_java_regex(content, file_path)

        return chunks

    def _extract_class_chunk(self, node: javalang.tree.ClassDeclaration,
                             lines: List[str], file_path: str, package_name: str) -> Optional[CodeChunk]:
        """Extract class declaration and fields"""
        if not hasattr(node, 'position') or not node.position:
            return None

        start_line = node.position.line
        # Find class end by looking for the closing brace
        end_line = self._find_class_end(lines, start_line - 1)

        if end_line == -1:
            return None

        content = '\n'.join(lines[start_line-1:end_line])

        # Determine layer type from class name and annotations
        layer_type = self._determine_layer_type(node.name, content)

        return CodeChunk(
            content=content,
            file_path=file_path,
            language=self.get_language(),
            start_line=start_line,
            end_line=end_line,
            class_name=node.name,
            layer_type=layer_type,
            metadata={
                'package': package_name,
                'modifiers': [mod for mod in node.modifiers] if node.modifiers else [],
                'extends': node.extends.name if node.extends else None,
                'implements': [impl.name for impl in node.implements] if node.implements else []
            }
        )

    def _extract_method_chunk(self, node: javalang.tree.MethodDeclaration,
                              lines: List[str], file_path: str, class_name: str,
                              package_name: str) -> Optional[CodeChunk]:
        """Extract method implementation"""
        if not hasattr(node, 'position') or not node.position:
            return None

        start_line = node.position.line
        end_line = self._find_method_end(lines, start_line - 1)

        if end_line == -1:
            return None

        content = '\n'.join(lines[start_line-1:end_line])

        return CodeChunk(
            content=content,
            file_path=file_path,
            language=self.get_language(),
            start_line=start_line,
            end_line=end_line,
            function_name=node.name,
            class_name=class_name,
            layer_type=self._determine_layer_type(class_name, content),
            metadata={
                'package': package_name,
                'return_type': node.return_type.name if node.return_type else 'void',
                'parameters': [param.name for param in node.parameters] if node.parameters else [],
                'modifiers': [mod for mod in node.modifiers] if node.modifiers else []
            }
        )

    def _extract_interface_chunk(self, node: javalang.tree.InterfaceDeclaration,
                                 lines: List[str], file_path: str, package_name: str) -> Optional[CodeChunk]:
        """Extract interface declaration"""
        if not hasattr(node, 'position') or not node.position:
            return None

        start_line = node.position.line
        end_line = self._find_class_end(lines, start_line - 1)

        if end_line == -1:
            return None

        content = '\n'.join(lines[start_line-1:end_line])

        return CodeChunk(
            content=content,
            file_path=file_path,
            language=self.get_language(),
            start_line=start_line,
            end_line=end_line,
            class_name=node.name,
            layer_type=LayerType.UNKNOWN,
            metadata={
                'package': package_name,
                'type': 'interface',
                'extends': [ext.name for ext in node.extends] if node.extends else []
            }
        )

    def _find_class_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end line of a class by matching braces"""
        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1

        return -1

    def _find_method_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end line of a method"""
        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            # Check for abstract method (ends with semicolon)
            if ';' in line and not found_opening:
                return i + 1

            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1

        return -1

    def _determine_layer_type(self, name: str, content: str) -> LayerType:
        """Determine Spring Boot layer type from class name and content"""
        name_lower = name.lower()
        content_lower = content.lower()

        # Controller layer
        if ('controller' in name_lower or
            '@controller' in content_lower or
                '@restcontroller' in content_lower):
            return LayerType.CONTROLLER

        # Service layer
        if ('service' in name_lower or
                '@service' in content_lower):
            return LayerType.SERVICE

        # Repository layer
        if ('repository' in name_lower or
            'dao' in name_lower or
                '@repository' in content_lower):
            return LayerType.REPOSITORY

        # Entity layer
        if ('entity' in name_lower or
            'model' in name_lower or
                '@entity' in content_lower):
            return LayerType.ENTITY

        # Configuration
        if ('config' in name_lower or
            'configuration' in name_lower or
                '@configuration' in content_lower):
            return LayerType.CONFIG

        # Test
        if ('test' in name_lower or
                '@test' in content_lower):
            return LayerType.TEST

        # Utility
        if ('util' in name_lower or
                'helper' in name_lower):
            return LayerType.UTIL

        return LayerType.UNKNOWN

    def _parse_java_regex(self, content: str, file_path: str) -> List[CodeChunk]:
        """Fallback regex-based parsing when AST fails"""
        chunks = []
        lines = content.split('\n')

        # Find class declarations
        class_pattern = r'^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)'
        method_pattern = r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'

        current_class = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for class declaration
            class_match = re.match(class_pattern, line)
            if class_match:
                current_class = class_match.group(1)
                # Find class end and create chunk
                end_idx = self._find_class_end(lines, i)
                if end_idx != -1:
                    chunk_content = '\n'.join(lines[i:end_idx])
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=self.get_language(),
                        start_line=i + 1,
                        end_line=end_idx,
                        class_name=current_class,
                        layer_type=self._determine_layer_type(current_class, chunk_content)
                    )
                    chunks.append(chunk)
                    i = end_idx
                    continue

            # Check for method declaration
            method_match = re.match(method_pattern, line)
            if method_match and current_class:
                method_name = method_match.group(1)
                end_idx = self._find_method_end(lines, i)
                if end_idx != -1:
                    chunk_content = '\n'.join(lines[i:end_idx])
                    chunk = CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        language=self.get_language(),
                        start_line=i + 1,
                        end_line=end_idx,
                        function_name=method_name,
                        class_name=current_class,
                        layer_type=self._determine_layer_type(current_class, chunk_content)
                    )
                    chunks.append(chunk)
                    i = end_idx
                    continue

            i += 1

        return chunks
