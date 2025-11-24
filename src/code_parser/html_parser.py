from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup
from .base_parser import BaseParser
from .models import ParsedFile, CodeChunk, CodeLanguage, LayerType


class HTMLParser(BaseParser):
    """Parser for HTML/Thymeleaf template files"""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in ['.html', '.htm']

    def get_language(self) -> CodeLanguage:
        return CodeLanguage.HTML

    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """Parse HTML/Thymeleaf file into chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file stats
            total_lines, last_modified = self.get_file_stats(file_path)
            file_hash = self.calculate_file_hash(file_path)

            # Parse HTML content
            chunks = self._parse_html_content(content, str(file_path))

            return ParsedFile(
                file_path=str(file_path),
                language=self.get_language(),
                chunks=chunks,
                total_lines=total_lines,
                file_hash=file_hash,
                last_modified=last_modified
            )

        except Exception as e:
            print(f"Error parsing HTML file {file_path}: {e}")
            return None

    def _parse_html_content(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse HTML content into meaningful chunks"""
        chunks = []
        lines = content.split('\n')

        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Extract Thymeleaf fragments
            fragments = self._extract_thymeleaf_fragments(soup, lines, file_path)
            chunks.extend(fragments)

            # Extract forms
            forms = self._extract_forms(soup, lines, file_path)
            chunks.extend(forms)

            # Extract tables
            tables = self._extract_tables(soup, lines, file_path)
            chunks.extend(tables)

            # Extract sections with Thymeleaf logic
            sections = self._extract_thymeleaf_sections(soup, lines, file_path)
            chunks.extend(sections)

            # If no specific chunks found, fall back to general chunking
            if not chunks:
                chunks = self.chunk_content(content, file_path, self.get_language())

        except Exception as e:
            print(f"HTML parsing failed for {file_path}: {e}")
            # Fall back to simple chunking
            chunks = self.chunk_content(content, file_path, self.get_language())

        return chunks

    def _extract_thymeleaf_fragments(self, soup: BeautifulSoup, lines: List[str],
                                     file_path: str) -> List[CodeChunk]:
        """Extract Thymeleaf fragments (th:fragment)"""
        chunks = []

        # Find elements with th:fragment attribute
        fragments = soup.find_all(attrs={'th:fragment': True})

        for fragment in fragments:
            fragment_name = fragment.get('th:fragment', 'unnamed')
            start_line, end_line = self._find_element_lines(fragment, lines)

            if start_line and end_line:
                content = '\n'.join(lines[start_line-1:end_line])
                chunk = CodeChunk(
                    content=content,
                    file_path=file_path,
                    language=self.get_language(),
                    start_line=start_line,
                    end_line=end_line,
                    function_name=f"fragment_{fragment_name}",
                    layer_type=LayerType.TEMPLATE,
                    metadata={
                        'fragment_name': fragment_name,
                        'element_type': fragment.name,
                        'thymeleaf_attributes': self._extract_thymeleaf_attrs(fragment)
                    }
                )
                chunks.append(chunk)

        return chunks

    def _extract_forms(self, soup: BeautifulSoup, lines: List[str], file_path: str) -> List[CodeChunk]:
        """Extract HTML forms"""
        chunks = []

        forms = soup.find_all('form')
        for i, form in enumerate(forms):
            start_line, end_line = self._find_element_lines(form, lines)

            if start_line and end_line:
                content = '\n'.join(lines[start_line-1:end_line])

                # Extract form details
                form_action = form.get('action', '')
                form_method = form.get('method', 'GET')
                form_id = form.get('id', f'form_{i+1}')

                chunk = CodeChunk(
                    content=content,
                    file_path=file_path,
                    language=self.get_language(),
                    start_line=start_line,
                    end_line=end_line,
                    function_name=f"form_{form_id}",
                    layer_type=LayerType.TEMPLATE,
                    metadata={
                        'form_action': form_action,
                        'form_method': form_method,
                        'form_id': form_id,
                        'thymeleaf_attributes': self._extract_thymeleaf_attrs(form),
                        'input_fields': self._extract_form_fields(form)
                    }
                )
                chunks.append(chunk)

        return chunks

    def _extract_tables(self, soup: BeautifulSoup, lines: List[str], file_path: str) -> List[CodeChunk]:
        """Extract HTML tables"""
        chunks = []

        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            start_line, end_line = self._find_element_lines(table, lines)

            if start_line and end_line:
                content = '\n'.join(lines[start_line-1:end_line])

                table_id = table.get('id', f'table_{i+1}')

                chunk = CodeChunk(
                    content=content,
                    file_path=file_path,
                    language=self.get_language(),
                    start_line=start_line,
                    end_line=end_line,
                    function_name=f"table_{table_id}",
                    layer_type=LayerType.TEMPLATE,
                    metadata={
                        'table_id': table_id,
                        'thymeleaf_attributes': self._extract_thymeleaf_attrs(table),
                        'column_count': len(table.find_all('th')) if table.find('th') else 0
                    }
                )
                chunks.append(chunk)

        return chunks

    def _extract_thymeleaf_sections(self, soup: BeautifulSoup, lines: List[str],
                                    file_path: str) -> List[CodeChunk]:
        """Extract sections with Thymeleaf logic (loops, conditionals)"""
        chunks = []

        # Find elements with Thymeleaf control attributes
        thymeleaf_attrs = ['th:each', 'th:if', 'th:unless', 'th:switch', 'th:case']

        for attr in thymeleaf_attrs:
            elements = soup.find_all(attrs={attr: True})

            for element in elements:
                start_line, end_line = self._find_element_lines(element, lines)

                if start_line and end_line:
                    content = '\n'.join(lines[start_line-1:end_line])

                    attr_value = element.get(attr, '')
                    element_id = element.get('id', element.name)

                    chunk = CodeChunk(
                        content=content,
                        file_path=file_path,
                        language=self.get_language(),
                        start_line=start_line,
                        end_line=end_line,
                        function_name=f"{attr.replace(':', '_')}_{element_id}",
                        layer_type=LayerType.TEMPLATE,
                        metadata={
                            'thymeleaf_type': attr,
                            'thymeleaf_value': attr_value,
                            'element_type': element.name,
                            'thymeleaf_attributes': self._extract_thymeleaf_attrs(element)
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _find_element_lines(self, element, lines: List[str]) -> tuple[Optional[int], Optional[int]]:
        """Find start and end line numbers for an HTML element"""
        element_str = str(element)
        element_lines = element_str.split('\n')

        if not element_lines:
            return None, None

        # Find the first line of the element
        first_element_line = element_lines[0].strip()
        start_line = None

        for i, line in enumerate(lines):
            if first_element_line in line.strip():
                start_line = i + 1
                break

        if start_line is None:
            return None, None

        # Calculate end line
        end_line = start_line + len(element_lines) - 1

        return start_line, end_line

    def _extract_thymeleaf_attrs(self, element) -> dict:
        """Extract all Thymeleaf attributes from an element"""
        thymeleaf_attrs = {}

        if element.attrs:
            for attr, value in element.attrs.items():
                if attr.startswith('th:') or attr.startswith('data-th-'):
                    thymeleaf_attrs[attr] = value

        return thymeleaf_attrs

    def _extract_form_fields(self, form) -> list:
        """Extract form field information"""
        fields = []

        inputs = form.find_all(['input', 'select', 'textarea'])
        for input_elem in inputs:
            field_info = {
                'type': input_elem.get('type', input_elem.name),
                'name': input_elem.get('name', ''),
                'id': input_elem.get('id', ''),
                'required': input_elem.has_attr('required'),
                'thymeleaf_attrs': self._extract_thymeleaf_attrs(input_elem)
            }
            fields.append(field_info)

        return fields
