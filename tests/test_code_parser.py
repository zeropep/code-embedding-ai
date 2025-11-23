"""
Tests for code parsing functionality
"""

import pytest
from pathlib import Path

from src.code_parser.models import CodeLanguage, LayerType, ParserConfig
from src.code_parser.java_parser import JavaParser
from src.code_parser.html_parser import HTMLParser
from src.code_parser.parser_factory import ParserFactory
from src.code_parser.code_parser import CodeParser


class TestParserModels:
    """Test parser model classes"""

    def test_code_chunk_creation(self, sample_code_chunk):
        """Test CodeChunk creation and properties"""
        chunk = sample_code_chunk

        assert chunk.file_path == "src/main/java/com/example/User.java"
        assert chunk.language == CodeLanguage.JAVA
        assert chunk.function_name == "getName"
        assert chunk.class_name == "User"
        assert chunk.layer_type == LayerType.ENTITY
        assert chunk.token_count == 12

    def test_code_chunk_to_dict(self, sample_code_chunk):
        """Test CodeChunk serialization"""
        chunk_dict = sample_code_chunk.to_dict()

        assert "content" in chunk_dict
        assert "file_path" in chunk_dict
        assert "language" in chunk_dict
        assert chunk_dict["language"] == "java"
        assert chunk_dict["layer_type"] == "Entity"

    def test_parser_config_defaults(self):
        """Test ParserConfig default values"""
        config = ParserConfig()

        assert config.min_tokens == 50
        assert config.max_tokens == 500
        assert config.overlap_tokens == 20
        assert config.include_comments is False
        assert ".java" in config.supported_extensions


class TestJavaParser:
    """Test Java code parsing"""

    def test_java_parser_can_parse(self, parser_config):
        """Test Java parser file detection"""
        parser = JavaParser(parser_config)

        assert parser.can_parse(Path("test.java")) is True
        assert parser.can_parse(Path("test.kt")) is False
        assert parser.can_parse(Path("test.html")) is False

    def test_java_parser_language(self, parser_config):
        """Test Java parser language identification"""
        parser = JavaParser(parser_config)
        assert parser.get_language() == CodeLanguage.JAVA

    def test_java_parser_parse_file(self, parser_config, temp_dir, sample_java_code):
        """Test parsing a Java file"""
        parser = JavaParser(parser_config)

        # Create test file
        java_file = temp_dir / "TestService.java"
        java_file.write_text(sample_java_code)

        # Parse file
        parsed_file = parser.parse_file(java_file)

        assert parsed_file is not None
        assert parsed_file.language == CodeLanguage.JAVA
        assert parsed_file.file_path == str(java_file)
        assert len(parsed_file.chunks) > 0

        # Check if we found the UserService class
        class_chunks = [c for c in parsed_file.chunks if c.class_name == "UserService"]
        assert len(class_chunks) > 0

        # Check if we found methods
        method_chunks = [c for c in parsed_file.chunks if c.function_name]
        assert len(method_chunks) > 0

    def test_java_parser_layer_detection(self, parser_config, temp_dir):
        """Test Spring Boot layer type detection"""
        parser = JavaParser(parser_config)

        # Test Controller
        controller_code = """
@RestController
public class UserController {
    @GetMapping("/users")
    public List<User> getUsers() { return null; }
}
"""
        java_file = temp_dir / "UserController.java"
        java_file.write_text(controller_code)

        parsed_file = parser.parse_file(java_file)
        assert parsed_file is not None

        controller_chunks = [c for c in parsed_file.chunks if c.layer_type == LayerType.CONTROLLER]
        assert len(controller_chunks) > 0

    def test_java_parser_chunking(self, temp_dir):
        """Test code chunking behavior"""
        config = ParserConfig(min_tokens=5, max_tokens=30)
        parser = JavaParser(config)

        # Create a larger Java file
        large_code = """
public class LargeClass {
""" + "\n".join([f"    public void method{i}() {{ System.out.println(\"Method {i}\"); }}" for i in range(20)]) + """
}
"""

        java_file = temp_dir / "LargeClass.java"
        java_file.write_text(large_code)

        parsed_file = parser.parse_file(java_file)

        assert parsed_file is not None
        assert len(parsed_file.chunks) > 1  # Should be split into multiple chunks

        # Check token counts
        for chunk in parsed_file.chunks:
            assert chunk.token_count <= config.max_tokens


class TestHTMLParser:
    """Test HTML/Thymeleaf parsing"""

    def test_html_parser_can_parse(self, parser_config):
        """Test HTML parser file detection"""
        parser = HTMLParser(parser_config)

        assert parser.can_parse(Path("test.html")) is True
        assert parser.can_parse(Path("test.htm")) is True
        assert parser.can_parse(Path("test.java")) is False

    def test_html_parser_parse_file(self, parser_config, temp_dir, sample_html_code):
        """Test parsing an HTML file"""
        parser = HTMLParser(parser_config)

        # Create test file
        html_file = temp_dir / "user-profile.html"
        html_file.write_text(sample_html_code)

        # Parse file
        parsed_file = parser.parse_file(html_file)

        assert parsed_file is not None
        assert parsed_file.language == CodeLanguage.HTML
        assert len(parsed_file.chunks) > 0

        # Check for Thymeleaf fragments
        fragment_chunks = [c for c in parsed_file.chunks if "fragment" in c.function_name]
        assert len(fragment_chunks) > 0

        # Check for forms
        form_chunks = [c for c in parsed_file.chunks if "form" in c.function_name]
        assert len(form_chunks) > 0

    def test_html_parser_thymeleaf_detection(self, parser_config, temp_dir):
        """Test Thymeleaf-specific parsing"""
        parser = HTMLParser(parser_config)

        thymeleaf_code = """
<div th:fragment="header">
    <h1>Header</h1>
</div>
<div th:each="item : ${items}">
    <span th:text="${item.name}">Item name</span>
</div>
"""

        html_file = temp_dir / "thymeleaf.html"
        html_file.write_text(thymeleaf_code)

        parsed_file = parser.parse_file(html_file)

        assert parsed_file is not None
        # Should find Thymeleaf constructs
        thymeleaf_chunks = [c for c in parsed_file.chunks
                           if any(attr.startswith('th:') for attr in str(c.metadata))]
        assert len(thymeleaf_chunks) > 0


class TestParserFactory:
    """Test parser factory functionality"""

    def test_parser_factory_initialization(self, parser_config):
        """Test parser factory setup"""
        factory = ParserFactory(parser_config)

        assert factory.config == parser_config
        assert len(factory._parsers) > 0

    def test_parser_factory_get_parser(self, parser_config):
        """Test getting appropriate parser for file types"""
        factory = ParserFactory(parser_config)

        java_parser = factory.get_parser(Path("test.java"))
        assert isinstance(java_parser, JavaParser)

        html_parser = factory.get_parser(Path("test.html"))
        assert isinstance(html_parser, HTMLParser)

    def test_parser_factory_can_parse_file(self, parser_config):
        """Test file type support checking"""
        factory = ParserFactory(parser_config)

        assert factory.can_parse_file(Path("test.java")) is True
        assert factory.can_parse_file(Path("test.html")) is True
        assert factory.can_parse_file(Path("test.txt")) is False

    def test_parser_factory_parse_directory(self, parser_config, temp_dir, sample_java_code):
        """Test parsing entire directory"""
        factory = ParserFactory(parser_config)

        # Create test files
        (temp_dir / "src").mkdir()
        java_file = temp_dir / "src" / "Test.java"
        java_file.write_text(sample_java_code)

        html_file = temp_dir / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        # Parse directory
        parsed_files = factory.parse_directory(temp_dir)

        assert len(parsed_files) >= 2
        file_types = [pf.language for pf in parsed_files]
        assert CodeLanguage.JAVA in file_types
        assert CodeLanguage.HTML in file_types


class TestCodeParser:
    """Test main code parser orchestrator"""

    def test_code_parser_initialization(self, parser_config):
        """Test CodeParser initialization"""
        parser = CodeParser(parser_config)

        assert parser.config == parser_config
        assert parser.parser_factory is not None

    def test_code_parser_parse_files(self, parser_config, temp_dir, sample_java_code):
        """Test parsing specific files"""
        parser = CodeParser(parser_config)

        # Create test file
        java_file = temp_dir / "Test.java"
        java_file.write_text(sample_java_code)

        # Parse files
        parsed_files = parser.parse_files([str(java_file)])

        assert len(parsed_files) == 1
        assert parsed_files[0].language == CodeLanguage.JAVA

    def test_code_parser_get_chunks_for_embedding(self, parser_config, create_test_chunks):
        """Test chunk filtering for embedding"""
        parser = CodeParser(parser_config)

        # Create chunks with different sizes
        chunks_data = []
        for i in range(5):
            chunk = create_test_chunks(1)[0]
            chunk.token_count = (i + 1) * 20  # 20, 40, 60, 80, 100
            chunks_data.append(chunk)

        # Create mock parsed files
        from src.code_parser.models import ParsedFile
        parsed_file = ParsedFile(
            file_path="test.java",
            language=CodeLanguage.JAVA,
            chunks=chunks_data,
            total_lines=100,
            file_hash="testhash",
            last_modified=0.0
        )

        # Filter chunks
        filtered_chunks = parser.get_chunks_for_embedding([parsed_file])

        # Should include chunks within token range
        assert len(filtered_chunks) > 0
        for chunk in filtered_chunks:
            assert parser_config.min_tokens <= chunk.token_count <= parser_config.max_tokens

    @pytest.mark.asyncio
    async def test_code_parser_async_operations(self, parser_config, temp_dir, sample_java_code):
        """Test async parsing operations"""
        parser = CodeParser(parser_config)

        # Create test file
        java_file = temp_dir / "Test.java"
        java_file.write_text(sample_java_code)

        # Test async repository parsing
        parsed_files = await parser.parse_repository_async(str(temp_dir))

        assert len(parsed_files) >= 1