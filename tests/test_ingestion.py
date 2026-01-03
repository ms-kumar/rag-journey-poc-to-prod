"""
Unit tests for document ingestion service.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.services.ingestion.client import IngestionClient


class TestIngestionClient:
    """Test IngestionClient with various scenarios."""

    def test_initialization(self):
        """Test client initialization."""
        client = IngestionClient(directory="./test_data")
        assert client.directory == Path("./test_data")

    def test_initialization_default_directory(self):
        """Test client with default directory."""
        client = IngestionClient()
        assert client.directory == Path("./data")

    def test_ingest_single_file(self):
        """Test ingesting a single text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test1.txt"
            test_file.write_text("This is test content.", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == "This is test content."

    def test_ingest_multiple_files(self):
        """Test ingesting multiple text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            (Path(tmpdir) / "file1.txt").write_text("Content 1", encoding="utf-8")
            (Path(tmpdir) / "file2.txt").write_text("Content 2", encoding="utf-8")
            (Path(tmpdir) / "file3.txt").write_text("Content 3", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 3
            assert set(documents) == {"Content 1", "Content 2", "Content 3"}

    def test_ingest_empty_directory(self):
        """Test ingesting from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert documents == []

    def test_ingest_no_txt_files(self):
        """Test directory with no .txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-txt files
            (Path(tmpdir) / "file1.md").write_text("Markdown", encoding="utf-8")
            (Path(tmpdir) / "file2.py").write_text("Python", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert documents == []

    def test_ingest_mixed_file_types(self):
        """Test directory with mixed file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mixed files
            (Path(tmpdir) / "doc.txt").write_text("Text file", encoding="utf-8")
            (Path(tmpdir) / "readme.md").write_text("Markdown", encoding="utf-8")
            (Path(tmpdir) / "script.py").write_text("Python", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            # Should only ingest .txt files
            assert len(documents) == 1
            assert documents[0] == "Text file"

    def test_ingest_empty_file(self):
        """Test ingesting empty text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty file
            (Path(tmpdir) / "empty.txt").write_text("", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == ""

    def test_ingest_whitespace_only_file(self):
        """Test ingesting file with only whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "whitespace.txt").write_text("   \n\t\n   ", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            # Should preserve whitespace
            assert documents[0] == "   \n\t\n   "

    def test_ingest_unicode_content(self):
        """Test ingesting files with unicode content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "unicode.txt").write_text("Hello 世界 🌍", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == "Hello 世界 🌍"

    def test_ingest_multiline_content(self):
        """Test ingesting files with multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "Line 1\nLine 2\nLine 3"
            (Path(tmpdir) / "multiline.txt").write_text(content, encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == content

    def test_ingest_large_file(self):
        """Test ingesting large file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large file
            large_content = "Lorem ipsum " * 10000
            (Path(tmpdir) / "large.txt").write_text(large_content, encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == large_content

    def test_ingest_special_characters(self):
        """Test ingesting files with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
            (Path(tmpdir) / "special.txt").write_text(content, encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == content

    def test_ingest_nested_directory_ignored(self):
        """Test that nested directories are not recursed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in root
            (Path(tmpdir) / "root.txt").write_text("Root file", encoding="utf-8")
            
            # Create nested directory with file
            nested_dir = Path(tmpdir) / "nested"
            nested_dir.mkdir()
            (nested_dir / "nested.txt").write_text("Nested file", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            # Should only ingest root file (glob "*.txt" doesn't recurse)
            assert len(documents) == 1
            assert documents[0] == "Root file"

    def test_ingest_file_with_dots_in_name(self):
        """Test ingesting files with dots in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.with.dots.txt").write_text("Dotted", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == "Dotted"

    def test_ingest_preserves_content_exactly(self):
        """Test that ingestion preserves content exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_content = "  Exact\n\nContent  \t\n"
            (Path(tmpdir) / "exact.txt").write_text(original_content, encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()
            
            assert len(documents) == 1
            assert documents[0] == original_content

    def test_ingest_nonexistent_directory(self):
        """Test ingesting from non-existent directory."""
        client = IngestionClient(directory="/nonexistent/path/12345")
        
        # Should handle gracefully (glob on non-existent path returns empty)
        documents = client.ingest()
        assert documents == []

    def test_ingest_multiple_calls_consistent(self):
        """Test that multiple ingest calls return consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("Content", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            docs1 = client.ingest()
            docs2 = client.ingest()
            
            assert docs1 == docs2

    def test_ingest_order_consistency(self):
        """Test that ingestion order is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            for i in range(5):
                (Path(tmpdir) / f"file{i}.txt").write_text(f"Content {i}", encoding="utf-8")
            
            client = IngestionClient(directory=tmpdir)
            docs1 = client.ingest()
            docs2 = client.ingest()
            
            # Order should be consistent between calls
            assert docs1 == docs2
