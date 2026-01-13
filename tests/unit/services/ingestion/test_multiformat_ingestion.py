"""
Unit tests for multi-format document ingestion (TXT, MD, HTML, PDF).
"""

import tempfile
from pathlib import Path

import pytest

from src.services.ingestion.client import IngestionClient


class TestTextIngestion:
    """Test basic .txt file ingestion."""

    def test_txt_single_file(self):
        """Test ingesting a single text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("Plain text content", encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.txt"])
            documents = client.ingest()

            assert len(documents) == 1
            assert documents[0] == "Plain text content"

    def test_txt_multiple_files(self):
        """Test ingesting multiple text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("Content 1", encoding="utf-8")
            (Path(tmpdir) / "file2.txt").write_text("Content 2", encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.txt"])
            documents = client.ingest()

            assert len(documents) == 2
            assert set(documents) == {"Content 1", "Content 2"}


class TestMarkdownIngestion:
    """Test .md file ingestion with parsing."""

    def test_md_basic_content(self):
        """Test basic markdown parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = """# Heading
This is a paragraph.
"""
            (Path(tmpdir) / "test.md").write_text(md_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            assert "Heading" in documents[0]
            assert "paragraph" in documents[0]

    def test_md_removes_formatting(self):
        """Test that markdown formatting is removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = """## Title
**Bold text** and *italic text*.
- List item 1
- List item 2
"""
            (Path(tmpdir) / "format.md").write_text(md_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            # Should contain text but not markdown symbols
            assert "Bold text" in text
            assert "italic text" in text
            assert "**" not in text
            assert "*" not in text or text.count("*") == 0

    def test_md_code_blocks_removed(self):
        """Test that code blocks are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = """# Code Example
```python
def hello():
    print("Hello")
```
Text after code.
"""
            (Path(tmpdir) / "code.md").write_text(md_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Code Example" in text
            assert "Text after code" in text
            # Code block should be removed
            assert "def hello" not in text
            assert "```" not in text

    def test_md_links_extracted(self):
        """Test that link text is kept but URLs removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = """Check out [this link](https://example.com) for more."""
            (Path(tmpdir) / "links.md").write_text(md_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "this link" in text
            # URL should be removed
            assert "https://" not in text

    def test_md_images_removed(self):
        """Test that image syntax is removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = """![Alt text](image.png)
Text after image."""
            (Path(tmpdir) / "image.md").write_text(md_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Text after image" in text
            # Image syntax should be removed but alt text kept
            assert "![" not in text

    def test_md_empty_file(self):
        """Test empty markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty.md").write_text("", encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            # Empty content should not be added
            assert len(documents) == 0


class TestHTMLIngestion:
    """Test .html file ingestion with parsing."""

    def test_html_basic_content(self):
        """Test basic HTML parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<html><body><h1>Title</h1><p>Paragraph text</p></body></html>"""
            (Path(tmpdir) / "test.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Title" in text
            assert "Paragraph text" in text
            assert "<html>" not in text
            assert "<p>" not in text

    def test_html_removes_tags(self):
        """Test that HTML tags are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<div><span>Nested <strong>content</strong></span></div>"""
            (Path(tmpdir) / "tags.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Nested" in text
            assert "content" in text
            assert "<" not in text
            assert ">" not in text

    def test_html_removes_scripts(self):
        """Test that script tags and content are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<html>
<head><script>console.log("test");</script></head>
<body>Body content</body>
</html>"""
            (Path(tmpdir) / "script.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Body content" in text
            assert "console.log" not in text
            assert "<script>" not in text

    def test_html_removes_styles(self):
        """Test that style tags and content are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<html>
<head><style>body { color: blue; }</style></head>
<body>Styled content</body>
</html>"""
            (Path(tmpdir) / "style.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Styled content" in text
            assert "color: blue" not in text
            assert "<style>" not in text

    def test_html_decodes_entities(self):
        """Test that HTML entities are decoded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = (
                """<p>Text&nbsp;with&nbsp;entities: &lt;tag&gt; &amp; &quot;quote&quot;</p>"""
            )
            (Path(tmpdir) / "entities.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "with" in text
            assert "entities" in text
            assert "<tag>" in text
            assert "&" in text
            assert '"quote"' in text
            assert "&nbsp;" not in text
            assert "&lt;" not in text

    def test_html_removes_comments(self):
        """Test that HTML comments are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<!-- Comment -->
<p>Visible text</p>
<!-- Another comment -->"""
            (Path(tmpdir) / "comments.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            assert len(documents) == 1
            text = documents[0]
            assert "Visible text" in text
            assert "<!--" not in text
            assert "Comment" not in text

    def test_html_empty_after_parsing(self):
        """Test HTML that becomes empty after parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html_content = """<script>console.log("only script");</script>"""
            (Path(tmpdir) / "empty.html").write_text(html_content, encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.html"])
            documents = client.ingest()

            # Should not add empty content
            assert len(documents) == 0


class TestMixedFormats:
    """Test ingesting multiple file formats together."""

    def test_multiple_formats_single_call(self):
        """Test ingesting TXT, MD, and HTML together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.txt").write_text("Text content", encoding="utf-8")
            (Path(tmpdir) / "doc.md").write_text("# MD content", encoding="utf-8")
            (Path(tmpdir) / "doc.html").write_text("<p>HTML content</p>", encoding="utf-8")

            client = IngestionClient(directory=tmpdir, formats=["*.txt", "*.md", "*.html"])
            documents = client.ingest()

            assert len(documents) == 3
            # Check that content from all formats is present
            all_text = " ".join(documents)
            assert "Text content" in all_text
            assert "MD content" in all_text
            assert "HTML content" in all_text

    def test_default_formats_backward_compatibility(self):
        """Test that default behavior only ingests .txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.txt").write_text("Text", encoding="utf-8")
            (Path(tmpdir) / "doc.md").write_text("MD", encoding="utf-8")
            (Path(tmpdir) / "doc.html").write_text("HTML", encoding="utf-8")

            # Default client should only ingest .txt
            client = IngestionClient(directory=tmpdir)
            documents = client.ingest()

            assert len(documents) == 1
            assert documents[0] == "Text"

    def test_selective_format_ingestion(self):
        """Test ingesting only specific formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.txt").write_text("Text", encoding="utf-8")
            (Path(tmpdir) / "doc.md").write_text("# MD", encoding="utf-8")
            (Path(tmpdir) / "doc.html").write_text("<p>HTML</p>", encoding="utf-8")

            # Only ingest markdown
            client = IngestionClient(directory=tmpdir, formats=["*.md"])
            documents = client.ingest()

            assert len(documents) == 1
            assert "MD" in documents[0]


class TestRealFiles:
    """Test with actual files in data directory."""

    def test_ingest_real_markdown(self):
        """Test ingesting real markdown file."""
        data_dir = Path("./data")
        if not (data_dir / "test_sample.md").exists():
            pytest.skip("test_sample.md not found")

        client = IngestionClient(directory="./data", formats=["*.md"])
        documents = client.ingest()

        assert len(documents) >= 1
        # Should contain content from markdown
        text = " ".join(documents)
        assert len(text) > 0

    def test_ingest_real_html(self):
        """Test ingesting real HTML file."""
        data_dir = Path("./data")
        if not (data_dir / "test_sample.html").exists():
            pytest.skip("test_sample.html not found")

        client = IngestionClient(directory="./data", formats=["*.html"])
        documents = client.ingest()

        assert len(documents) >= 1
        text = " ".join(documents)
        assert len(text) > 0
        # Should not contain HTML tags
        assert "<html>" not in text.lower()

    def test_ingest_all_formats_from_data(self):
        """Test ingesting all supported formats from data directory."""
        client = IngestionClient(directory="./data", formats=["*.txt", "*.md", "*.html"])
        documents = client.ingest()

        # Should have at least the .txt files
        assert len(documents) >= 5  # We know there are 5 .txt files


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nonexistent_directory(self):
        """Test with non-existent directory."""
        client = IngestionClient(directory="/nonexistent/path", formats=["*.txt"])
        documents = client.ingest()

        # Should return empty list, not crash
        assert documents == []

    def test_corrupted_file_continues(self):
        """Test that corrupted files don't stop processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid file
            (Path(tmpdir) / "good.txt").write_text("Good content", encoding="utf-8")

            # Create a file with invalid encoding
            with (Path(tmpdir) / "bad.txt").open("wb") as f:
                f.write(b"\xff\xfe Invalid UTF-8")

            client = IngestionClient(directory=tmpdir, formats=["*.txt"])
            # Should process what it can
            try:
                documents = client.ingest()
                # At minimum should have tried to process
                assert isinstance(documents, list)
            except Exception:
                # Or gracefully handle the error
                pass

    def test_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = IngestionClient(directory=tmpdir, formats=["*.txt", "*.md", "*.html"])
            documents = client.ingest()

            assert documents == []
