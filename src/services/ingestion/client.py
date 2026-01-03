import re
from pathlib import Path


class IngestionClient:
    """
    Document ingestion client supporting multiple file formats.

    Supported formats:
    - .txt: Plain text files
    - .md: Markdown files (strips formatting)
    - .html: HTML files (extracts text content)
    - .pdf: PDF files (requires PyPDF2, optional)
    """

    def __init__(self, directory: str = "./data", formats: list[str] | None = None):
        self.directory = Path(directory)
        # Default to .txt only for backward compatibility
        self.formats = formats or ["*.txt"]

    def _parse_html(self, content: str) -> str:
        """Extract text content from HTML, removing tags and scripts."""
        # Remove script and style tags with their content
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Decode HTML entities
        content = content.replace("&nbsp;", " ")
        content = content.replace("&lt;", "<")
        content = content.replace("&gt;", ">")
        content = content.replace("&amp;", "&")
        content = content.replace("&quot;", '"')

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content)
        return content.strip()

    def _parse_markdown(self, content: str) -> str:
        """Extract text from Markdown, removing formatting symbols."""
        # Remove code blocks
        content = re.sub(r"```[\s\S]*?```", "", content)
        content = re.sub(r"`[^`]+`", "", content)

        # Remove images
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", content)

        # Remove links but keep text
        content = re.sub(r"\[([^\]]*)\]\([^\)]+\)", r"\1", content)

        # Remove headers #
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)

        # Remove bold/italic
        content = re.sub(r"\*\*([^\*]+)\*\*", r"\1", content)
        content = re.sub(r"\*([^\*]+)\*", r"\1", content)
        content = re.sub(r"__([^_]+)__", r"\1", content)
        content = re.sub(r"_([^_]+)_", r"\1", content)

        # Remove list markers
        content = re.sub(r"^[\s]*[-\*\+]\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"^[\s]*\d+\.\s+", "", content, flags=re.MULTILINE)

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def _parse_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file (requires PyPDF2)."""
        try:
            import PyPDF2

            text_parts = []
            with file_path.open("rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except ImportError:
            # PyPDF2 not installed, return empty
            return f"[PDF support requires PyPDF2: {file_path.name}]"
        except Exception as e:
            return f"[Error reading PDF {file_path.name}: {str(e)}]"

    def ingest(self) -> list[str]:
        """
        Reads files from the directory based on configured formats.
        Supports .txt, .md, .html, and .pdf (if PyPDF2 installed).

        Returns:
            List of document contents as strings.
        """
        documents = []

        for pattern in self.formats:
            for file_path in self.directory.glob(pattern):
                try:
                    suffix = file_path.suffix.lower()

                    if suffix == ".txt":
                        with file_path.open("r", encoding="utf-8") as f:
                            documents.append(f.read())

                    elif suffix == ".md":
                        with file_path.open("r", encoding="utf-8") as f:
                            content = f.read()
                            parsed = self._parse_markdown(content)
                            if parsed:
                                documents.append(parsed)

                    elif suffix in [".html", ".htm"]:
                        with file_path.open("r", encoding="utf-8") as f:
                            content = f.read()
                            parsed = self._parse_html(content)
                            if parsed:
                                documents.append(parsed)

                    elif suffix == ".pdf":
                        parsed = self._parse_pdf(file_path)
                        if parsed:
                            documents.append(parsed)

                except Exception as e:
                    # Log error but continue processing other files
                    print(f"Warning: Could not process {file_path}: {e}")
                    continue

        return documents
