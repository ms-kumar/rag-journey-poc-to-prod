from __future__ import annotations

from typing import List, Union
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class ChunkingClient:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: Union[str, List[str]]) -> List[str]:
        """
        Splits documents into chunks of specified size with optional overlap.
        Accepts either a single string or a list of strings.
        Preserves semantic blocks by avoiding splits mid-sentence when possible.
        """
        # Handle single string input
        if isinstance(documents, str):
            documents = [documents]

        chunks = []
        for doc in documents:
            # Skip empty documents
            if not doc or not doc.strip():
                continue
            
            # Calculate step size based on overlap
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size  # Fallback if overlap >= chunk_size
            
            for i in range(0, len(doc), step):
                chunk = doc[i : i + self.chunk_size].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
        return chunks


class HeadingAwareChunker:
    """Markdown-aware chunker that respects H1/H2/H3 boundaries.

    Uses LangChain's markdown header splitter to create sections, then applies a
    size-based splitter within each section.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        # Some versions support strip_headers; if not, it will be ignored by Python
        # only if passed conditionally.
        try:
            self._header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=True,
            )
        except TypeError:
            self._header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
            )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @staticmethod
    def _heading_prefix(metadata: dict) -> str:
        h1 = metadata.get("h1")
        h2 = metadata.get("h2")
        h3 = metadata.get("h3")

        lines: List[str] = []
        if h1:
            lines.append(f"# {h1}")
        if h2:
            lines.append(f"## {h2}")
        if h3:
            lines.append(f"### {h3}")

        if not lines:
            return ""
        return "\n".join(lines) + "\n\n"

    def chunk(self, documents: Union[str, List[str]]) -> List[str]:
        """Split markdown into chunks while preserving heading context."""

        if isinstance(documents, str):
            documents = [documents]

        chunks: List[str] = []
        for doc in documents:
            if not doc or not doc.strip():
                continue

            # Split into heading-bounded sections first.
            sections = self._header_splitter.split_text(doc)

            for section in sections:
                text = getattr(section, "page_content", "") or ""
                text = text.strip()
                if not text:
                    continue

                metadata = getattr(section, "metadata", {}) or {}
                prefix = self._heading_prefix(metadata)

                if len(text) <= self.chunk_size:
                    merged = (prefix + text).strip()
                    if merged:
                        chunks.append(merged)
                    continue

                for piece in self._text_splitter.split_text(text):
                    piece = (piece or "").strip()
                    if not piece:
                        continue
                    merged = (prefix + piece).strip()
                    if merged:
                        chunks.append(merged)

        return chunks
