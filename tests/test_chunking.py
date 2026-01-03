"""
Unit tests for chunking services with focus on boundary conditions.
"""

import pytest
from src.services.chunking.client import ChunkingClient, HeadingAwareChunker


class TestChunkingClient:
    """Test ChunkingClient with various boundary conditions."""

    def test_basic_chunking_no_overlap(self):
        """Test basic chunking without overlap."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=0)
        text = "0123456789abcdefghij"
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 2
        assert chunks[0] == "0123456789"
        assert chunks[1] == "abcdefghij"

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdefghij"
        chunks = chunker.chunk(text)
        
        # step = 10 - 3 = 7
        # chunk 1: [0:10] = "0123456789"
        # chunk 2: [7:17] = "789abcdefg"
        # chunk 3: [14:24] = "efghij" (remaining)
        assert len(chunks) == 3
        assert chunks[0] == "0123456789"
        assert chunks[1] == "789abcdefg"
        assert chunks[2] == "efghij"

    def test_overlap_equals_chunk_size(self):
        """Test when overlap equals chunk size (edge case)."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=10)
        text = "0123456789abcdefghij"
        chunks = chunker.chunk(text)
        
        # step = 10 - 10 = 0, fallback to chunk_size
        # Should behave like no overlap
        assert len(chunks) == 2

    def test_overlap_greater_than_chunk_size(self):
        """Test when overlap is greater than chunk size (invalid config)."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=15)
        text = "0123456789abcdefghij"
        chunks = chunker.chunk(text)
        
        # step = 10 - 15 = -5, fallback to chunk_size
        assert len(chunks) == 2

    def test_empty_document(self):
        """Test with empty document."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=2)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_single_string_input(self):
        """Test with single string input."""
        chunker = ChunkingClient(chunk_size=5, chunk_overlap=0)
        text = "hello world"
        chunks = chunker.chunk(text)
        
        # Character-based chunking: [0:5]="hello", [5:10]=" worl", [10:15]="d"
        assert len(chunks) == 3
        assert chunks[0] == "hello"
        assert chunks[1] == "worl"  # Space stripped, only 4 chars remain

    def test_list_of_strings_input(self):
        """Test with list of strings input."""
        chunker = ChunkingClient(chunk_size=5, chunk_overlap=0)
        texts = ["hello", "world"]
        chunks = chunker.chunk(texts)
        
        assert len(chunks) == 2
        assert "hello" in chunks
        assert "world" in chunks

    def test_document_smaller_than_chunk_size(self):
        """Test when document is smaller than chunk size."""
        chunker = ChunkingClient(chunk_size=100, chunk_overlap=10)
        text = "short"
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == "short"

    def test_exact_chunk_size_boundary(self):
        """Test when document length is exactly chunk size."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=0)
        text = "0123456789"
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == "0123456789"

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped from chunks."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=0)
        text = "  hello    world  "
        chunks = chunker.chunk(text)
        
        # Whitespace should be stripped
        assert all(chunk == chunk.strip() for chunk in chunks)

    def test_mixed_empty_and_valid_documents(self):
        """Test with mix of empty and valid documents."""
        chunker = ChunkingClient(chunk_size=5, chunk_overlap=0)
        texts = ["", "hello", "   ", "world"]
        chunks = chunker.chunk(texts)
        
        # Only non-empty documents should produce chunks
        assert len(chunks) == 2
        assert "hello" in chunks
        assert "world" in chunks

    def test_large_overlap_with_small_text(self):
        """Test large overlap with small text."""
        chunker = ChunkingClient(chunk_size=20, chunk_overlap=15)
        text = "hello world"
        chunks = chunker.chunk(text)
        
        # step = 20 - 15 = 5
        # chunk 1: [0:20] = "hello world" (all 11 chars)
        # chunk 2: [5:25] = " world" (6 chars, stripped)
        # chunk 3: [10:30] = "d" (1 char)
        assert len(chunks) == 3
        assert chunks[0] == "hello world"
        assert "world" in chunks[1]
        assert chunks[2] == "d"


class TestHeadingAwareChunker:
    """Test HeadingAwareChunker with markdown heading boundaries."""

    def test_basic_heading_split(self):
        """Test basic splitting on headings."""
        chunker = HeadingAwareChunker(chunk_size=100, chunk_overlap=0)
        text = "# Title\nContent1\n## Section\nContent2"
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 2
        # Each chunk should preserve heading context
        assert any("Title" in chunk for chunk in chunks)

    def test_empty_document(self):
        """Test with empty document."""
        chunker = HeadingAwareChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_no_headings(self):
        """Test plain text without headings."""
        chunker = HeadingAwareChunker(chunk_size=20, chunk_overlap=0)
        text = "This is plain text without any markdown headings."
        chunks = chunker.chunk(text)
        
        # Should still chunk the text
        assert len(chunks) >= 1

    def test_nested_headings(self):
        """Test with nested heading structure."""
        chunker = HeadingAwareChunker(chunk_size=50, chunk_overlap=0)
        text = """# H1
Content1
## H2
Content2
### H3
Content3"""
        chunks = chunker.chunk(text)
        
        # Should preserve heading hierarchy
        assert len(chunks) >= 3

    def test_chunk_size_smaller_than_content(self):
        """Test when content exceeds chunk size."""
        chunker = HeadingAwareChunker(chunk_size=10, chunk_overlap=0)
        text = "# Title\n" + "a" * 100
        chunks = chunker.chunk(text)
        
        # Should split large content into multiple chunks
        assert len(chunks) > 1
        # All chunks should have the heading prefix
        assert all("Title" in chunk for chunk in chunks)

    def test_heading_prefix_metadata(self):
        """Test that headings are properly prefixed to chunks."""
        chunker = HeadingAwareChunker(chunk_size=100, chunk_overlap=0)
        text = "## Section\nSome content here"
        chunks = chunker.chunk(text)
        
        # Chunk should contain heading
        assert len(chunks) >= 1
        assert "Section" in chunks[0]

    def test_multiple_documents(self):
        """Test with list of documents."""
        chunker = HeadingAwareChunker(chunk_size=50, chunk_overlap=0)
        texts = ["# Doc1\nContent1", "# Doc2\nContent2"]
        chunks = chunker.chunk(texts)
        
        assert len(chunks) >= 2
        assert any("Doc1" in chunk for chunk in chunks)
        assert any("Doc2" in chunk for chunk in chunks)

    def test_whitespace_sections(self):
        """Test handling of sections with only whitespace."""
        chunker = HeadingAwareChunker(chunk_size=50, chunk_overlap=0)
        text = "# Title\n\n\n## Section\n   \n### Subsection\nContent"
        chunks = chunker.chunk(text)
        
        # Should skip empty sections
        assert all(chunk.strip() for chunk in chunks)

    def test_overlap_with_headings(self):
        """Test overlap behavior with heading-aware chunking."""
        chunker = HeadingAwareChunker(chunk_size=30, chunk_overlap=10)
        text = "# Title\n" + "word " * 50
        chunks = chunker.chunk(text)
        
        # Should create overlapping chunks with heading preserved
        assert len(chunks) > 1
        assert all("Title" in chunk for chunk in chunks)

    def test_exact_chunk_size_with_heading(self):
        """Test when section is exactly chunk size."""
        chunker = HeadingAwareChunker(chunk_size=30, chunk_overlap=0)
        text = "## Test\n" + "x" * 20
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1

    def test_very_long_heading_name(self):
        """Test with very long heading names."""
        chunker = HeadingAwareChunker(chunk_size=50, chunk_overlap=0)
        long_heading = "a" * 100
        text = f"# {long_heading}\nContent"
        chunks = chunker.chunk(text)
        
        # Should handle long headings gracefully
        assert len(chunks) >= 1


class TestChunkingBoundaryEdgeCases:
    """Test edge cases and boundary conditions across chunking implementations."""

    def test_zero_chunk_size(self):
        """Test with invalid chunk size of zero."""
        with pytest.raises(Exception):
            chunker = ChunkingClient(chunk_size=0, chunk_overlap=0)
            chunker.chunk("test")

    def test_negative_chunk_size(self):
        """Test with negative chunk size (undefined behavior)."""
        # Note: ChunkingClient doesn't validate chunk_size, so negative values
        # will cause undefined behavior (likely infinite loop or no chunks)
        chunker = ChunkingClient(chunk_size=-10, chunk_overlap=0)
        text = "test"
        # Just verify it doesn't crash catastrophically
        try:
            chunks = chunker.chunk(text)
            # If it returns, it should return a list (possibly empty)
            assert isinstance(chunks, list)
        except (ValueError, IndexError):
            # These exceptions are acceptable for invalid config
            pass

    def test_negative_overlap(self):
        """Test with negative overlap."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=-5)
        text = "0123456789abcdefghij"
        # Should handle gracefully, likely treating as step > chunk_size
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_unicode_characters(self):
        """Test with unicode characters."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=2)
        text = "Hello 世界 🌍 test"
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        # Should preserve unicode characters
        assert any("世界" in chunk or "🌍" in chunk for chunk in chunks)

    def test_newline_handling(self):
        """Test proper handling of newlines."""
        chunker = ChunkingClient(chunk_size=20, chunk_overlap=0)
        text = "line1\nline2\nline3\nline4"
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        # Newlines should be preserved in chunks
        assert any("\n" in chunk for chunk in chunks)

    def test_single_character_chunks(self):
        """Test with chunk size of 1."""
        chunker = ChunkingClient(chunk_size=1, chunk_overlap=0)
        text = "abc"
        chunks = chunker.chunk(text)
        
        # Should create individual character chunks
        assert len(chunks) == 3
        assert chunks == ["a", "b", "c"]

    def test_all_whitespace_document(self):
        """Test document containing only whitespace."""
        chunker = ChunkingClient(chunk_size=10, chunk_overlap=0)
        text = "     \n\n\t\t\r\r\n     "
        chunks = chunker.chunk(text)
        
        # Should return empty list or skip whitespace-only chunks
        assert len(chunks) == 0
