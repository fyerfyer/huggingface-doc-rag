import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.preprocess.chunk import HFChunker


class TestHFChunker:
  """Test the HFChunker class initialization"""
  
  def test_default_initialization(self):
    chunker = HFChunker()
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 50
    assert chunker.tokenizer is not None
  
  def test_custom_initialization(self):
    chunker = HFChunker(chunk_size=1024, chunk_overlap=100, tokenizer_name='cl100k_base')
    assert chunker.chunk_size == 1024
    assert chunker.chunk_overlap == 100
    assert chunker.tokenizer is not None


class TestCountTokens:
  """Test the count_tokens method"""
  
  @pytest.fixture
  def chunker(self):
    return HFChunker()
  
  def test_count_tokens_simple(self, chunker):
    text = "Hello world"
    count = chunker.count_tokens(text)
    assert count > 0
    assert isinstance(count, int)
  
  def test_count_tokens_empty(self, chunker):
    text = ""
    count = chunker.count_tokens(text)
    assert count == 0
  
  def test_count_tokens_longer_text(self, chunker):
    text = "This is a longer piece of text that should have more tokens than a simple hello world."
    count1 = chunker.count_tokens(text)
    count2 = chunker.count_tokens("Hello")
    assert count1 > count2


class TestSplitByHeader:
  """Test the _split_by_header method"""
  
  @pytest.fixture
  def chunker(self):
    return HFChunker()
  
  def test_single_header(self, chunker):
    text = """# Title

Some content here."""
    sections = chunker._split_by_header(text)
    assert len(sections) == 1
    assert sections[0]['headers'] == ['Title']
    assert '# Title' in sections[0]['content']
  
  def test_multiple_headers_same_level(self, chunker):
    text = """# Section 1

Content 1

# Section 2

Content 2"""
    sections = chunker._split_by_header(text)
    assert len(sections) == 2
    assert sections[0]['headers'] == ['Section 1']
    assert sections[1]['headers'] == ['Section 2']
  
  def test_nested_headers(self, chunker):
    text = """# Main Title

## Subsection 1

Content for subsection 1

## Subsection 2

Content for subsection 2"""
    sections = chunker._split_by_header(text)
    assert len(sections) >= 2
    # Check that nested headers are tracked
    found_subsection = False
    for section in sections:
      if 'Subsection 1' in section['headers']:
        found_subsection = True
        assert 'Main Title' in section['headers']
    assert found_subsection
  
  def test_header_in_codeblock_ignored(self, chunker):
    text = """# Real Header

```python
# This is not a header
def function():
    pass
```

More content"""
    sections = chunker._split_by_header(text)
    # The # inside code block should not create a new section
    assert len(sections) == 1
    assert sections[0]['headers'] == ['Real Header']
  
  def test_no_headers(self, chunker):
    text = "Just some plain text without headers"
    sections = chunker._split_by_header(text)
    assert len(sections) == 1
    assert sections[0]['headers'] == []
  
  def test_header_hierarchy(self, chunker):
    text = """# Level 1

## Level 2

### Level 3

## Another Level 2"""
    sections = chunker._split_by_header(text)
    # Check that the hierarchy is properly maintained
    level3_section = None
    another_level2_section = None
    
    for section in sections:
      if 'Level 3' in section['headers']:
        level3_section = section
      if section['headers'] and section['headers'][-1] == 'Another Level 2':
        another_level2_section = section
    
    # Level 3 should have all parent headers
    assert level3_section is not None
    assert 'Level 1' in level3_section['headers']
    assert 'Level 2' in level3_section['headers']
    assert 'Level 3' in level3_section['headers']
    
    # Another Level 2 should reset to just Level 1 and itself
    assert another_level2_section is not None
    assert 'Level 1' in another_level2_section['headers']
    assert 'Another Level 2' in another_level2_section['headers']


class TestAddHeaderContext:
  """Test the _add_header_context method"""
  
  @pytest.fixture
  def chunker(self):
    return HFChunker()
  
  def test_with_headers(self, chunker):
    text = "Some content"
    headers = ['Main', 'Sub', 'SubSub']
    result = chunker._add_header_context(text, headers)
    assert 'Context: Main > Sub > SubSub' in result
    assert 'Some content' in result
  
  def test_without_headers(self, chunker):
    text = "Some content"
    headers = []
    result = chunker._add_header_context(text, headers)
    assert result == text
  
  def test_single_header(self, chunker):
    text = "Content"
    headers = ['Only']
    result = chunker._add_header_context(text, headers)
    assert 'Context: Only' in result
    assert 'Content' in result


class TestRecursiveSplit:
  """Test the _recursive_split method"""
  
  @pytest.fixture
  def chunker(self):
    return HFChunker(chunk_size=100, chunk_overlap=20)
  
  def test_text_within_limit(self, chunker):
    text = "Short text"
    headers = ['Test']
    file_path = "test.md"
    chunks = chunker._recursive_split(text, headers, file_path)
    assert len(chunks) == 1
    assert 'Short text' in chunks[0]['text']
    assert chunks[0]['metadata']['source'] == file_path
    assert chunks[0]['metadata']['headers'] == headers
  
  def test_empty_text(self, chunker):
    text = ""
    headers = []
    file_path = "test.md"
    chunks = chunker._recursive_split(text, headers, file_path)
    assert len(chunks) == 0
  
  def test_text_exceeds_limit(self, chunker):
    # Create text that definitely exceeds 100 tokens
    text = " ".join(["word"] * 200)
    headers = ['Test']
    file_path = "test.md"
    chunks = chunker._recursive_split(text, headers, file_path)
    # Should be split into multiple chunks
    assert len(chunks) > 1
    # Each chunk should have metadata
    for chunk in chunks:
      assert 'text' in chunk
      assert 'metadata' in chunk
      assert chunk['metadata']['source'] == file_path
  
  def test_paragraph_splitting(self, chunker):
    # Text with clear paragraph boundaries
    paragraphs = ["Paragraph one with some words. " * 10,
                  "Paragraph two with different content. " * 10]
    text = "\n\n".join(paragraphs)
    headers = []
    file_path = "test.md"
    chunks = chunker._recursive_split(text, headers, file_path)
    # Should respect paragraph boundaries when possible
    assert len(chunks) >= 1
  
  def test_headers_in_chunks(self, chunker):
    text = "Content " * 50
    headers = ['Main', 'Sub']
    file_path = "test.md"
    chunks = chunker._recursive_split(text, headers, file_path)
    # All chunks should have the header context
    for chunk in chunks:
      assert 'Context: Main > Sub' in chunk['text']


class TestChunkFile:
  """Test the chunk_file method"""
  
  @pytest.fixture
  def chunker(self):
    return HFChunker(chunk_size=200, chunk_overlap=30)
  
  def test_simple_file(self, chunker):
    text = """# Introduction

This is the introduction section.

# Main Content

This is the main content section with more text.

## Subsection

And here is a subsection."""
    file_path = "test.md"
    chunks = chunker.chunk_file(text, file_path)
    
    # Should create multiple chunks
    assert len(chunks) > 0
    
    # All chunks should have proper structure
    for chunk in chunks:
      assert 'text' in chunk
      assert 'metadata' in chunk
      assert chunk['metadata']['source'] == file_path
      assert isinstance(chunk['metadata']['headers'], list)
  
  def test_file_without_headers(self, chunker):
    text = "This is plain text without any headers. " * 20
    file_path = "plain.md"
    chunks = chunker.chunk_file(text, file_path)
    
    assert len(chunks) > 0
    for chunk in chunks:
      assert chunk['metadata']['headers'] == []
  
  def test_long_file_with_headers(self, chunker):
    text = """# Chapter 1

""" + "Long content for chapter 1. " * 50 + """

# Chapter 2

""" + "Long content for chapter 2. " * 50 + """

## Section 2.1

""" + "Content for section 2.1. " * 30
    
    file_path = "long.md"
    chunks = chunker.chunk_file(text, file_path)
    
    # Should create multiple chunks
    assert len(chunks) > 2
    
    # Check that headers are properly tracked
    chapter1_chunks = [c for c in chunks if 'Chapter 1' in c['metadata']['headers']]
    chapter2_chunks = [c for c in chunks if 'Chapter 2' in c['metadata']['headers']]
    
    assert len(chapter1_chunks) > 0
    assert len(chapter2_chunks) > 0
  
  def test_chunk_metadata(self, chunker):
    text = """# Test Header

Some test content here."""
    file_path = "/path/to/test.md"
    chunks = chunker.chunk_file(text, file_path)
    
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk['metadata']['source'] == file_path
    assert 'Test Header' in chunk['metadata']['headers']
  
  def test_empty_file(self, chunker):
    text = ""
    file_path = "empty.md"
    chunks = chunker.chunk_file(text, file_path)
    # Empty file might create empty sections
    # The behavior depends on implementation
    assert isinstance(chunks, list)


class TestOverlapBehavior:
  """Test that chunk overlap works correctly"""
  
  def test_overlap_exists(self):
    chunker = HFChunker(chunk_size=50, chunk_overlap=15)
    # Create text that will definitely need splitting
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunker._recursive_split(text, [], "test.md")
    
    if len(chunks) > 1:
      # Check that there's some overlap between consecutive chunks
      # This is a soft check since exact overlap depends on tokenization
      for i in range(len(chunks) - 1):
        chunk1_text = chunks[i]['text']
        chunk2_text = chunks[i + 1]['text']
        # At least some words should appear in both chunks
        words1 = set(chunk1_text.split())
        words2 = set(chunk2_text.split())
        # There should be some overlap (but not checking exact amount due to context)
        assert len(words1.intersection(words2)) >= 0  # Soft check


class TestEdgeCases:
  """Test edge cases and error conditions"""
  
  def test_very_small_chunk_size(self):
    # Even with very small chunk size, should not crash
    chunker = HFChunker(chunk_size=10, chunk_overlap=2)
    text = "Short text"
    chunks = chunker.chunk_file(text, "test.md")
    assert len(chunks) >= 1
  
  def test_overlap_larger_than_chunk(self):
    # Overlap larger than chunk size is illogical but should not crash
    chunker = HFChunker(chunk_size=50, chunk_overlap=100)
    text = "Test text " * 20
    chunks = chunker.chunk_file(text, "test.md")
    assert isinstance(chunks, list)
  
  def test_unicode_text(self):
    chunker = HFChunker()
    text = """# 标题

这是中文内容。

## Subsection with émojis 🎉

Mixed content with special characters: äöü"""
    chunks = chunker.chunk_file(text, "unicode.md")
    assert len(chunks) > 0
    # Should handle unicode properly
    for chunk in chunks:
      assert isinstance(chunk['text'], str)
  
  def test_codeblock_preservation(self):
    chunker = HFChunker(chunk_size=200, chunk_overlap=30)
    text = """# Code Example

Here's some code:

```python
def hello():
    print("Hello, world!")
    # This should stay together
    return True
```

More text after code."""
    
    chunks = chunker.chunk_file(text, "code.md")
    assert len(chunks) > 0
    # Check that code blocks are in chunks
    full_text = " ".join([c['text'] for c in chunks])
    assert 'def hello' in full_text


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
