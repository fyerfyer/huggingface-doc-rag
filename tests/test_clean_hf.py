import pytest
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.preprocess.clean_hf import (
  _strip_license,
  _normalize_whitespace,
  _convert_hfoptions,
  _patch_relative_links,
  _handle_html_images,
  _convert_html_fragments,
  clean_hf_markdown
)


class TestStripLicense:
  """Test the _strip_license function"""
  
  def test_strip_standard_license(self):
    text = """<!-- Copyright 2023 The HuggingFace Team. All rights reserved. -->

# Title

Content here"""
    result = _strip_license(text)
    assert not result.startswith("<!--")
    assert "# Title" in result
    
  def test_no_license(self):
    text = "# Title\n\nContent"
    result = _strip_license(text)
    assert result == text
    
  def test_multiline_license(self):
    text = """<!-- Copyright 2023
    The HuggingFace Team
    All rights reserved
    -->
    
# Content"""
    result = _strip_license(text)
    assert not result.startswith("<!--")


class TestNormalizeWhitespace:
  """Test the _normalize_whitespace function"""
  
  def test_windows_line_endings(self):
    text = "Line 1\r\nLine 2\r\nLine 3"
    result = _normalize_whitespace(text)
    assert "\r\n" not in result
    assert "Line 1\nLine 2\nLine 3" in result
    
  def test_trailing_spaces(self):
    text = "Line 1   \nLine 2  \nLine 3"
    result = _normalize_whitespace(text)
    assert result == "Line 1\nLine 2\nLine 3"
    
  def test_multiple_blank_lines(self):
    text = "Line 1\n\n\n\nLine 2"
    result = _normalize_whitespace(text)
    assert result == "Line 1\n\nLine 2"
    
  def test_combined(self):
    text = "Line 1  \r\n\r\n\r\nLine 2   "
    result = _normalize_whitespace(text)
    assert result == "Line 1\n\nLine 2"


class TestConvertHfoptions:
  """Test the _convert_hfoptions function"""
  
  def test_standard_hfoptions(self):
    text = """<hfoptions id="install">
<hfoption id="PyTorch">
Install PyTorch
</hfoption>
<hfoption id="TensorFlow">
Install TensorFlow
</hfoption>
</hfoptions>"""
    result = _convert_hfoptions(text)
    assert "#### PyTorch" in result
    assert "#### TensorFlow" in result
    assert "<hfoptions" not in result
    assert "<hfoption" not in result
    
  def test_no_hfoptions(self):
    text = "Regular markdown content"
    result = _convert_hfoptions(text)
    assert result == text
    
  def test_empty_hfoptions(self):
    text = "<hfoptions>\n</hfoptions>"
    result = _convert_hfoptions(text)
    assert "<hfoptions" not in result


class TestPatchRelativeLinks:
  """Test the _patch_relative_links function"""
  
  @pytest.fixture
  def temp_doc_dir(self, tmp_path):
    """Create a temporary document directory structure"""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    
    (doc_dir / "test.md").write_text("# Test")
    (doc_dir / "other.md").write_text("# Other")
    (doc_dir / "image.png").write_text("fake image")
    
    return doc_dir
  
  def test_external_links(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "[Google](https://google.com) and [HF](http://huggingface.co)"
    result, unsolved = _patch_relative_links(text, str(test_file))
    assert "https://google.com" in result
    assert "http://huggingface.co" in result
    assert len(unsolved) == 0
    
  def test_internal_anchor(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "[Section](#section-1)"
    result, unsolved = _patch_relative_links(text, str(test_file))
    assert "[Section](#section-1)" in result
    assert len(unsolved) == 0
    
  def test_relative_doc_link(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "[Other Doc](./other.md)"
    result, unsolved = _patch_relative_links(text, str(test_file))
    assert "transformers-endocs/other" in result
    assert ".md" not in result  # Extension should be removed
    assert len(unsolved) == 0
    
  def test_image_link(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "[Image](./image.png)"
    result, unsolved = _patch_relative_links(text, str(test_file))
    assert "Asset path:" in result
    assert "image.png" in result
    assert len(unsolved) == 0
    
  def test_nonexistent_link(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "[Missing](./nonexistent.txt)"
    result, unsolved = _patch_relative_links(text, str(test_file))
    assert len(unsolved) == 1
    assert "./nonexistent.txt" in unsolved[0]


class TestHandleHtmlImages:
  @pytest.fixture
  def temp_doc_dir(self, tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "local.png").write_text("fake")
    return doc_dir
  
  def test_external_image(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = '<div><img src="https://example.com/image.png"/></div>'
    result = _handle_html_images(text, str(test_file))
    assert "Asset path: https://example.com/image.png" in result
    assert "Caption: [Image: image.png]" in result
    
  def test_local_image(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = '<div><img src="./local.png"/></div>'
    result = _handle_html_images(text, str(test_file))
    assert "Asset path:" in result
    assert "local.png" in result
    
  def test_no_html_images(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    text = "Regular markdown ![image](./image.png)"
    result = _handle_html_images(text, str(test_file))
    assert result == text


class TestConvertHtmlFragments:
  def test_tip_tag(self):
    text = "<tip>This is a helpful tip</tip>"
    result = _convert_html_fragments(text)
    assert "> **Tip**" in result
    assert "helpful tip" in result
    assert "<tip>" not in result
    
  def test_note_tag(self):
    text = "<note>This is an important note</note>"
    result = _convert_html_fragments(text)
    assert "> **Note**" in result
    assert "important note" in result
    
  def test_warning_tag(self):
    text = "<warning>This is a warning</warning>"
    result = _convert_html_fragments(text)
    assert "> **Warning**" in result
    assert "warning" in result
    
  def test_sup_tag(self):
    text = "Text<sup>1</sup> with superscript"
    result = _convert_html_fragments(text)
    assert "^1^" in result
    assert "<sup>" not in result
    
  def test_multiline_tags(self):
    text = """<tip>
This is a
multiline tip
</tip>"""
    result = _convert_html_fragments(text)
    assert "> **Tip**" in result
    assert "multiline tip" in result


class TestCleanHfMarkdown:
  @pytest.fixture
  def temp_doc_dir(self, tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "other.md").write_text("# Other")
    return doc_dir
  
  def test_full_cleaning(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    test_file.write_text("# Test")
    
    text = """<!-- Copyright 2023 -->

# Title

<tip>Helpful info</tip>

[Link](./other.md)

Line with spaces   \r\n\r\n\r\n

Another line"""
    
    result, unsolved = clean_hf_markdown(text, str(test_file))
    
    assert "Copyright" not in result
    assert "> **Tip**" in result
    assert "transformers-endocs" in result
    assert "\r\n" not in result
    assert "spaces   \n" not in result
    
  def test_with_unsolved_links(self, temp_doc_dir):
    test_file = temp_doc_dir / "test.md"
    test_file.write_text("# Test")
    
    text = "[Missing](./missing.txt)"
    result, unsolved = clean_hf_markdown(text, str(test_file))
    
    assert len(unsolved) > 0

if __name__ == "__main__":
  pytest.main([__file__, "-v"])