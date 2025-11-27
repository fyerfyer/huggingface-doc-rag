import argparse
from pathlib import Path
import re 
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse

LICENSE_RE = re.compile(r"^<!--\s*Copyright[\s\S]*?-->\s*", re.I)
HFO_BEGIN = re.compile(r"<hfoptions.*?>", re.I)
HFO_OPTION = re.compile(r"<hfoption\s+id=\"([^\"]+)\".*?>", re.I)
HFO_END = re.compile(r"</hfoptions>", re.I)
MARKDOWN_LINK_RE = re.compile(r"(\[.*?\])\((.*?)\)")  
CODE_LINK_RE = re.compile(r"\[(.*?)\]")
IMG_DIV_RE = re.compile(r'<div[^>]*>\s*<img\s+src="([^"]+)"[^>]*/>\s*</div>',re.IGNORECASE)                                                                                               
           
TIP_TAG = re.compile(r"<tip>\s*(.*?)\s*</tip>", flags=re.DOTALL | re.IGNORECASE)
NOTE_TAG = re.compile(r"<note>\s*(.*?)\s*</note>", flags=re.DOTALL | re.IGNORECASE)
WARNING_TAG = re.compile(r"<warning>\s*(.*?)\s*</warning>", flags=re.DOTALL | re.IGNORECASE)
SUP_TAG = re.compile(r"<sup>(.*?)</sup>", flags=re.IGNORECASE)

# 额外的 HTML 标签清理
IFRAME_RE = re.compile(r'<iframe[^>]*>.*?</iframe>', flags=re.DOTALL | re.IGNORECASE)
FIGURE_RE = re.compile(r'<figure[^>]*>(.*?)</figure>', flags=re.DOTALL | re.IGNORECASE)
IMG_TAG_RE = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*/>', flags=re.IGNORECASE)
TABLE_RE = re.compile(r'<table[^>]*>.*?</table>', flags=re.DOTALL | re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r'<!--.*?-->', flags=re.DOTALL)
STYLE_SCRIPT_RE = re.compile(r'<(style|script)[^>]*>.*?</\1>', flags=re.DOTALL | re.IGNORECASE)
SPECIAL_TOKEN_RE = re.compile(r"<\|([^|<>]+)\|>")

def main():
  parser = argparse.ArgumentParser(description="Preprocess HuggingFace document")
  parser.add_argument('--root', type=str, default='transformers-endocs')
  parser.add_argument('--output', type=str, default='preprocessed')
  args = parser.parse_args()

  root_path = Path(args.root).resolve()
  output_path = Path(args.output).resolve()

  all_unsolved_links = []
  
  output_path.mkdir(parents=True, exist_ok=True)
  
  for md_path in root_path.rglob('*.md'):
    if any(part.startswith('.') for part in md_path.parts) or "contributing" in md_path.name:
      continue
    
    print(f"Processing: {md_path.relative_to(root_path)}")
    
    with open(md_path, 'r', encoding='utf-8') as f_in:
      text = f_in.read()
    
    cleaned_text, unsolved_links = clean_hf_markdown(text, str(md_path))
    
    if unsolved_links:
      all_unsolved_links.append((str(md_path.relative_to(root_path)), unsolved_links))
    
    relative_path = md_path.relative_to(root_path)
    output_file_path = output_path / relative_path
    
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
      f_out.write(cleaned_text)
    
    print(f"  ✓ Saved to: {output_file_path.relative_to(output_path)}")
  
  if all_unsolved_links:
    print("\n! Unsolved links found:")
    for file_path, links in all_unsolved_links:
      print(f"  {file_path}:")
      for link in links:
        print(f"    - {link}")
  
  print(f"\nProcessing complete. Output saved to: {output_path}")

def clean_hf_markdown(text: str, file_path: str) -> Tuple[str, List[str]]:
  text = _strip_license(text)
  text = _sanitize_special_tokens(text)
  text = _convert_hfoptions(text)
  text, unsolved_links = _patch_relative_links(text, file_path)
  text = _handle_html_images(text, file_path)
  text = _convert_html_fragments(text)
  text = _remove_unwanted_html(text)
  text = _convert_markdown_code_references(text)
  text = _normalize_whitespace(text)
  return text, unsolved_links

def _sanitize_special_tokens(text: str) -> str:
  """Replace tiktoken-style special tokens (e.g. <|endoftext|>) with a
  safe, human-readable placeholder so downstream tokenizers won't raise
  on disallowed special tokens. We keep the inner token name for traceability.
  """
  return SPECIAL_TOKEN_RE.sub(lambda m: f"[SPECIAL_TOKEN:{m.group(1)}]", text)

def _strip_license(text: str) -> str:
  return LICENSE_RE.sub("", text)

def _normalize_whitespace(text: str) -> str:
  text = text.replace('\r\n', '\n')
  text = "\n".join(line.rstrip() for line in text.split("\n"))
  text = re.sub(r"\n{3,}", "\n\n", text) 
  return text 

def _convert_hfoptions(text: str) -> str:
  lines = text.split("\n")
  new_lines = []
  in_hfoptions_blocks = False 
  
  for line in lines:
    if HFO_BEGIN.search(line):
      in_hfoptions_blocks = True 
      continue
      
    if HFO_END.search(line):
      in_hfoptions_blocks = False
      continue
  
    if in_hfoptions_blocks:
      match = HFO_OPTION.search(line)
      if match:
        option_id = match.group(1)
        new_lines.append(f"#### {option_id}")
        continue
      else:
        new_lines.append(line)
    else:
      new_lines.append(line)
    
  return '\n'.join(new_lines)

def _patch_relative_links(text: str, file_path: str, doc_base_url: str = "transformers-endocs") -> Tuple[str, List[str]]:
  doc_dir = Path(file_path).parent
  unsolved_links = []

  DOC_EXTENSIONS = {'.md', '.mdx'}
  ASSET_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.mp4', '.webp'}

  def replacer(match : re.Match) -> str:
    link_text = match.group(1)
    link_target = match.group(2).strip() 

    if link_target.startswith('https://') or link_target.startswith('http://'):
      # 对外部链接 URL,先原样返回
      return f"{link_text}({link_target})"

    parsed_target = urlparse(link_target)
    relative_path = parsed_target.path

    # 内部锚点无路径
    if not relative_path:
      return f"{link_text}({link_target})"
    
    abs_target_path = (doc_dir / relative_path).resolve()
    try:
      if abs_target_path.suffix in DOC_EXTENSIONS or abs_target_path.is_dir():
        # 去掉拓展名，语义化 URL
        doc_name = abs_target_path.stem if abs_target_path.suffix else abs_target_path.name
        new_target_path = doc_base_url.rstrip('/') + '/' + doc_name
        new_target = urlunparse(('', '', new_target_path, '', parsed_target.query, parsed_target.fragment))
        return f"{link_text}({new_target})"
      elif abs_target_path.suffix in ASSET_EXTENSIONS: 
        return f"Asset path: {abs_target_path}, Caption: {link_text}"
      else: 
        unsolved_links.append(link_target)
        return link_text.strip('[]')
    except (OSError, ValueError) as e:
      unsolved_links.append(link_target)
      return link_text.strip('[]')
  
  new_text = MARKDOWN_LINK_RE.sub(replacer, text)
  return new_text, unsolved_links

def _handle_html_images(text :str, file_path: str ) ->str:
  doc_dir = Path(file_path).parent
  
  def replacer(match : re.Match) -> str:
    img_src = match.group(1)
    if img_src.startswith('http'):
      asset_path = img_src
    else:
      asset_path = (doc_dir / img_src).resolve()
    
    caption = f"[Image: {Path(img_src).name}]"
    return f"Asset path: {asset_path}, Caption: {caption}"

  return IMG_DIV_RE.sub(replacer, text)

def _convert_html_fragments(text: str) -> str:
  text = TIP_TAG.sub(r"> **Tip**\n> \1", text)
  text = NOTE_TAG.sub(r"> **Note**\n> \1", text)
  text = WARNING_TAG.sub(r"> **Warning**\n> \1", text)
  text = SUP_TAG.sub(r"^\1^", text)
  return text 

def _remove_unwanted_html(text: str) -> str:
  text = IFRAME_RE.sub('[Interactive Demo - See online documentation]', text)
  text = STYLE_SCRIPT_RE.sub('', text)
  text = HTML_COMMENT_RE.sub('', text)
  
  def handle_figure(match):
    content = match.group(1)
    # 提取 figcaption
    figcaption_match = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', content, re.DOTALL | re.IGNORECASE)
    if figcaption_match:
      return f"\n{figcaption_match.group(1).strip()}\n"
    return content
  
  text = FIGURE_RE.sub(handle_figure, text)
  
  def handle_img(match):
    src = match.group(1)
    return f"[Image: {Path(src).name}]"
  
  text = IMG_TAG_RE.sub(handle_img, text)
  
  # TODO: 移除 HTML 表格（通常在 markdown 中已经有对应的表格）
  # 如果需要保留表格，可以尝试转换为 markdown 表格
  text = TABLE_RE.sub('[Table - See online documentation]', text)
  
  # 移除常见的 HTML 标签，保留内容
  text = re.sub(r'</?(?:div|span|p|section|article|header|footer|nav|aside|main)[^>]*>', '', text, flags=re.IGNORECASE)
  
  # <br> 换行符转为真正的换行
  text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
  
  # <hr> 分隔符转为 markdown
  text = re.sub(r'<hr\s*/?>', '\n---\n', text, flags=re.IGNORECASE)
  
  # 移除其他单个标签（如 <strong>, <em>, <b>, <i> 等，保留内容）
  text = re.sub(r'</?(?:strong|em|b|i|u|mark|small|del|ins|sub|code|kbd|var|samp)[^>]*>', '', text, flags=re.IGNORECASE)
  
  # 移除所有剩余的 HTML 标签
  text = re.sub(r'<[^>]+>', '', text)
  
  return text

def _convert_markdown_code_references(text: str) -> str: 
  return CODE_LINK_RE.sub(r"\1", text)

if __name__ == "__main__":
  main()
