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
IMG_DIV_RE = re.compile(r'<div[^>]*>\s*<img\s+src="([^"]+)"[^>]*/>\s*</div>',re.IGNORECASE)                                                                                               
           
TIP_TAG = re.compile(r"<tip>\s*(.*?)\s*</tip>", flags=re.DOTALL | re.IGNORECASE)
NOTE_TAG = re.compile(r"<note>\s*(.*?)\s*</note>", flags=re.DOTALL | re.IGNORECASE)
WARNING_TAG = re.compile(r"<warning>\s*(.*?)\s*</warning>", flags=re.DOTALL | re.IGNORECASE)
SUP_TAG = re.compile(r"<sup>(.*?)</sup>", flags=re.IGNORECASE)

def main():
  parser = argparse.ArgumentParser(description="Preprocess HuggingFace document")
  parser.add_argument('--root', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()

  root_path = Path(args.root).resolve()
  output_path = Path(args.output)

  all_unsolved_links = []
  
  with open(output_path, 'w', encoding='utf-8') as f_out:
    for md_path in root_path.rglob('*.md'):
      if any(part.startswith('.') for part in md_path.parts) or "contributing" in md_path.name:
        continue
      
      print(f"Processing: {md_path.relative_to(root_path)}")
      
      with open(md_path, 'r', encoding='utf-8') as f_in:
        text = f_in.read()
      
      # Clean the markdown
      cleaned_text, unsolved_links = clean_hf_markdown(text, str(md_path))
      
      # Record unsolved links
      if unsolved_links:
        all_unsolved_links.append((str(md_path.relative_to(root_path)), unsolved_links))
      
      # Write cleaned content
      f_out.write(f"\n{'='*80}\n")
      f_out.write(f"File: {md_path.relative_to(root_path)}\n")
      f_out.write(f"{'='*80}\n\n")
      f_out.write(cleaned_text)
      f_out.write("\n\n")
  
  if all_unsolved_links:
    print("\n⚠️  Unsolved links found:")
    for file_path, links in all_unsolved_links:
      print(f"  {file_path}:")
      for link in links:
        print(f"    - {link}")
  
  print(f"\nProcessing complete. Output saved to: {output_path}")

def clean_hf_markdown(text: str, file_path: str) -> Tuple[str, List[str]]:
  text = _strip_license(text)
  text = _convert_hfoptions(text)
  text, unsolved_links = _patch_relative_links(text, file_path)
  text = _handle_html_images(text, file_path)
  text = _convert_html_fragments(text)
  text = _normalize_whitespace(text)
  return text, unsolved_links

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


if __name__ == "__main__":
  main()
