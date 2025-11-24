import argparse
import json
from pathlib import Path
import re
from typing import Dict, List
import tiktoken

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
CODEBLOCK_RE = re.compile(r"^```")

class HFChunker:
  def __init__(self, chunk_size: int = 512, chunk_overlap : int = 50, tokenizer_name: str= 'cl100k_base'):
    self.chunk_size = chunk_size 
    self.chunk_overlap = chunk_overlap
    self.tokenizer = tiktoken.get_encoding(tokenizer_name)
  
  def count_tokens(self, text: str) -> int:
    return len(self.tokenizer.encode(text))
  
  def chunk_file(self, text: str, file_path: str) -> List[Dict]:
    sections = self._split_by_header(text)
    chunks = []
    for section in sections:
      header_path = section['headers']
      content = section['content']

      sub_chunks = self._recursive_split(content, header_path, file_path)
      chunks.extend(sub_chunks)
    
    return chunks

  def _split_by_header(self, text: str) -> List[Dict]:
    lines = text.split('\n')
    sections = []
    current_headers = []
    current_lines = []
    in_codeblock = False

    for line in lines:
      if CODEBLOCK_RE.match(line):
        in_codeblock = not in_codeblock
      
      match = HEADER_RE.match(line)
      if match and not in_codeblock:
        if current_lines: 
          sections.append({
            'headers': list(current_headers), 
            'content': '\n'.join(current_lines).strip()
          })
        
        level = len(match.group(1))
        title = match.group(2)
        current_headers = [h for i, h in enumerate(current_headers) if i < level - 1]
        current_headers.append(title)
        current_lines = [f"{'#' * level} {title}"]
      else:
        current_lines.append(line)
      
    if current_lines:
      sections.append({
        'headers': list(current_headers),
        'content': '\n'.join(current_lines).strip()
      })
    
    return sections
  
  def _add_header_context(self, text: str, headers: List[str]) -> str:
    if not headers:
      return text 
    headers_chain = ' > '.join(headers).strip()
    return f"Context: {headers_chain}\n\n{text}" 

  def _recursive_split(self, text: str, headers: List[str], file_path: str) -> List[Dict]:
    if not text:
      return []
    
    base_context = self._add_header_context("", headers)
    base_context_token = self.count_tokens(base_context)

    if self.count_tokens(text) + base_context_token <= self.chunk_size:
      return [{
        'text': self._add_header_context(text, headers),
        'metadata': {'source': file_path, 'headers': headers}
      }]
    
    separators = ['\n\n', '\n', '.', ' ']
    top_separator = None
    for sep in separators:
      if sep in text and sep != text:
        top_separator = sep
        break

    if top_separator is None:
      # 没找到合适的分隔符就使用字符级别拆分
      parts = list(text)
      top_separator = ''
    else:
      parts = text.split(top_separator)
    final_chunks = []
    current_chunk_parts = []
    current_chunk_tokens = base_context_token

    for part in parts:
      if not part:
        continue

      part_tokens = self.count_tokens(part)
      sep_tokens = self.count_tokens(top_separator) if (current_chunk_parts and top_separator) else 0
      
      # 如果添加当前 part 超限，保存当前状态并设置 overlap
      if current_chunk_tokens + sep_tokens + part_tokens > self.chunk_size:
        if current_chunk_parts:
          final_chunks.append({
            'text': self._add_header_context(top_separator.join(current_chunk_parts), headers),
            'metadata': {'source': file_path, 'headers': headers}
          })
          
          overlap_chunks = []
          overlap_tokens = base_context_token
          for p in reversed(current_chunk_parts):
            p_tokens = self.count_tokens(p) + (self.count_tokens(top_separator) if top_separator else 0)
            if overlap_tokens + p_tokens <= self.chunk_overlap:
              overlap_tokens += p_tokens
              overlap_chunks.insert(0, p)
            else:
              break
          
          current_chunk_parts = overlap_chunks
          joined = top_separator.join(current_chunk_parts) if top_separator else ''.join(current_chunk_parts)
          current_chunk_tokens = base_context_token + self.count_tokens(joined)
          if current_chunk_parts and top_separator:
            current_chunk_tokens += self.count_tokens(top_separator)
        
        # 如果当前 part 已经超限了，就递归拆分它
        if part_tokens + base_context_token > self.chunk_size:
          recursive_chunks = self._recursive_split(part, headers, file_path)
          final_chunks.extend(recursive_chunks)
          current_chunk_parts = []
          current_chunk_tokens = base_context_token
          continue
      
      current_chunk_parts.append(part)
      current_chunk_tokens += part_tokens
      if len(current_chunk_parts) > 1:
        current_chunk_tokens += self.count_tokens(top_separator)
    
    if current_chunk_parts:
      final_chunks.append({
        'text': self._add_header_context(top_separator.join(current_chunk_parts), headers),
        'metadata': {'source': file_path, 'headers': headers}
      })
    
    return final_chunks
    
def main():
  parser = argparse.ArgumentParser(description="chunk HuggingFace documentation for RAG")
  parser.add_argument('--processed_file_root', type=str, required=True)
  parser.add_argument('--chunk_size', type=int, required=True)
  parser.add_argument('--chunk_overlap', type=int, required=True)
  parser.add_argument('--output', type=str, required=True )
  args = parser.parse_args()

  root_path = Path(args.processed_file_root).resolve()
  output_path = Path(args.output).resolve()

  output_dir = output_path
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file_path = output_dir / 'chunks.json'

  chunker = HFChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
  all_chunks = []

  for md_path in root_path.rglob('*.md'):
    if any(part.startswith('.') for part in md_path.parts) or "contributing" in md_path.name:
      continue

    print(f"Processing: {md_path.relative_to(root_path)}")

    with open(md_path, 'r', encoding='utf-8') as f_in:
      text = f_in.read()

    file_chunks = chunker.chunk_file(text, str(md_path))
    all_chunks.extend(file_chunks)
    print(f"  ✓ Generated {len(file_chunks)} chunks.")
  
  print(f"\nTotal chunks generated: {len(all_chunks)}")
  # 将所有 chunks 写入一个json文件
  with open(output_file_path, 'w', encoding='utf-8') as f_out:
    for chunk in all_chunks:
      f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')

  print(f"✓ Processing complete. RAG dataset saved to: {output_file_path}")
  
if __name__ == "__main__":
  main()


