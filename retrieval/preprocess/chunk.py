import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Set
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
  
  def chunk_file(self, text: str, file_path: str, root_path: str = None) -> List[Dict]:
    # 计算相对路径
    if root_path:
      try:
        relative_path = Path(file_path).relative_to(Path(root_path))
        source_path = str(relative_path)
      except ValueError:
        # 如果无法计算相对路径，使用文件名
        source_path = Path(file_path).name
    else:
      source_path = Path(file_path).name
    
    sections = self._split_by_header(text)
    chunks = []
    for section in sections:
      header_path = section['headers']
      content = section['content']

      sub_chunks = self._recursive_split(content, header_path, source_path)
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
      
      # 检测超长行（通常是格式化不当的代码或数据）
      # 如果行太长，尝试在合适的位置截断或换行
      if len(line) > 500 and not in_codeblock:
        # 尝试在逗号、空格等处分割超长行
        if ', ' in line:
          parts = line.split(', ')
          # 每50个元素一行
          for i in range(0, len(parts), 50):
            current_lines.append(', '.join(parts[i:i+50]))
          continue
      
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
    
    # 增加更多语义边界
    separators = [
      '\n\n',      # 段落
      '\n',        # 行
      '. ',        # 句子（保留空格）
      '! ',        # 感叹号
      '? ',        # 问号
      '; ',        # 分号
      ', ',        # 逗号
      ' ',         # 空格
    ]
    
    top_separator = None
    for sep in separators:
      if sep in text:
        top_separator = sep
        break

    if top_separator is None:
      # 如果完全找不到分隔符，强制在固定长度处截断（保留完整性）
      # 这种情况很少见，通常是超长的 URL 或代码
      max_chars = self.chunk_size * 4  # 粗略估计字符数
      if len(text) > max_chars:
        return [{
          'text': self._add_header_context(text[:max_chars] + '...', headers),
          'metadata': {'source': file_path, 'headers': headers}
        }]
      else:
        # 文本虽长但没超限，直接返回
        return [{
          'text': self._add_header_context(text, headers),
          'metadata': {'source': file_path, 'headers': headers}
        }]
    
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
        
        # 如果当前 part 本身就超限，需要特殊处理
        if part_tokens + base_context_token > self.chunk_size:
          # 先保存已有的 chunk
          if current_chunk_parts:
            final_chunks.append({
              'text': self._add_header_context(top_separator.join(current_chunk_parts), headers),
              'metadata': {'source': file_path, 'headers': headers}
            })
            current_chunk_parts = []
            current_chunk_tokens = base_context_token
          
          # 递归拆分这个超大的 part
          recursive_chunks = self._recursive_split(part, headers, file_path)
          final_chunks.extend(recursive_chunks)
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
  parser.add_argument('--processed_file_root', type=str, default='preprocessed')
  parser.add_argument('--chunk_size', type=int, default=512)
  parser.add_argument('--chunk_overlap', type=int, default=64)
  parser.add_argument('--output', type=str, default='chunked')
  args = parser.parse_args()

  root_path = Path(args.processed_file_root).resolve()
  output_path = Path(args.output).resolve()

  output_dir = output_path
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file_path = output_dir / 'chunks.jsonl'

  chunker = HFChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
  all_chunks = []
  seen_hashes: Set[str] = set()  # 用于去重的哈希集合

  for md_path in root_path.rglob('*.md'):
    if any(part.startswith('.') for part in md_path.parts) or "contributing" in md_path.name:
      continue

    print(f"Processing: {md_path.relative_to(root_path)}")

    with open(md_path, 'r', encoding='utf-8') as f_in:
      text = f_in.read()

    # 传入 root_path 用于生成相对路径
    file_chunks = chunker.chunk_file(text, str(md_path), str(root_path))
    all_chunks.extend(file_chunks)
    print(f"  ✓ Generated {len(file_chunks)} chunks.")
  
  print(f"\nTotal chunks before deduplication: {len(all_chunks)}")
  
  # 用哈希去重，因为每个文本可能包括 Huggingface document 
  # 这样的重复且不包含任何信息的内容
  unique_chunks = []
  duplicates_count = 0
  
  for chunk in all_chunks:
    # 计算文本内容的哈希值
    text_hash = hashlib.sha256(chunk['text'].encode('utf-8')).hexdigest()
    
    if text_hash not in seen_hashes:
      seen_hashes.add(text_hash)
      unique_chunks.append(chunk)
    else:
      duplicates_count += 1
  
  print(f"Removed {duplicates_count} duplicate chunks.")
  print(f"Total unique chunks: {len(unique_chunks)}")
  
  # 将所有 unique chunks 写入一个json文件
  with open(output_file_path, 'w', encoding='utf-8') as f_out:
    for chunk in unique_chunks:
      f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')

  print(f"✓ Processing complete. RAG dataset saved to: {output_file_path}")
  
if __name__ == "__main__":
  main()


