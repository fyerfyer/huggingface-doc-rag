import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import random
import httpx

load_dotenv()

client = OpenAI(
  api_key=os.environ.get('DEEPSEEK_API_KEY'),
  base_url="https://api.deepseek.com",
  http_client=httpx.Client(proxy=None, trust_env=False) # 开了代理的话要加这个，不然会报错
)

def generate_questions(chunk): 
  text = chunk['text']
  source = chunk['metadata']['source']
  
  prompt = f"""
  You are an expert AI developer working with the Hugging Face Transformers library.
  Based strictly on the following documentation excerpt, generate 3 technical questions that a developer might ask.

  Documentation Source: {source}
    
  Context:
  {text}

  Requirements:
  1. The questions must be answerable **solely** based on the provided context.
  2. Include specific specific Python class names, function names, arguments, or technical terms found in the text (e.g., 'logits', 'AutoModel', 'training_args').
  3. The questions should be concise and professional.
  4. Output ONLY a valid JSON list of strings.
    
  Output Format:
  ["Question 1?", "Question 2?", "Question 3?"]
  """

  response = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{
      'role': 'user',
      'content': prompt
    }],
    temperature=0.7
  )

  # 从 response 中提取 content
  try:
    content = response.choices[0].message.content
    # 解析 JSON 数组格式的问题列表
    questions = json.loads(content)
    
    if isinstance(questions, list):
      return [str(q).strip() for q in questions if q and str(q).strip()]
    else:
      # 如果不是列表，尝试按行分割
      lines = [l.strip() for l in content.splitlines() if l.strip()]
      return lines if lines else [content.strip()]
      
  except (json.JSONDecodeError, AttributeError, IndexError) as e:
    print(f"    [WARNING] Failed to parse response: {e}")
    print(f"    Raw content: {content if 'content' in locals() else 'N/A'}")
    return []

# 直接从 headers 生成问题来节约 API 调用成本
def generate_questions_from_headers(chunk):
    """Generates rule-based questions from metadata headers."""
    headers = chunk.get('metadata', {}).get('headers')
    if not headers:
        return []

    questions = []
    last_header = headers[-1]
    
    questions.append(f"What does the documentation say about '{last_header}'?")

    if len(headers) > 1:
        parent_headers = " > ".join(headers[:-1])
        questions.append(f"Explain the '{last_header}' section within '{parent_headers}'.")
    
    questions.append(f"How do I use or implement '{last_header}' according to the provided text?")

    return questions

def main():
  parser = argparse.ArgumentParser(description="Finetune data generation for RAG")
  parser.add_argument('--chunk_path', type=str, default='chunked')
  parser.add_argument('--output', type=str, default='embedding_train_dataset')
  parser.add_argument('--api_sample_size', type=int, default=2000)
  args = parser.parse_args()

  input_path = Path(args.chunk_path).resolve()
  input_file_path = input_path / 'chunks.jsonl'

  output_path = Path(args.output).resolve()
  output_dir = output_path
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file_path = output_dir / 'train_dataset.jsonl'

  print(f"Loading chunks from {input_file_path}...")
  with open(input_file_path, 'r', encoding='utf-8') as f:
    all_chunks = [json.loads(line) for line in f]
  print(f"Loaded {len(all_chunks)} total chunks.")

  random.shuffle(all_chunks)

  api_sample_size = args.api_sample_size
  if len(all_chunks) < api_sample_size:
      print(f"Warning: Total chunks ({len(all_chunks)}) is less than sample size ({api_sample_size}). Processing all with API.")
      api_chunks = all_chunks
      header_chunks = []
  else:
      api_chunks = all_chunks[:api_sample_size]
      header_chunks = all_chunks[api_sample_size:]

  print(f"Will process {len(api_chunks)} chunks with DeepSeek API.")
  print(f"Will process {len(header_chunks)} chunks with header-based rules.")

  def is_valid_chunk(chunk):
    """检查 chunk 是否包含有意义的内容"""
    text = chunk.get('text', '')
    
    # 移除 Context 前缀
    if 'Context:' in text:
      text = text.split('\n\n', 1)[-1] if '\n\n' in text else text
    
    # 检查是否太短
    if len(text.strip()) < 100:
      return False
    
    # 检查是否主要是数据（超过50%是数字、逗号、引号、冒号）
    data_chars = sum(1 for c in text if c in '0123456789,:\'"{}[]')
    if data_chars / len(text) > 0.5:
      return False
    
    # 检查是否包含至少一些自然语言（单词）
    words = [w for w in text.split() if w.isalpha() and len(w) > 2]
    if len(words) < 10:
      return False
    
    return True
  
  with open(output_file_path, 'w', encoding='utf-8') as f_out:
    # 使用 API 规则处理部分 chunks
    print("\n--- Starting API Generation ---")
    skipped = 0
    for i, chunk in enumerate(api_chunks):
      # 跳过无效的 chunk
      if not is_valid_chunk(chunk):
        skipped += 1
        continue
        
      print(f"Generating API questions for chunk {i+1}/{len(api_chunks)} (skipped {skipped})...")
      try:
        questions = generate_questions(chunk)
        
        # 为每个问题创建一个单独的训练样本，都对应同一个 context
        context_text = chunk.get('text', '')
        
        # chunk 的 text 已经包含了 "Context: ..." 前缀，直接使用即可
        # 不需要再添加额外的 header 信息
        for q in questions:
          data_pair = {
            'query': q,
            'pos': context_text
          }
          f_out.write(json.dumps(data_pair, ensure_ascii=False) + '\n')
          print(f"  - Generated question: {q[:80]}...")
          
      except Exception as e:
        print(f"    [ERROR] Failed to generate API questions for chunk from source '{chunk.get('metadata', {}).get('source')}': {e}")
        continue
    
    print(f"\nSkipped {skipped} invalid chunks out of {len(api_chunks)}")
    
    # 使用 header-based 规则处理剩余的 chunks
    print("\n--- Starting Header-based Generation ---")
    for i, chunk in enumerate(header_chunks):
      questions = generate_questions_from_headers(chunk)
      if not questions:
        continue
      
      # 为每个问题创建一个单独的训练样本
      context_text = chunk.get('text', '')
      
      for q in questions:
        data_pair = {
          'query': q,
          'pos': context_text
        }
        f_out.write(json.dumps(data_pair, ensure_ascii=False) + '\n')
        
  print(f"\nProcessing complete. Dataset with {len(api_chunks) + len(header_chunks)} entries saved to {output_file_path}")
        
if __name__ == "__main__":
  main()