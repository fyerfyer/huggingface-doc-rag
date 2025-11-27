import argparse
import json
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
import nltk 
from tqdm import tqdm 
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.find('corpora/stopwords')
nltk.download('stopwords')
nltk.data.find('tokenizers/punkt')

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = PorterStemmer()

def tokenizer(text: str) -> List[str]:
  text = str(text or "").lower()
  tokens = nltk.word_tokenize(text)

  # 移除标点与非数字
  tokens = [word for word in tokens if word.isalnum()]
  tokens = [word for word in tokens if word not in stopwords]
  tokens = [stemmer.stem(word) for word in tokens]
  return [token for token in tokens if token]

def main():
  parser = argparse.ArgumentParser(description="Negative example generation for RAG")
  parser.add_argument('--corpus_path', type=str, default='chunked')
  parser.add_argument('--train_dataset_path', type=str, default='embedding_train_dataset')
  parser.add_argument('--output', type=str, default='processed_embedding_train_dataset')
  args = parser.parse_args()

  corpus_path = Path(args.corpus_path).resolve()
  dataset_path = Path(args.train_dataset_path).resolve()
  corpus_file_path = corpus_path / 'chunks.jsonl'
  dataset_file_path = dataset_path / 'train_dataset.jsonl'

  output_path = Path(args.output).resolve()
  output_file_path = output_path / 'train_dataset.jsonl'

  corpus_chunks = []
  with open(corpus_file_path, 'r', encoding='utf-8') as f_in:
    for line in f_in:
      data = json.loads(line)
      corpus_chunks.append(data)

  # 提取文本字段并 token 化
  corpus_texts = [chunk.get('text', '') for chunk in corpus_chunks]
  tokenized_corpus = [tokenizer(text) for text in tqdm(corpus_texts, desc='Tokenizing')]

  # 建立 BM25 索引
  bm25 = BM25Okapi(tokenized_corpus)
  output_triplets = []

  # 开始负样本挖掘
  # Ensure output directory exists
  output_path.mkdir(parents=True, exist_ok=True)

  with open(dataset_file_path, 'r', encoding='utf-8') as f_in:
    for line in tqdm(f_in, desc='Processing query'):
      item = json.loads(line)
      query = item['query']
      pos_doc = item['pos']
      tokenized_query = tokenizer(query)
      
      # 检索 Top 10
      top_n = bm25.get_top_n(tokenized_query, corpus_texts, n=10)
      negatives = [doc for doc in top_n if doc != pos_doc]
      hard_negative = None 
      if negatives: 
        hard_negative = negatives[0]
      if hard_negative: 
        output_triplets.append({
          'query': query, 
          'pos': pos_doc, 
          'neg': hard_negative
        })
      else :
        print(f"Warning: No hard negative found for query: {query}")
    
  with open(output_file_path, 'w', encoding='utf-8') as f_out:
    for item in output_triplets: 
      print("Writing item with query:", item['query'])
      f_out.write(json.dumps(item) + '\n')
  
if __name__ == "__main__":
  main()

