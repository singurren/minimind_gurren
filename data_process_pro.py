import re
import os
import hashlib
import json
import argparse
from typing import List, Set

# 尝试导入 datasketch，如果不存在则使用 Mock
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    print("⚠️ Warning: 'datasketch' library not found. Using simple hash-based deduplication fallback.")

class DataProcessor:
    """
    工业级数据处理 Pipeline
    功能：
    1. MinHash LSH 模糊去重：解决大规模语料中的近义重复问题。
    2. 启发式质量过滤：去除低质量、噪声大的文本。
    """
    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.seen_hashes = set()
        self.total_processed = 0
        self.total_deduplicated = 0
        self.total_filtered = 0

    def _get_minhash(self, text: str):
        """生成文本的 MinHash 指纹"""
        m = MinHash(num_perm=self.num_perm)
        # N-gram shingling
        tokens = [text[i:i+3] for i in range(len(text)-2)]
        for t in tokens:
            m.update(t.encode('utf8'))
        return m

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """
        判断是否为重复文本
        如果 datasketch 可用，使用 LSH；否则使用精准 Hash 匹配。
        """
        if HAS_DATASKETCH:
            minhash = self._get_minhash(text)
            results = self.lsh.query(minhash)
            if results:
                return True
            self.lsh.insert(doc_id, minhash)
            return False
        else:
            # Fallback: Simple MD5 exact match
            h = hashlib.md5(text.encode('utf-8')).hexdigest()
            if h in self.seen_hashes:
                return True
            self.seen_hashes.add(h)
            return False

    def quality_filter(self, text: str) -> bool:
        """
        基于启发式规则的质量过滤
        Returns: True if passed (keep), False if filtered (drop)
        """
        if len(text) < 20: # 过滤太短的文本
            return False
        
        # 过滤代码密度过高的文本 (如果目标是通用自然语言)
        if text.count('{') + text.count('}') > len(text) * 0.1:
            return False

        # 过滤标点符号过少的文本 (可能是噪音)
        import string
        punc_count = sum([1 for char in text if char in string.punctuation])
        if punc_count / len(text) < 0.01:
            return False

        return True

    def process_file(self, input_path: str, output_path: str):
        print(f"🔄 Processing {input_path}...")
        
        # 模拟读取和处理
        # 实际场景中应流式读取 (yield) 以处理大文件
        if not os.path.exists(input_path):
             # 创建一个 Dummy 文件用于演示
             print("Create dummy file for demo...")
             with open(input_path, 'w', encoding='utf-8') as f:
                 f.write(json.dumps({"text": "DeepSeek 是一个很棒的模型。"}) + "\n")
                 f.write(json.dumps({"text": "DeepSeek 是一个很棒的模型。"}) + "\n") # Duplicate
                 f.write(json.dumps({"text": "垃圾数据"}) + "\n") # Low quality

        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for i, line in enumerate(fin):
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    doc_id = f"doc_{i}"

                    # 1. Quality Filter
                    if not self.quality_filter(text):
                        self.total_filtered += 1
                        continue

                    # 2. Deduplication
                    if self.is_duplicate(text, doc_id):
                        self.total_deduplicated += 1
                        continue

                    # Write clean data
                    fout.write(line)
                    self.total_processed += 1

                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Processing Done.")
        print(f"   Saved: {self.total_processed}")
        print(f"   Filtered (Quality): {self.total_filtered}")
        print(f"   Filtered (Duplicate): {self.total_deduplicated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Pipeline")
    parser.add_argument("--input", type=str, default="dataset/raw_data.jsonl")
    parser.add_argument("--output", type=str, default="dataset/clean_data.jsonl")
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs("dataset", exist_ok=True)
    
    processor = DataProcessor()
    processor.process_file(args.input, args.output)
