import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, dataset_dir: str, tokenizer, block_size: int):
        if block_size < 2:
            raise ValueError("block_size must be at least 2.")
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        self._load_dataset()

    def _load_dataset(self):
        jsonl_files = sorted(self.dataset_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.dataset_dir}")
        total_documents = 0
        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text:
                        continue
                    token_ids = self.tokenizer.encode(text)
                    if len(token_ids) < 2:
                        continue
                    total_documents += 1
                    chunk_size = self.block_size + 1
                    for start in range(0, len(token_ids) - 1, chunk_size):
                        chunk = token_ids[start:start + chunk_size]
                        if len(chunk) < 2:
                            continue
                        self.examples.append(chunk)
        print(f"Loaded {len(self.examples):,} training chunks from {total_documents:,} documents.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        token_ids = self.examples[index]
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, targets