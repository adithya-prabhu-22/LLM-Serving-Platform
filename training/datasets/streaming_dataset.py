from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class StreamingDataset(Dataset):
    def __init__(self, chunk_path: str, block_size: int):
        self.chunk_path = Path(chunk_path)
        if not self.chunk_path.exists():
            raise FileNotFoundError(f"Chunk not found: {self.chunk_path}")
        self.block_size = block_size
        self.tokens = np.fromfile(self.chunk_path, dtype=np.uint16)
        self.num_samples = len(self.tokens) - block_size - 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Chunk {self.chunk_path} has {len(self.tokens)} tokens, "
                f"which is <= block_size+1 ({block_size+1}). Cannot produce any sample."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input_ids = torch.tensor(
            self.tokens[index: index + self.block_size],
            dtype=torch.long,
        )
        targets = torch.tensor(
            self.tokens[index + 1: index + self.block_size + 1],
            dtype=torch.long,
        )
        return input_ids, targets