from pathlib import Path

import numpy as np


def load_chunk(chunk_path: str) -> np.ndarray:
    chunk_path = Path(chunk_path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")

    tokens = np.fromfile(chunk_path, dtype=np.uint16)
    if len(tokens) == 0:
        raise ValueError(f"Empty chunk: {chunk_path}")

    return tokens


def get_chunk_token_count(chunk_path: str) -> int:
    return len(load_chunk(chunk_path))