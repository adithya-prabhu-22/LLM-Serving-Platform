import json
from pathlib import Path


REQUIRED_FIELDS = (
    "dataset",
    "tokenizer",
    "dtype",
    "chunk_size",
    "total_chunks",
    "total_tokens",
    "chunks",
)


def load_manifest(manifest_path: str):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    for field in REQUIRED_FIELDS:
        if field not in manifest:
            raise ValueError(f"Missing field: {field}")

    if not isinstance(manifest["chunks"], list):
        raise ValueError("'chunks' must be a list.")

    if len(manifest["chunks"]) == 0:
        raise ValueError("Manifest contains no chunks.")

    return manifest