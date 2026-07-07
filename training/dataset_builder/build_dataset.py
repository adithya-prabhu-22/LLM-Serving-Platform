import os
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import boto3
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from training.dataset_builder.build_manifest import build_manifest

DATASETS = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
    "healix": ("health360/Healix-Shot", None),
}

DTYPE = np.uint16
MIN_TOKENS_PER_ARTICLE = 100
MIN_FINAL_CHUNK_TOKENS = 0
S3_RETRY_ATTEMPTS = 5


def upload_chunk(s3_client, bucket: str, prefix: str, chunk_path: Path):
    key = f"{prefix}/{chunk_path.name}"
    for attempt in range(S3_RETRY_ATTEMPTS):
        try:
            s3_client.upload_file(str(chunk_path), bucket, key)
            break
        except Exception as e:
            if attempt == S3_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
    print(f"Uploaded {key} ({chunk_path.stat().st_size / (1024**2):.2f} MB)")
    chunk_path.unlink()
    print(f"Deleted {chunk_path.name}")


def save_chunk(token_buffer, chunk_id, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / f"chunk_{chunk_id:06d}.bin"
    np.array(token_buffer, dtype=DTYPE).tofile(chunk_path)
    print(f"Saved {chunk_path.name} ({len(token_buffer):,} tokens)")
    return chunk_path


def save_progress(output_dir, prefix, chunk_id, total_tokens, total_articles):
    sanitized_prefix = prefix.replace("/", "_")
    progress_path = output_dir.parent / f"{sanitized_prefix}_progress.json"
    with open(progress_path, "w") as f:
        json.dump({
            "chunk_id": chunk_id,
            "total_tokens": total_tokens,
            "total_articles": total_articles
        }, f, indent=2)


def load_progress(output_dir, prefix):
    sanitized_prefix = prefix.replace("/", "_")
    progress_path = output_dir.parent / f"{sanitized_prefix}_progress.json"
    if progress_path.exists():
        with open(progress_path, "r") as f:
            data = json.load(f)
        return data.get("chunk_id", 1), data.get("total_tokens", 0), data.get("total_articles", 0)
    return 1, 0, 0


def build_dataset(
    dataset_name: str,
    bucket: str,
    prefix: str,
    chunk_size: int = 2_000_000,
    tokenizer_name: str = "gpt2",
    max_articles: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    s3_client = boto3.client("s3")

    output_dir = Path("storage/dataset_build") / f"{prefix}_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name not in DATASETS:
        raise ValueError(f"dataset_name must be one of {list(DATASETS.keys())}")

    dataset_id, config_name = DATASETS[dataset_name]
    if config_name is None:
        dataset = load_dataset(dataset_id, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_id, config_name, split="train", streaming=True)

    chunk_id, total_tokens, total_articles = load_progress(output_dir, prefix)
    token_buffer = []
    total_chunks = chunk_id - 1
    skipped = 0

    for sample in tqdm(dataset):
        if max_articles is not None and total_articles >= max_articles:
            break

        if skipped < total_articles:
            skipped += 1
            continue

        title = sample.get("title", "")
        text = sample.get("text", "")
        if not text:
            continue
        if title:
            text = title + "\n\n" + text

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < MIN_TOKENS_PER_ARTICLE:
            continue

        token_buffer.extend(token_ids)
        total_tokens += len(token_ids)
        total_articles += 1

        while len(token_buffer) >= chunk_size:
            chunk_tokens = token_buffer[:chunk_size]
            chunk_path = save_chunk(chunk_tokens, chunk_id, output_dir)
            upload_chunk(s3_client, bucket, prefix, chunk_path)
            token_buffer = token_buffer[chunk_size:]
            chunk_id += 1
            total_chunks += 1
            save_progress(output_dir, prefix, chunk_id, total_tokens, total_articles)

            print()
            print("==============================")
            print(f"Articles : {total_articles:,}")
            print(f"Tokens   : {total_tokens:,}")
            print(f"Chunks   : {total_chunks:,}")
            print("==============================")
            print()

    if len(token_buffer) >= MIN_FINAL_CHUNK_TOKENS:
        chunk_path = save_chunk(token_buffer, chunk_id, output_dir)
        upload_chunk(s3_client, bucket, prefix, chunk_path)
        total_chunks += 1
        save_progress(output_dir, prefix, chunk_id + 1, total_tokens, total_articles)

    build_manifest(
        bucket_name=bucket,
        dataset_type=prefix,
        chunk_size=chunk_size,
        tokenizer=tokenizer_name,
    )

    print("\n========== DATASET COMPLETE ==========")
    print(f"Articles      : {total_articles:,}")
    print(f"Total Tokens  : {total_tokens:,}")
    print(f"Total Chunks  : {total_chunks:,}")
    print(f"S3 Bucket     : {bucket}")
    print(f"Prefix        : {prefix}")
    print("======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a tokenized dataset from Wikipedia or Healix-Shot and upload chunks to S3."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikipedia", "healix"],
        required=True,
        help="Which dataset to process (wikipedia or healix).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket name to upload chunks to.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="S3 prefix (folder path) under the bucket.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Number of tokens per chunk (default: 2,000,000).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer name (default: gpt2).",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for testing).",
    )

    args = parser.parse_args()

    build_dataset(
        dataset_name=args.dataset,
        bucket=args.bucket,
        prefix=args.prefix,
        chunk_size=args.chunk_size,
        tokenizer_name=args.tokenizer,
        max_articles=args.max_articles,
    )