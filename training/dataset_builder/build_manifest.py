import json
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import boto3


def build_manifest(
    bucket_name: str,
    dataset_type: str,
    chunk_size: int = 1_000_000,
    tokenizer: str = "gpt2",
    dtype: str = "uint16",
):
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    chunks = []
    total_tokens = 0
    for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{dataset_type}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".bin"):
                continue
            chunks.append(key)
            total_tokens += obj["Size"] // 2

    chunks.sort()
    total_chunks = len(chunks)

    if total_chunks == 0:
        raise ValueError(f"No chunks found under prefix {dataset_type}/ in bucket {bucket_name}")

    manifest = {
        "dataset": dataset_type,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "chunks": chunks,
    }

    sanitized_prefix = dataset_type.replace("/", "_")
    manifest_key = f"manifests/{sanitized_prefix}_manifest.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=4),
        ContentType="application/json",
    )

    local_dir = Path("storage/dataset_build")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{sanitized_prefix}_manifest.json"
    with open(local_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print("\n===== MANIFEST CREATED =====")
    print(f"Dataset      : {dataset_type}")
    print(f"Chunks       : {total_chunks:,}")
    print(f"Total Tokens : {total_tokens:,}")
    print(f"Uploaded     : s3://{bucket_name}/{manifest_key}")
    print(f"Local copy   : {local_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--dataset-type",
        choices=["general", "medical"],
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    build_manifest(
        bucket_name=args.bucket,
        dataset_type=args.dataset_type,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()