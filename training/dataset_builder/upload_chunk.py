import argparse
from pathlib import Path

import boto3


def upload_chunk(
    bucket_name: str,
    chunk_path: Path,
    dataset_type: str,
):
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")

    s3_client = boto3.client("s3")
    object_key = f"{dataset_type}/{chunk_path.name}"

    print(f"Uploading {chunk_path.name}")
    s3_client.upload_file(str(chunk_path), bucket_name, object_key)
    print(f"Uploaded to s3://{bucket_name}/{object_key}")


def upload_directory(
    bucket_name: str,
    directory: str,
    dataset_type: str,
):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    chunk_files = sorted(directory.glob("*.bin"))

    if not chunk_files:
        raise FileNotFoundError(f"No .bin files found in {directory}")

    print(f"Found {len(chunk_files):,} chunks")

    for chunk_file in chunk_files:
        upload_chunk(
            bucket_name=bucket_name,
            chunk_path=chunk_file,
            dataset_type=dataset_type,
        )

    print("\nUpload completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument(
        "--dataset-type",
        choices=["general", "medical"],
        required=True,
    )
    args = parser.parse_args()

    upload_directory(
        bucket_name=args.bucket,
        directory=args.directory,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()