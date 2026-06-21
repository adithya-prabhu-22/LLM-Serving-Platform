import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def load_text_files(input_dir: Path):
    text_files = sorted(input_dir.rglob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    for file_path in text_files:
        print(f"Reading: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            yield file.read()


def save_chunk(token_buffer: list[int], chunk_id: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"chunk_{chunk_id:06d}.bin"
    np.array(token_buffer, dtype=np.uint16).tofile(output_path)
    print(f"Saved {output_path.name} ({len(token_buffer):,} tokens)")


def build_chunks(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 1_000_000,
    tokenizer_name: str = "gpt2",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_buffer = []
    chunk_id = 1
    total_tokens = 0
    total_chunks = 0

    for text in load_text_files(input_dir):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        token_buffer.extend(token_ids)

        while len(token_buffer) >= chunk_size:
            chunk_tokens = token_buffer[:chunk_size]
            save_chunk(
                token_buffer=chunk_tokens,
                chunk_id=chunk_id,
                output_dir=output_dir,
            )
            token_buffer = token_buffer[chunk_size:]
            chunk_id += 1
            total_chunks += 1

    if token_buffer:
        save_chunk(
            token_buffer=token_buffer,
            chunk_id=chunk_id,
            output_dir=output_dir,
        )
        total_chunks += 1

    print("\n========== SUMMARY ==========")
    print(f"Total Tokens : {total_tokens:,}")
    print(f"Total Chunks : {total_chunks:,}")
    print(f"Chunk Size   : {chunk_size:,}")
    print(f"Output Dir   : {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()

    build_chunks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        tokenizer_name=args.tokenizer,
    )


if __name__ == "__main__":
    main()