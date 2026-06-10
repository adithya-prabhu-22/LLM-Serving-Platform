from pathlib import Path

import torch
import tiktoken

from torch.utils.data import Dataset


class TextChunkDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        stride: int = 256,
    ):
        self.samples = []

        self.block_size = (
            block_size
        )

        self.tokenizer = (
            tiktoken.get_encoding(
                "gpt2"
            )
        )

        data_dir = Path(
            data_dir
        )

        text_files = sorted(
            data_dir.glob(
                "*.txt"
            )
        )

        if not text_files:
            raise ValueError(
                f"No .txt files found in "
                f"{data_dir}"
            )

        print(
            f"Found {len(text_files)} "
            f"text files"
        )

        for text_file in text_files:

            with open(
                text_file,
                "r",
                encoding="utf-8",
            ) as file:

                text = file.read()

            token_ids = (
                self.tokenizer.encode(
                    text
                )
            )

            if (
                len(token_ids)
                < block_size + 1
            ):
                continue

            for start in range(
                0,
                len(token_ids)
                - block_size
                - 1,
                stride,
            ):

                self.samples.append(
                    (
                        token_ids,
                        start,
                    )
                )

        print(
            f"Created "
            f"{len(self.samples):,} "
            f"training samples"
        )

        if len(self.samples) == 0:
            raise ValueError(
                "No training samples created."
            )

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):

        token_ids, start = (
            self.samples[idx]
        )

        window = token_ids[
            start :
            start
            + self.block_size
            + 1
        ]

        input_ids = torch.tensor(
            window[:-1],
            dtype=torch.long,
        )

        targets = torch.tensor(
            window[1:],
            dtype=torch.long,
        )

        return (
            input_ids,
            targets,
        )