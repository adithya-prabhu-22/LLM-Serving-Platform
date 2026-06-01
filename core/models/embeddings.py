import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be a positive integer."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be a positive integer."
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be a positive integer."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids shape "
                f"(batch_size, seq_len), "
                f"but got {tuple(input_ids.shape)}"
            )

        _, seq_len = input_ids.shape

        if start_pos < 0:
            raise ValueError(
                "start_pos must be non-negative."
            )

        if start_pos + seq_len > self.block_size:
            raise ValueError(
                f"Sequence length exceeds block size "
                f"({self.block_size})."
            )

        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        return self.dropout(x)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, "
            f"block_size={self.block_size}"
        )