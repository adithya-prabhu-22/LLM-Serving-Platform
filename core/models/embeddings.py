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

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids shape "
                f"(batch_size, seq_len), "
                f"but got {tuple(input_ids.shape)}"
            )

        token_embeddings = (
            self.token_embedding(
                input_ids
            )
        )

        x = token_embeddings

        x = self.dropout(x)

        return x