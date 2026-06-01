import torch
import torch.nn as nn

from core.models.activations import (
    get_activation,
    is_gated_activation,
)


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        ff_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model must be a positive integer."
            )

        if ff_dim is None:
            ff_dim = 4 * d_model

        if ff_dim <= 0:
            raise ValueError(
                "ff_dim must be a positive integer."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.d_model = d_model
        self.ff_dim = ff_dim

        self.activation_name = (
            activation.lower()
        )

        self.is_gated = (
            is_gated_activation(
                self.activation_name
            )
        )

        self.activation = get_activation(
            self.activation_name
        )

        hidden_dim = (
            2 * ff_dim
            if self.is_gated
            else ff_dim
        )

        self.fc1 = nn.Linear(
            d_model,
            hidden_dim,
            bias=bias,
        )

        self.fc2 = nn.Linear(
            ff_dim,
            d_model,
            bias=bias,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape "
                f"(batch_size, seq_len, d_model), "
                f"but got {tuple(x.shape)}"
            )

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected last dimension "
                f"{self.d_model}, "
                f"but got {x.size(-1)}"
            )

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.dropout(x)

        return x