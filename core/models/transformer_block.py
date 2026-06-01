import torch
import torch.nn as nn

from core.models.normalization import LayerNorm
from core.models.feedforward import FeedForward
from core.models.attention import (
    MultiHeadCausalSelfAttention,
)


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model must be a positive integer."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be a positive integer."
            )

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads

        self.ln1 = LayerNorm(
            d_model=d_model,
            bias=bias,
        )

        self.attn = (
            MultiHeadCausalSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                use_flash_attention=(
                    use_flash_attention
                ),
            )
        )

        self.ln2 = LayerNorm(
            d_model=d_model,
            bias=bias,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: (
            tuple[
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None,
        use_cache: bool = False,
    ):

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

        attn_input = self.ln1(x)

        if use_cache:

            attn_output, present_kv = (
                self.attn(
                    attn_input,
                    past_kv=past_kv,
                    use_cache=True,
                )
            )

        else:

            attn_output = self.attn(
                attn_input
            )

            present_kv = None

        x = x + attn_output

        ffn_input = self.ln2(x)

        ffn_output = self.ffn(
            ffn_input
        )

        x = x + ffn_output

        if use_cache:
            return x, present_kv

        return x