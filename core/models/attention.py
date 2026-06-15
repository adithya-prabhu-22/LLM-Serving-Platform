import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.cache.kv_cache import KVCache


class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
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

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=bias,
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):

        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape "
                f"(batch_size, seq_len, d_model), "
                f"but got {tuple(x.shape)}"
            )

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.d_model}, got {d_model}"
            )

        qkv = self.qkv_proj(x)

        queries, keys, values = qkv.chunk(
            3,
            dim=-1,
        )

        queries = queries.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        keys = keys.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        values = values.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        past_length = 0

        if isinstance(
            past_kv,
            KVCache,
        ):

            past_length = len(
                past_kv
            )

            past_kv.append(
                keys,
                values,
            )

            keys = past_kv.get_keys()

            values = past_kv.get_values()

            present_kv = past_kv

        elif use_cache:

            raise ValueError(
                "KVCache expected when "
                "use_cache=True"
            )

        else:

            present_kv = None

        if (
            self.use_flash_attention
            and seq_len > 1
        ):

            context = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=(
                    self.dropout_p
                    if self.training
                    else 0.0
                ),
                is_causal=True,
            )

        else:

            scores = queries @ keys.transpose(
                -2,
                -1,
            )

            scores = scores / math.sqrt(
                self.head_dim
            )

            total_length = keys.size(2)

            query_positions = torch.arange(
                past_length,
                past_length + seq_len,
                device=x.device,
            ).unsqueeze(-1)

            key_positions = torch.arange(
                total_length,
                device=x.device,
            ).unsqueeze(0)

            causal_mask = (
                key_positions
                <= query_positions
            )

            scores = scores.masked_fill(
                ~causal_mask.view(
                    1,
                    1,
                    seq_len,
                    total_length,
                ),
                float("-inf"),
            )

            attention_weights = torch.softmax(
                scores,
                dim=-1,
            )

            attention_weights = (
                self.attn_dropout(
                    attention_weights
                )
            )

            context = (
                attention_weights
                @ values
            )

        context = context.transpose(
            1,
            2,
        )

        context = context.contiguous().view(
            batch_size,
            seq_len,
            d_model,
        )

        output = self.out_proj(context)

        if use_cache:
            return output, present_kv

        return output