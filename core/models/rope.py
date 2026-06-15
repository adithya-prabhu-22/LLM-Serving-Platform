import torch
import torch.nn as nn


def rotate_half(
    x: torch.Tensor,
):

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack(
        (
            -x2,
            x1,
        ),
        dim=-1,
    ).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):

    cos = torch.repeat_interleave(
        cos,
        repeats=2,
        dim=-1,
    )

    sin = torch.repeat_interleave(
        sin,
        repeats=2,
        dim=-1,
    )

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q = (
        q * cos
        + rotate_half(q) * sin
    )

    k = (
        k * cos
        + rotate_half(k) * sin
    )

    return q, k


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos_cached",
            torch.cos(freqs),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            torch.sin(freqs),
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
    ):

        return (
            self.cos_cached[positions],
            self.sin_cached[positions],
        )