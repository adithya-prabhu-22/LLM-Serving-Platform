import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos_cached = torch.cos(emb)
        sin_cached = torch.sin(emb)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, positions: torch.Tensor):
        if positions.numel() == 0:
            raise ValueError("positions tensor is empty.")
        max_position = int(positions.max().item())
        if max_position >= self.max_seq_len:
            raise ValueError(f"Position index {max_position} exceeds max_seq_len={self.max_seq_len}.")
        if positions.dim() == 1:
            cos = self.cos_cached[positions]
            sin = self.sin_cached[positions]
        elif positions.dim() == 2:
            flat_pos = positions.reshape(-1)
            cos = self.cos_cached[flat_pos].reshape(positions.shape + (-1,))
            sin = self.sin_cached[flat_pos].reshape(positions.shape + (-1,))
        else:
            raise ValueError(f"Unsupported positions shape: {tuple(positions.shape)}")
        return cos, sin