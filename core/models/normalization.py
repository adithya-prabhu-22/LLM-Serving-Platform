import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, but got {x.size(-1)}")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.beta is not None:
            return self.gamma * x_hat + self.beta
        return self.gamma * x_hat