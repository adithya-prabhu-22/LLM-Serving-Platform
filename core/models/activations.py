import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

SUPPORTED_ACTIVATIONS = {"gelu", "relu", "silu", "tanh"}

def get_activation(name: str) -> nn.Module:
    if not isinstance(name, str):
        raise ValueError("Activation name must be a string.")
    name = name.lower()
    if name == "gelu":
        return GELU()
    elif name == "relu":
        return ReLU()
    elif name == "silu":
        return SiLU()
    elif name == "tanh":
        return Tanh()
    raise ValueError(f"Unsupported activation function: {name}. Supported activations: {sorted(SUPPORTED_ACTIVATIONS)}")