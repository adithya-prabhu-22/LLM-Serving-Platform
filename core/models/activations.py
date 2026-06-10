import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return F.gelu(x)


class ReLU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return F.relu(x)


class SiLU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return F.silu(x)


class Tanh(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.tanh(x)


SUPPORTED_ACTIVATIONS = {
    "gelu",
    "relu",
    "silu",
    "tanh",
}


def get_activation(
    name: str,
) -> nn.Module:

    name = name.lower()

    if name == "gelu":
        return GELU()

    if name == "relu":
        return ReLU()

    if name == "silu":
        return SiLU()

    if name == "tanh":
        return Tanh()

    raise ValueError(
        f"Unsupported activation function: {name}. "
        f"Supported activations: "
        f"{SUPPORTED_ACTIVATIONS}"
    )