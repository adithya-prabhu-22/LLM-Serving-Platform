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


class SwiGLU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate, value = x.chunk(
            2,
            dim=-1,
        )

        return F.silu(gate) * value


class GEGLU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate, value = x.chunk(
            2,
            dim=-1,
        )

        return F.gelu(gate) * value


class ReGLU(nn.Module):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate, value = x.chunk(
            2,
            dim=-1,
        )

        return F.relu(gate) * value


GATED_ACTIVATIONS = {
    "swiglu",
    "geglu",
    "reglu",
}


def is_gated_activation(
    name: str,
) -> bool:

    return (
        name.lower()
        in GATED_ACTIVATIONS
    )


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

    if name == "swiglu":
        return SwiGLU()

    if name == "geglu":
        return GEGLU()

    if name == "reglu":
        return ReGLU()

    raise ValueError(
        f"Unsupported activation function: {name}. "
        f"Supported activations: "
        f"gelu, relu, silu, tanh, "
        f"swiglu, geglu, reglu."
    )