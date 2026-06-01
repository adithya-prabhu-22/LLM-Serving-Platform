import pytest
import torch

from core.models.activations import (
    GELU,
    ReLU,
    SiLU,
    Tanh,
    SwiGLU,
    GEGLU,
    ReGLU,
    get_activation,
    is_gated_activation,
)


def test_gelu():

    activation = GELU()

    x = torch.randn(
        2,
        8,
        768,
    )

    output = activation(x)

    assert output.shape == x.shape


def test_relu():

    activation = ReLU()

    x = torch.randn(
        2,
        8,
        768,
    )

    output = activation(x)

    assert output.shape == x.shape


def test_silu():

    activation = SiLU()

    x = torch.randn(
        2,
        8,
        768,
    )

    output = activation(x)

    assert output.shape == x.shape


def test_tanh():

    activation = Tanh()

    x = torch.randn(
        2,
        8,
        768,
    )

    output = activation(x)

    assert output.shape == x.shape


def test_swiglu():

    activation = SwiGLU()

    x = torch.randn(
        2,
        8,
        6144,
    )

    output = activation(x)

    assert output.shape == (
        2,
        8,
        3072,
    )


def test_geglu():

    activation = GEGLU()

    x = torch.randn(
        2,
        8,
        6144,
    )

    output = activation(x)

    assert output.shape == (
        2,
        8,
        3072,
    )


def test_reglu():

    activation = ReGLU()

    x = torch.randn(
        2,
        8,
        6144,
    )

    output = activation(x)

    assert output.shape == (
        2,
        8,
        3072,
    )


def test_get_activation():

    activations = [
        "gelu",
        "relu",
        "silu",
        "tanh",
        "swiglu",
        "geglu",
        "reglu",
    ]

    for name in activations:

        activation = get_activation(
            name
        )

        assert activation is not None


def test_invalid_activation():

    with pytest.raises(
        ValueError
    ):

        get_activation(
            "invalid_activation"
        )


def test_gated_activation_check():

    assert is_gated_activation(
        "swiglu"
    )

    assert is_gated_activation(
        "geglu"
    )

    assert is_gated_activation(
        "reglu"
    )

    assert not is_gated_activation(
        "gelu"
    )

    assert not is_gated_activation(
        "relu"
    )