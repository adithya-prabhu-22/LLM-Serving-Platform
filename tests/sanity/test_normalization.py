import pytest
import torch

from core.models.normalization import LayerNorm


def test_layernorm_forward():

    layer_norm = LayerNorm(
        d_model=768,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = layer_norm(x)

    assert output.shape == (
        2,
        8,
        768,
    )


def test_layernorm_without_bias():

    layer_norm = LayerNorm(
        d_model=768,
        bias=False,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = layer_norm(x)

    assert output.shape == (
        2,
        8,
        768,
    )

    assert layer_norm.beta is None


def test_layernorm_with_bias():

    layer_norm = LayerNorm(
        d_model=768,
        bias=True,
    )

    assert layer_norm.beta is not None


def test_output_mean_close_to_zero():

    layer_norm = LayerNorm(
        d_model=768,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = layer_norm(x)

    mean = output.mean(
        dim=-1,
    )

    assert torch.allclose(
        mean,
        torch.zeros_like(mean),
        atol=1e-4,
    )


def test_output_variance_close_to_one():

    layer_norm = LayerNorm(
        d_model=768,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = layer_norm(x)

    variance = output.var(
        dim=-1,
        unbiased=False,
    )

    assert torch.allclose(
        variance,
        torch.ones_like(variance),
        atol=1e-3,
    )


def test_invalid_d_model():

    with pytest.raises(
        ValueError
    ):

        LayerNorm(
            d_model=0,
        )