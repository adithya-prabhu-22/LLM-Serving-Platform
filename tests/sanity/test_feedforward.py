import pytest
import torch

from core.models.feedforward import FeedForward


def test_feedforward_forward():

    model = FeedForward(
        d_model=768,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = model(x)

    assert output.shape == (
        2,
        8,
        768,
    )


def test_all_activations():

    activations = [
        "gelu",
        "relu",
        "silu",
        "tanh",
        "swiglu",
        "geglu",
        "reglu",
    ]

    x = torch.randn(
        2,
        8,
        768,
    )

    for activation in activations:

        model = FeedForward(
            d_model=768,
            ff_dim=3072,
            activation=activation,
            dropout=0.1,
        )

        output = model(x)

        assert output.shape == (
            2,
            8,
            768,
        )


def test_default_ff_dim():

    model = FeedForward(
        d_model=768,
        activation="gelu",
    )

    assert model.ff_dim == 3072


def test_gated_activation_hidden_dim():

    model = FeedForward(
        d_model=768,
        ff_dim=3072,
        activation="swiglu",
    )

    assert model.fc1.out_features == 6144


def test_invalid_input_shape():

    model = FeedForward(
        d_model=768,
    )

    x = torch.randn(
        8,
        768,
    )

    with pytest.raises(ValueError):
        model(x)


def test_invalid_embedding_dimension():

    model = FeedForward(
        d_model=768,
    )

    x = torch.randn(
        2,
        8,
        512,
    )

    with pytest.raises(ValueError):
        model(x)


def test_invalid_d_model():

    with pytest.raises(ValueError):

        FeedForward(
            d_model=0,
        )


def test_invalid_ff_dim():

    with pytest.raises(ValueError):

        FeedForward(
            d_model=768,
            ff_dim=0,
        )


def test_invalid_dropout():

    with pytest.raises(ValueError):

        FeedForward(
            d_model=768,
            dropout=1.5,
        )