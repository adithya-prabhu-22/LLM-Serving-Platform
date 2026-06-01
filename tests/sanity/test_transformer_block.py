import pytest
import torch

from core.models.transformer_block import (
    TransformerDecoderBlock,
)


def test_transformer_block_forward():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = block(x)

    assert output.shape == (
        2,
        8,
        768,
    )


def test_transformer_block_flash_attention():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
        use_flash_attention=True,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output = block(x)

    assert output.shape == (
        2,
        8,
        768,
    )


def test_transformer_block_kv_cache():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output, present_kv = block(
        x,
        use_cache=True,
    )

    assert output.shape == (
        2,
        8,
        768,
    )

    assert present_kv is not None


def test_transformer_block_kv_cache_reuse():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    _, past_kv = block(
        x,
        use_cache=True,
    )

    next_token = torch.randn(
        2,
        1,
        768,
    )

    output, new_kv = block(
        next_token,
        past_kv=past_kv,
        use_cache=True,
    )

    assert output.shape == (
        2,
        1,
        768,
    )

    assert new_kv is not None


def test_invalid_input_shape():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
    )

    x = torch.randn(
        8,
        768,
    )

    with pytest.raises(
        ValueError
    ):
        block(x)


def test_invalid_embedding_dimension():

    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
    )

    x = torch.randn(
        2,
        8,
        512,
    )

    with pytest.raises(
        ValueError
    ):
        block(x)


def test_invalid_d_model():

    with pytest.raises(
        ValueError
    ):

        TransformerDecoderBlock(
            d_model=0,
            num_heads=12,
        )


def test_invalid_num_heads():

    with pytest.raises(
        ValueError
    ):

        TransformerDecoderBlock(
            d_model=768,
            num_heads=0,
        )


def test_invalid_head_divisibility():

    with pytest.raises(
        ValueError
    ):

        TransformerDecoderBlock(
            d_model=768,
            num_heads=7,
        )