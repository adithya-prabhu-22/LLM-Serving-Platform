import pytest

from core.config.gpt_config import GPTConfig


def test_valid_config():

    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        d_model=768,
        num_heads=12,
        num_layers=12,
    )

    assert config.vocab_size == 50257
    assert config.block_size == 1024
    assert config.ff_dim == 3072


def test_auto_ff_dim():

    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        d_model=512,
        num_heads=8,
        num_layers=6,
    )

    assert config.ff_dim == 2048


def test_invalid_vocab_size():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=0,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_block_size():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=0,
            d_model=768,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_d_model():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=0,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_num_heads():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=0,
            num_layers=12,
        )


def test_invalid_num_layers():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=0,
        )


def test_invalid_head_divisibility():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=7,
            num_layers=12,
        )


def test_invalid_dropout():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            dropout=1.5,
        )


def test_invalid_ff_dim():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            ff_dim=-1,
        )


def test_empty_activation():

    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            activation="",
        )