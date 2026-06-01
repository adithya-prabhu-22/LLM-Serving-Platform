from types import SimpleNamespace

import pytest
import torch

from core.models.gpt import GPTModel


def get_config():

    return SimpleNamespace(
        vocab_size=50257,
        block_size=1024,
        d_model=768,
        num_heads=12,
        num_layers=2,
        dropout=0.1,
        qkv_bias=False,
        ff_dim=3072,
        activation="gelu",
        use_flash_attention=False,
    )


def test_gpt_forward():

    config = get_config()

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )


def test_gpt_flash_attention():

    config = get_config()

    config.use_flash_attention = True

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )


def test_gpt_kv_cache_creation():

    config = get_config()

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    logits, present_kv = model(
        input_ids,
        use_cache=True,
    )

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )

    assert len(present_kv) == (
        config.num_layers
    )


def test_gpt_kv_cache_reuse():

    config = get_config()

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    _, past_kv = model(
        input_ids,
        use_cache=True,
    )

    next_token = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 1),
    )

    logits, new_kv = model(
        next_token,
        past_kv=past_kv,
        use_cache=True,
    )

    assert logits.shape == (
        2,
        1,
        config.vocab_size,
    )

    assert len(new_kv) == (
        config.num_layers
    )


def test_num_parameters():

    config = get_config()

    model = GPTModel(config)

    params = model.num_parameters()

    assert params > 0


def test_trainable_parameters():

    config = get_config()

    model = GPTModel(config)

    params = model.num_parameters(
        trainable_only=True,
    )

    assert params > 0


def test_invalid_input_shape():

    config = get_config()

    model = GPTModel(config)

    invalid_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(8,),
    )

    with pytest.raises(
        ValueError
    ):
        model(invalid_input)


def test_invalid_past_kv_length():

    config = get_config()

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    invalid_past_kv = [None]

    with pytest.raises(
        ValueError
    ):
        model(
            input_ids,
            past_kv=invalid_past_kv,
        )


def test_sequence_length_exceeded():

    config = get_config()

    config.block_size = 8

    model = GPTModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 16),
    )

    with pytest.raises(
        ValueError
    ):
        model(input_ids)


def test_missing_config_field():

    config = SimpleNamespace(
        vocab_size=50257,
    )

    with pytest.raises(
        ValueError
    ):
        GPTModel(config)