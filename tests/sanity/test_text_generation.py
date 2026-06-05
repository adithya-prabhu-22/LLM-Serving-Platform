import torch

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel

from backend.services.text_generation import (
    generate_tokens,
)


def test_generate_tokens():

    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )

    model = GPTModel(
        config
    )

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 5),
    )

    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=10,
    )

    assert (
        output_ids.shape[0]
        == 1
    )

    assert (
        output_ids.shape[1]
        == 15
    )


def test_generate_tokens_respects_block_size():

    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )

    model = GPTModel(
        config
    )

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 8),
    )

    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=4,
    )

    assert (
        output_ids.shape[1]
        == 12
    )