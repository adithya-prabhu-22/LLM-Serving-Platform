from pathlib import Path
import json

import torch
from safetensors.torch import save_model

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel

from backend.services.model_loader import (
    load_model,
)


def test_load_model_weights():

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

    test_dir = Path(
        "tests/resources/model_loader"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        test_dir
        / "config.json"
    )

    weights_path = (
        test_dir
        / "model.safetensors"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config.__dict__,
            file,
            indent=4,
        )

    save_model(
        model,
        str(weights_path),
    )

    loaded_model = load_model(
        config_path=config_path,
        weights_path=weights_path,
    )

    original_state = (
        model.state_dict()
    )

    loaded_state = (
        loaded_model.state_dict()
    )

    for key in original_state:

        assert torch.equal(
            original_state[key],
            loaded_state[key],
        )