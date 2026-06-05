import json
from pathlib import Path

import pytest
from safetensors.torch import save_model

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel

from backend.services.registry_service import (
    register_model,
    delete_model,
    update_model_status,
)

from backend.services.inference_engine import (
    load_registered_model,
    get_loaded_model,
    unload_model,
    is_model_loaded,
)


def test_model_loading_and_unloading():

    model_id = "inference_test"

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
        "tests/resources/inference_engine"
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

    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path=str(
            config_path
        ),
        weights_path=str(
            weights_path
        ),
        tokenizer_path=None,
    )

    update_model_status(
        model_id,
        "READY",
    )

    assert not is_model_loaded(
        model_id
    )

    loaded_model = (
        load_registered_model(
            model_id
        )
    )

    assert loaded_model is not None

    assert is_model_loaded(
        model_id
    )

    cached_model = (
        get_loaded_model(
            model_id
        )
    )

    assert (
        cached_model
        is loaded_model
    )

    unload_model(
        model_id
    )

    assert not is_model_loaded(
        model_id
    )

    delete_model(
        model_id
    )


def test_reject_non_ready_model():

    model_id = (
        "not_ready_test"
    )

    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
        tokenizer_path=None,
    )

    with pytest.raises(
        ValueError
    ):

        load_registered_model(
            model_id
        )

    delete_model(
        model_id
    )