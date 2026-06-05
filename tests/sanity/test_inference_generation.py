import json
from pathlib import Path

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
    unload_model,
    generate,
)


def test_generate_text():

    model_id = "generation_test"

    config = GPTConfig(
        vocab_size=50257,
        block_size=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )

    model = GPTModel(
        config
    )

    test_dir = Path(
        "tests/resources/inference_generation"
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
        name="Generation Test",
        architecture="GPT",
        config_path=str(
            config_path
        ),
        weights_path=str(
            weights_path
        ),
        tokenizer_backend="huggingface",
        tokenizer_path="gpt2",
    )

    update_model_status(
        model_id,
        "READY",
    )

    load_registered_model(
        model_id
    )

    response = generate(
        model_id=model_id,
        prompt="Hello",
        max_new_tokens=5,
    )

    assert isinstance(
        response,
        str,
    )

    unload_model(
        model_id
    )

    delete_model(
        model_id
    )