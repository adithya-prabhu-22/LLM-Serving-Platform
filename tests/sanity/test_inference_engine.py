import json
from pathlib import Path

import pytest
from safetensors.torch import save_model

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel
from backend.services.registry_service import register_model, delete_model
from backend.services.inference_engine import build_model, get_loaded_model, unload_model, is_model_built


def test_model_loading_and_unloading():
    model_id = "inference_test"

    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        cache_type="ring",
    )

    model = GPTModel(config)

    test_dir = Path("tests/resources/inference_engine")
    test_dir.mkdir(parents=True, exist_ok=True)

    config_path = test_dir / "config.json"
    weights_path = test_dir / "model.safetensors"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config.__dict__, file, indent=4)

    save_model(model, str(weights_path))

    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path=str(config_path),
        weights_path=str(weights_path),
    )

    assert not is_model_built(model_id)

    loaded_model = build_model(model_id)
    assert loaded_model is not None
    assert is_model_built(model_id)

    cached_model = get_loaded_model(model_id)
    assert cached_model is loaded_model

    unload_model(model_id)
    assert not is_model_built(model_id)

    delete_model(model_id)

    print("✓ Model loading and unloading")


def test_get_unknown_model():
    with pytest.raises(ValueError):
        get_loaded_model("unknown_model")
    print("✓ Unknown model validation")


def run_all_tests():
    print("\nRunning Inference Engine tests...\n")
    test_model_loading_and_unloading()
    test_get_unknown_model()
    print("\n✓ All Inference Engine tests passed")


if __name__ == "__main__":
    run_all_tests()