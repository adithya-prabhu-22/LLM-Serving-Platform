import json
from pathlib import Path

from safetensors.torch import load_file

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel


def normalize_path(
    path: str | Path,
) -> Path:

    return (
        Path(
            str(path)
            .replace("\\", "/")
        )
        .resolve()
    )


def load_config(
    config_path: str | Path,
) -> GPTConfig:

    config_path = normalize_path(
        config_path
    )

    print(
        "\n===== CONFIG PATH ====="
    )

    print(
        config_path
    )

    print(
        "=======================\n"
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    print(
        "\n===== CONFIG FILE ====="
    )

    print(
        text
    )

    print(
        "=======================\n"
    )

    config_data = json.loads(
        text
    )

    print(
        "\n===== CONFIG DATA ====="
    )

    print(
        config_data
    )

    print(
        "=======================\n"
    )

    return GPTConfig(
        **config_data
    )


def build_model(
    config: GPTConfig,
) -> GPTModel:

    return GPTModel(
        config
    )


def load_model_structure(
    config_path: str | Path,
) -> GPTModel:

    print(
        "Loading model structure..."
    )

    config = load_config(
        config_path
    )

    model = build_model(
        config
    )

    print(
        "Model structure loaded"
    )

    return model


def load_model_weights(
    model: GPTModel,
    weights_path: str | Path,
) -> GPTModel:

    weights_path = normalize_path(
        weights_path
    )

    print(
        "\n===== WEIGHTS PATH ====="
    )

    print(
        weights_path
    )

    print(
        "========================\n"
    )

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )

    print(
        "Loading safetensors..."
    )

    state_dict = load_file(
        str(weights_path)
    )

    print(
        "Safetensors loaded"
    )

    print(
        "Number of tensors:",
        len(state_dict)
    )

    print(
        "Loading state dict..."
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        "State dict loaded"
    )

    print(
        "\n===== MISSING KEYS ====="
    )

    for key in missing_keys:
        print(key)

    print(
        "\n===== UNEXPECTED KEYS ====="
    )

    for key in unexpected_keys:
        print(key)

    print(
        "\n=========================="
    )

    model.eval()

    return model


def load_model(
    config_path: str | Path,
    weights_path: str | Path,
) -> GPTModel:

    print(
        "\n===== LOAD MODEL START ====="
    )

    model = load_model_structure(
        config_path
    )

    model = load_model_weights(
        model,
        weights_path,
    )

    print(
        "===== LOAD MODEL COMPLETE =====\n"
    )

    return model