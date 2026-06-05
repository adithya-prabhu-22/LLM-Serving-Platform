import json
from pathlib import Path

from safetensors.torch import load_file

from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel


def load_config(
    config_path: str | Path,
) -> GPTConfig:

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config_data = json.load(file)

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

    config = load_config(
        config_path
    )

    model = build_model(
        config
    )

    return model


def load_model_weights(
    model: GPTModel,
    weights_path: str | Path,
) -> GPTModel:

    weights_path = Path(
        weights_path
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )

    state_dict = load_file(
        str(weights_path)
    )

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.eval()

    return model


def load_model(
    config_path: str | Path,
    weights_path: str | Path,
) -> GPTModel:

    model = load_model_structure(
        config_path
    )

    model = load_model_weights(
        model,
        weights_path,
    )

    return model