from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE

from backend.services.registry_service import (
    register_model,
    delete_model,
)

from backend.services.tokenizer_loader import (
    load_tokenizer,
)


def test_load_huggingface_tokenizer():

    model_id = (
        "hf_tokenizer_test"
    )

    try:
        delete_model(
            model_id
        )
    except ValueError:
        pass

    register_model(
        model_id=model_id,
        name="HF Model",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
        tokenizer_backend="huggingface",
        tokenizer_path="gpt2",
    )

    tokenizer = load_tokenizer(
        model_id
    )

    assert tokenizer is not None

    delete_model(
        model_id
    )


def test_load_custom_bpe_tokenizer():

    model_id = (
        "custom_bpe_test"
    )

    try:
        delete_model(
            model_id
        )
    except ValueError:
        pass

    tokenizer_dir = Path(
        "tests/resources/tokenizers"
    )

    tokenizer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = (
        tokenizer_dir
        / "tokenizer.json"
    )

    tokenizer = Tokenizer(
        BPE()
    )

    tokenizer.save(
        str(
            tokenizer_path
        )
    )

    register_model(
        model_id=model_id,
        name="Custom BPE Model",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
        tokenizer_backend="custom_bpe",
        tokenizer_path=str(
            tokenizer_path
        ),
    )

    loaded_tokenizer = (
        load_tokenizer(
            model_id
        )
    )

    assert (
        loaded_tokenizer
        is not None
    )

    delete_model(
        model_id
    )