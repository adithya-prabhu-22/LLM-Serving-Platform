from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from backend.services.registry_service import (
    get_model,
)


SUPPORTED_BACKENDS = {
    "huggingface",
    "custom_bpe",
}


def load_tokenizer(
    model_id: str,
):

    model = get_model(
        model_id
    )

    tokenizer_backend = model.get(
        "tokenizer_backend"
    )

    tokenizer_path = model.get(
        "tokenizer_path"
    )

    if tokenizer_backend is None:
        raise ValueError(
            f"No tokenizer backend configured "
            f"for model '{model_id}'."
        )

    if tokenizer_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported tokenizer backend: "
            f"{tokenizer_backend}"
        )

    if tokenizer_path is None:
        raise ValueError(
            f"No tokenizer path configured "
            f"for model '{model_id}'."
        )

    if tokenizer_backend == "huggingface":

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                tokenizer_path
            )
        )

        return tokenizer

    tokenizer_path = Path(
        tokenizer_path
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: "
            f"{tokenizer_path}"
        )

    tokenizer = (
        Tokenizer
        .from_file(
            str(
                tokenizer_path
            )
        )
    )

    return tokenizer