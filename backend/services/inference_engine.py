import torch

from core.models.gpt import GPTModel

from backend.services.registry_service import (
    get_model,
)

from backend.services.model_loader import (
    load_model,
)

from backend.services.tokenizer_loader import (
    load_tokenizer,
)

from backend.services.text_generation import (
    generate_tokens,
)


LOADED_MODELS: dict[
    str,
    GPTModel,
] = {}

LOADED_TOKENIZERS: dict[
    str,
    object,
] = {}


def load_registered_model(
    model_id: str,
) -> GPTModel:

    model_info = get_model(
        model_id
    )

    if (
        model_info["status"]
        != "READY"
    ):
        raise ValueError(
            f"Model '{model_id}' "
            f"is not ready."
        )

    model = load_model(
        config_path=model_info[
            "config_path"
        ],
        weights_path=model_info[
            "weights_path"
        ],
    )

    tokenizer = load_tokenizer(
        model_id
    )

    LOADED_MODELS[
        model_id
    ] = model

    LOADED_TOKENIZERS[
        model_id
    ] = tokenizer

    return model


def get_loaded_model(
    model_id: str,
) -> GPTModel:

    if (
        model_id
        not in LOADED_MODELS
    ):
        raise ValueError(
            f"Model '{model_id}' "
            f"is not loaded."
        )

    return LOADED_MODELS[
        model_id
    ]


def get_loaded_tokenizer(
    model_id: str,
):

    if (
        model_id
        not in LOADED_TOKENIZERS
    ):
        raise ValueError(
            f"Tokenizer for "
            f"'{model_id}' "
            f"is not loaded."
        )

    return LOADED_TOKENIZERS[
        model_id
    ]


def unload_model(
    model_id: str,
):

    if (
        model_id
        in LOADED_MODELS
    ):
        del LOADED_MODELS[
            model_id
        ]

    if (
        model_id
        in LOADED_TOKENIZERS
    ):
        del LOADED_TOKENIZERS[
            model_id
        ]


def is_model_loaded(
    model_id: str,
) -> bool:

    return (
        model_id
        in LOADED_MODELS
    )


def generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
):

    model = get_loaded_model(
        model_id
    )

    tokenizer = (
        get_loaded_tokenizer(
            model_id
        )
    )

    encoded = tokenizer.encode(
        prompt
    )

    if hasattr(
        encoded,
        "ids",
    ):
        token_ids = (
            encoded.ids
        )
    else:
        token_ids = encoded

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )

    output_ids = (
        output_ids[0]
        .tolist()
    )

    return tokenizer.decode(
        output_ids
    )