import torch

from core.models.gpt import GPTModel

from backend.services.registry_service import (
    get_model,
    update_model_status,
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


def build_model(
    model_id: str,
) -> GPTModel:

    model_info = get_model(
        model_id
    )

    update_model_status(
        model_id,
        "LOADING",
    )

    try:

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

        update_model_status(
            model_id,
            "READY",
        )

        return model

    except Exception:

        update_model_status(
            model_id,
            "FAILED",
        )

        raise


def get_loaded_model(
    model_id: str,
) -> GPTModel:

    if (
        model_id
        not in LOADED_MODELS
    ):
        raise ValueError(
            f"Model '{model_id}' "
            f"is not built."
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
            f"is not built."
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

    update_model_status(
        model_id,
        "REGISTERED",
    )


def is_model_built(
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

    if not is_model_built(
        model_id
    ):
        raise ValueError(
            f"Model '{model_id}' "
            f"is not built."
        )

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

    print(
        "\n========== GENERATION DEBUG =========="
    )

    print(
        "Model:",
        model_id
    )

    print(
        "Requested Max Tokens:",
        max_new_tokens
    )

    print(
        "Input Tokens:",
        len(token_ids)
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

    generated_ids = output_ids[
        len(token_ids):
    ]

    print(
        "Output Tokens:",
        len(output_ids)
    )

    print(
        "Generated Tokens:",
        len(generated_ids)
    )

    print(
        "=====================================\n"
    )

    generated_text = (
        tokenizer.decode(
            generated_ids
        )
    )

    return generated_text
