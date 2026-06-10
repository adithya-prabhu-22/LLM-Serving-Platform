import torch

from core.models.gpt import GPTModel

from backend.services.registry_service import (
    get_model,
    update_model_status,
)

from backend.services.model_loader import (
    load_model,
)

from backend.services.text_generation import (
    generate_tokens,
)

from backend.services.tokenizer_service import (
    encode,
    decode,
)


LOADED_MODELS: dict[
    str,
    GPTModel,
] = {}


def build_model(
    model_id: str,
) -> GPTModel:

    model_info = get_model(
        model_id
    )

    print(
        "\n===== MODEL INFO ====="
    )

    print(
        model_info
    )

    print(
        "======================\n"
    )

    update_model_status(
        model_id,
        "LOADING",
    )

    try:

        print(
            "STEP 1: Loading model..."
        )

        model = load_model(
            config_path=model_info[
                "config_path"
            ],
            weights_path=model_info[
                "weights_path"
            ],
        )

        print(
            "STEP 2: Model loaded"
        )

        LOADED_MODELS[
            model_id
        ] = model

        update_model_status(
            model_id,
            "READY",
        )

        print(
            "STEP 3: Model ready"
        )

        return model

    except Exception as error:

        print(
            "\n========== MODEL BUILD FAILED =========="
        )

        print(
            "Exception Type:"
        )

        print(
            type(error)
        )

        print(
            "\nException Message:"
        )

        print(
            str(error)
        )

        print(
            "========================================\n"
        )

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

    token_ids = encode(
        prompt
    )

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

    generated_ids = output_ids[
        len(token_ids):
    ]

    generated_text = decode(
        generated_ids
    )

    return generated_text