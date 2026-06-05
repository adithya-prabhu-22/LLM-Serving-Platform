from core.models.gpt import GPTModel

from backend.services.registry_service import (
    get_model,
)

from backend.services.model_loader import (
    load_model,
)


LOADED_MODELS: dict[
    str,
    GPTModel,
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

    LOADED_MODELS[
        model_id
    ] = model

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
):

    model = get_loaded_model(
        model_id
    )

    raise NotImplementedError(
        "Tokenizer integration "
        "and text generation "
        "not implemented yet."
    )