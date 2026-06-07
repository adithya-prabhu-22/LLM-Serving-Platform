from backend.services.registry_service import (
    list_models,
    get_model,
)

from backend.services.inference_engine import (
    build_model,
)


def get_models():

    return list_models()


def get_model_by_id(
    model_id: str,
):

    return get_model(
        model_id
    )


def get_model_status(
    model_id: str,
):

    model = get_model(
        model_id
    )

    return {
        "model_id": model_id,
        "status": model[
            "status"
        ],
    }


def build_model_route(
    model_id: str,
):

    model = get_model(
        model_id
    )

    if (
        model["status"]
        == "READY"
    ):

        return {
            "message":
            f"Model '{model_id}' is already built.",
            "status":
            "READY",
        }

    try:

        build_model(
            model_id
        )

        return {
            "message":
            f"Model '{model_id}' built successfully.",
            "status":
            "READY",
        }

    except Exception as error:

        raise ValueError(
            f"Failed to build model "
            f"'{model_id}': {str(error)}"
        )