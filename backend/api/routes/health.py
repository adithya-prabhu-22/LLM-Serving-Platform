from backend.services.registry_service import (
    list_models,
)

from backend.services.inference_engine import (
    LOADED_MODELS,
)


def get_health():

    registered_models = len(
        list_models()
    )

    loaded_models = len(
        LOADED_MODELS
    )

    return {
        "status": "healthy",
        "registered_models":
        registered_models,
        "loaded_models":
        loaded_models,
        "version": "0.2.0",
    }