from backend.services.upload_service import (
    onboard_model,
    delete_model_files,
)

from backend.services.registry_service import (
    delete_model,
)

from backend.services.inference_engine import (
    unload_model,
)


def upload_model_route(
    model_id: str,
    name: str,
    architecture: str,
    config_content: bytes,
    weights_content: bytes,
    tokenizer_content: bytes,
):

    onboard_model(
        model_id=model_id,
        name=name,
        architecture=architecture,
        config_content=config_content,
        weights_content=weights_content,
        tokenizer_content=tokenizer_content,
    )

    return {
        "model_id": model_id,
        "status": "REGISTERED",
        "message":
        f"Model '{model_id}' uploaded successfully.",
    }


def delete_model_route(
    model_id: str,
):

    try:

        unload_model(
            model_id
        )

    except Exception:

        pass

    delete_model_files(
        model_id
    )

    delete_model(
        model_id
    )

    return {
        "message":
        f"Model '{model_id}' deleted successfully."
    }