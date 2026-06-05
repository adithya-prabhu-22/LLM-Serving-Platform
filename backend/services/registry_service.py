import json
from pathlib import Path


REGISTRY_PATH = Path(
    "storage/registry/models.json"
)


STATUS_VALUES = {
    "UPLOADED",
    "VALIDATING",
    "DEPLOYING",
    "LOADING",
    "READY",
    "FAILED",
}


def _initialize_registry():

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                [],
                file,
                indent=4,
            )


def _load_registry() -> list[dict]:

    _initialize_registry()

    with open(
        REGISTRY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def _save_registry(
    registry_data: list[dict],
):

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry_data,
            file,
            indent=4,
        )


def register_model(
    model_id: str,
    name: str,
    architecture: str,
    config_path: str,
    weights_path: str,
    tokenizer_path: str | None = None,
):

    registry = _load_registry()

    for model in registry:

        if model["model_id"] == model_id:
            raise ValueError(
                f"Model '{model_id}' already exists."
            )

    registry.append(
        {
            "model_id": model_id,
            "name": name,
            "architecture": architecture,
            "status": "UPLOADED",
            "config_path": config_path,
            "weights_path": weights_path,
            "tokenizer_path": tokenizer_path,
        }
    )

    _save_registry(
        registry
    )


def get_model(
    model_id: str,
) -> dict:

    registry = _load_registry()

    for model in registry:

        if model["model_id"] == model_id:
            return model

    raise ValueError(
        f"Model '{model_id}' not found."
    )


def list_models() -> list[dict]:

    return _load_registry()


def update_model_status(
    model_id: str,
    status: str,
):

    if status not in STATUS_VALUES:
        raise ValueError(
            f"Invalid status: {status}"
        )

    registry = _load_registry()

    for model in registry:

        if model["model_id"] == model_id:

            model["status"] = status

            _save_registry(
                registry
            )

            return

    raise ValueError(
        f"Model '{model_id}' not found."
    )


def delete_model(
    model_id: str,
):

    registry = _load_registry()

    updated_registry = [
        model
        for model in registry
        if model["model_id"] != model_id
    ]

    if len(updated_registry) == len(
        registry
    ):
        raise ValueError(
            f"Model '{model_id}' not found."
        )

    _save_registry(
        updated_registry
    )