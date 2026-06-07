from pathlib import Path
from shutil import rmtree

from backend.services.registry_service import (
    register_model,
)


MODEL_STORAGE_DIR = Path(
    "storage/deployed_models"
)


def create_model_directory(
    model_id: str,
):

    model_dir = (
        MODEL_STORAGE_DIR
        / model_id
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return model_dir


def get_config_path(
    model_id: str,
):

    return (
        MODEL_STORAGE_DIR
        / model_id
        / "config.json"
    )


def get_weights_path(
    model_id: str,
):

    return (
        MODEL_STORAGE_DIR
        / model_id
        / "model.safetensors"
    )


def get_tokenizer_path(
    model_id: str,
):

    return (
        MODEL_STORAGE_DIR
        / model_id
        / "tokenizer.json"
    )


def save_file(
    destination_path: Path,
    content: bytes,
):

    with open(
        destination_path,
        "wb",
    ) as file:

        file.write(
            content
        )


def save_model_files(
    model_id: str,
    config_content: bytes,
    weights_content: bytes,
    tokenizer_content: bytes,
):

    create_model_directory(
        model_id
    )

    save_file(
        get_config_path(
            model_id
        ),
        config_content,
    )

    save_file(
        get_weights_path(
            model_id
        ),
        weights_content,
    )

    save_file(
        get_tokenizer_path(
            model_id
        ),
        tokenizer_content,
    )


def get_model_artifacts(
    model_id: str,
):

    return {
        "config_path": str(
            get_config_path(
                model_id
            )
        ),
        "weights_path": str(
            get_weights_path(
                model_id
            )
        ),
        "tokenizer_path": str(
            get_tokenizer_path(
                model_id
            )
        ),
    }


def register_uploaded_model(
    model_id: str,
    name: str,
    architecture: str,
):

    artifacts = (
        get_model_artifacts(
            model_id
        )
    )

    register_model(
        model_id=model_id,
        name=name,
        architecture=architecture,
        config_path=artifacts[
            "config_path"
        ],
        weights_path=artifacts[
            "weights_path"
        ],
        tokenizer_backend="custom_bpe",
        tokenizer_path=artifacts[
            "tokenizer_path"
        ],
    )


def onboard_model(
    model_id: str,
    name: str,
    architecture: str,
    config_content: bytes,
    weights_content: bytes,
    tokenizer_content: bytes,
):

    save_model_files(
        model_id=model_id,
        config_content=config_content,
        weights_content=weights_content,
        tokenizer_content=tokenizer_content,
    )

    register_uploaded_model(
        model_id=model_id,
        name=name,
        architecture=architecture,
    )


def model_directory_exists(
    model_id: str,
) -> bool:

    return (
        MODEL_STORAGE_DIR
        / model_id
    ).exists()


def delete_model_files(
    model_id: str,
):

    model_dir = (
        MODEL_STORAGE_DIR
        / model_id
    )

    if not model_dir.exists():

        raise ValueError(
            f"Model directory not found: "
            f"{model_dir}"
        )

    rmtree(
        model_dir
    )