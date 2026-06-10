from pathlib import Path


def validate_config_file(
    filename: str,
):

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension != ".json":

        raise ValueError(
            "Invalid config file. "
            "Expected .json"
        )


def validate_weights_file(
    filename: str,
):

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension != ".safetensors":

        raise ValueError(
            "Invalid weights file. "
            "Expected .safetensors"
        )