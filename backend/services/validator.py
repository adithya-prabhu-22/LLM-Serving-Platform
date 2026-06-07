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


def validate_tokenizer_file(
    filename: str,
):

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension != ".json":

        raise ValueError(
            "Invalid tokenizer file. "
            "Expected .json"
        )


def validate_tokenizer_name(
    tokenizer_name: str,
):

    if not tokenizer_name:

        raise ValueError(
            "Tokenizer name "
            "cannot be empty."
        )

    if len(
        tokenizer_name.strip()
    ) == 0:

        raise ValueError(
            "Tokenizer name "
            "cannot be empty."
        )


def validate_tokenizer_input(
    tokenizer_file: str | None = None,
    tokenizer_name: str | None = None,
):

    if (
        tokenizer_file is None
        and tokenizer_name is None
    ):
        raise ValueError(
            "Provide either "
            "tokenizer file or "
            "tokenizer name."
        )

    if (
        tokenizer_file is not None
        and tokenizer_name is not None
    ):
        raise ValueError(
            "Provide only one of "
            "tokenizer file or "
            "tokenizer name."
        )

    if tokenizer_file is not None:

        validate_tokenizer_file(
            tokenizer_file
        )

    if tokenizer_name is not None:

        validate_tokenizer_name(
            tokenizer_name
        )