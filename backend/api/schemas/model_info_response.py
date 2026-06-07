from pydantic import BaseModel


class ModelInfoResponse(
    BaseModel
):

    model_id: str

    name: str

    architecture: str

    status: str

    config_path: str

    weights_path: str

    tokenizer_backend: str | None = None

    tokenizer_path: str | None = None