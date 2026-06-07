from pydantic import BaseModel


class ModelStatusResponse(
    BaseModel
):

    model_id: str

    status: str