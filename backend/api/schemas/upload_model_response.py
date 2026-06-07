from pydantic import BaseModel


class UploadModelResponse(
    BaseModel
):

    model_id: str

    status: str

    message: str