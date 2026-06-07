from pydantic import BaseModel


class GenerateResponse(
    BaseModel
):

    response: str