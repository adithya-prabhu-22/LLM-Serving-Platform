from pydantic import BaseModel


class GenerateRequest(
    BaseModel
):

    model_id: str

    prompt: str

    max_new_tokens: int = 50