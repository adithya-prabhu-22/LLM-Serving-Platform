from pydantic import (
    BaseModel,
    Field,
)


class GenerateRequest(
    BaseModel
):

    model_id: str

    prompt: str

    max_new_tokens: int = 50

    temperature: float | None = Field(
        default=None,
        ge=0.1,
        le=2.0,
        description="Sampling temperature",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Top-K sampling",
    )