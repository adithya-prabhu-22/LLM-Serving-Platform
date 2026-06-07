from backend.services.inference_engine import (
    generate,
)


def generate_text(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
):

    response = generate(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    return {
        "response": response
    }