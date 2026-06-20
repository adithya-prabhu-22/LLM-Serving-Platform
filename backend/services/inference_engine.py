import time

import torch

from core.models.gpt import GPTModel
from backend.services.registry_service import (
    get_model,
    update_model_status,
)
from backend.services.model_loader import load_model
from backend.services.text_generation import (
    generate_tokens,
    generate_tokens_stream,
)
from backend.services.tokenizer_service import (
    encode,
    decode,
)
from backend.services.metrics_service import (
    MODEL_LOAD_LATENCY_SECONDS,
    MODEL_LOAD_FAILURES_TOTAL,
    LOADED_MODELS as LOADED_MODELS_GAUGE,
    TOKENS_PER_SECOND,
    PROMPT_TOKENS_TOTAL,
    GENERATED_TOKENS_TOTAL,
    REQUEST_INPUT_TOKENS,
    REQUEST_OUTPUT_TOKENS,
)

LOADED_MODELS: dict[str, GPTModel] = {}


def build_model(model_id: str) -> GPTModel:
    start_time = time.time()

    model_info = get_model(model_id)

    print("\n===== MODEL INFO =====")
    print(model_info)
    print("======================\n")

    update_model_status(model_id, "LOADING")

    try:
        print("STEP 1: Loading model...")

        model = load_model(
            config_path=model_info["config_path"],
            weights_path=model_info["weights_path"],
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print("STEP 2: Model loaded")

        LOADED_MODELS[model_id] = model
        LOADED_MODELS_GAUGE.set(len(LOADED_MODELS))

        MODEL_LOAD_LATENCY_SECONDS.observe(time.time() - start_time)

        update_model_status(model_id, "READY")

        print("STEP 3: Model ready")

        return model

    except Exception:
        MODEL_LOAD_FAILURES_TOTAL.inc()
        update_model_status(model_id, "FAILED")
        raise


def get_loaded_model(model_id: str) -> GPTModel:
    if model_id not in LOADED_MODELS:
        raise ValueError(f"Model '{model_id}' is not built.")
    return LOADED_MODELS[model_id]


def unload_model(model_id: str) -> None:
    if model_id in LOADED_MODELS:
        del LOADED_MODELS[model_id]
        LOADED_MODELS_GAUGE.set(len(LOADED_MODELS))

    update_model_status(model_id, "REGISTERED")


def is_model_built(model_id: str) -> bool:
    return model_id in LOADED_MODELS


def _validate_generation_length(
    model: GPTModel,
    token_ids: list[int],
    max_new_tokens: int,
) -> int:
    context_window = model.config.block_size
    available_tokens = context_window - len(token_ids)

    if available_tokens <= 0:
        raise ValueError(
            f"Prompt length ({len(token_ids)}) already exceeds "
            f"context window ({context_window})."
        )

    return min(max_new_tokens, available_tokens)


def generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float | None = None,
    top_k: int | None = None,
) -> str:
    if not is_model_built(model_id):
        raise ValueError(f"Model '{model_id}' is not built.")

    model = get_loaded_model(model_id)

    temperature = temperature if temperature is not None else model.config.temperature
    top_k = top_k if top_k is not None else model.config.top_k

    token_ids = encode(prompt)

    max_new_tokens = _validate_generation_length(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
    )

    PROMPT_TOKENS_TOTAL.inc(len(token_ids))
    REQUEST_INPUT_TOKENS.observe(len(token_ids))

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    start_time = time.time()

    output_ids = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    elapsed_time = time.time() - start_time

    output_ids = output_ids[0].tolist()
    generated_ids = output_ids[len(token_ids):]

    GENERATED_TOKENS_TOTAL.inc(len(generated_ids))
    REQUEST_OUTPUT_TOKENS.observe(len(generated_ids))

    if elapsed_time > 0 and len(generated_ids) > 0:
        TOKENS_PER_SECOND.set(len(generated_ids) / elapsed_time)

    return decode(generated_ids)


def generate_stream(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float | None = None,
    top_k: int | None = None,
):
    if not is_model_built(model_id):
        raise ValueError(f"Model '{model_id}' is not built.")

    model = get_loaded_model(model_id)

    temperature = temperature if temperature is not None else model.config.temperature
    top_k = top_k if top_k is not None else model.config.top_k

    token_ids = encode(prompt)

    max_new_tokens = _validate_generation_length(
        model=model,
        token_ids=token_ids,
        max_new_tokens=max_new_tokens,
    )

    PROMPT_TOKENS_TOTAL.inc(len(token_ids))
    REQUEST_INPUT_TOKENS.observe(len(token_ids))

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated_count = 0
    start_time = time.time()

    for token_id in generate_tokens_stream(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    ):
        generated_count += 1
        GENERATED_TOKENS_TOTAL.inc()
        yield decode([token_id])

    elapsed_time = time.time() - start_time

    REQUEST_OUTPUT_TOKENS.observe(generated_count)

    if elapsed_time > 0 and generated_count > 0:
        TOKENS_PER_SECOND.set(generated_count / elapsed_time)