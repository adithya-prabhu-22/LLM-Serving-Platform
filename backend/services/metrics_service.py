from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

import psutil


REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total requests received",
    ["endpoint"],
)

REQUEST_ERRORS_TOTAL = Counter(
    "llm_request_errors_total",
    "Total failed requests",
    ["endpoint"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "Request latency in seconds",
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently running requests",
)


MODEL_LOAD_LATENCY_SECONDS = Histogram(
    "llm_model_load_latency_seconds",
    "Model load latency in seconds",
)

MODEL_LOAD_FAILURES_TOTAL = Counter(
    "llm_model_load_failures_total",
    "Total model load failures",
)

LOADED_MODELS = Gauge(
    "llm_loaded_models",
    "Currently loaded models",
)


PROMPT_TOKENS_TOTAL = Counter(
    "llm_prompt_tokens_total",
    "Total prompt tokens processed",
)

GENERATED_TOKENS_TOTAL = Counter(
    "llm_generated_tokens_total",
    "Total generated tokens",
)


REQUEST_INPUT_TOKENS = Histogram(
    "llm_request_input_tokens",
    "Prompt size in tokens",
    buckets=(
        1,
        5,
        10,
        20,
        50,
        100,
        250,
        500,
        1000,
        2000,
        4000,
        8000,
        float("inf"),
    ),
)

REQUEST_OUTPUT_TOKENS = Histogram(
    "llm_request_output_tokens",
    "Generated size in tokens",
    buckets=(
        1,
        5,
        10,
        20,
        50,
        100,
        250,
        500,
        1000,
        2000,
        4000,
        float("inf"),
    ),
)


TOKENS_PER_SECOND = Gauge(
    "llm_tokens_per_second",
    "Generation throughput",
)


CPU_USAGE_PERCENT = Gauge(
    "llm_cpu_usage_percent",
    "CPU usage percentage",
)

RAM_USAGE_MB = Gauge(
    "llm_ram_usage_mb",
    "RAM usage in MB",
)


def update_system_metrics():

    CPU_USAGE_PERCENT.set(
        psutil.cpu_percent()
    )

    RAM_USAGE_MB.set(
        psutil.virtual_memory().used
        / (1024 * 1024)
    )