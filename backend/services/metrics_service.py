from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
)

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total requests received",
)

REQUEST_ERRORS_TOTAL = Counter(
    "llm_request_errors_total",
    "Total failed requests",
)

REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "Request latency in seconds",
)

MODEL_LOAD_LATENCY_SECONDS = Histogram(
    "llm_model_load_latency_seconds",
    "Model load latency in seconds",
)

LOADED_MODELS = Gauge(
    "llm_loaded_models",
    "Currently loaded models",
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


import psutil


def update_system_metrics():

    CPU_USAGE_PERCENT.set(
        psutil.cpu_percent()
    )

    RAM_USAGE_MB.set(
        psutil.virtual_memory().used
        / (1024 * 1024)
    )