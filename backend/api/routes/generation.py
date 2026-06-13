import time

from backend.services.inference_engine import (
    generate,
)

from backend.services.inference_engine import (
    generate_stream,
)

from backend.services.metrics_service import (
    REQUESTS_TOTAL,
    REQUEST_ERRORS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    ACTIVE_REQUESTS,
)


def generate_text(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
):

    REQUESTS_TOTAL.labels(
        endpoint="/generate"
    ).inc()

    ACTIVE_REQUESTS.inc()

    start_time = time.time()

    try:

        response = generate(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        REQUEST_LATENCY_SECONDS.observe(
            time.time() - start_time
        )

        return {
            "response": response
        }

    except Exception:

        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/generate"
        ).inc()

        raise

    finally:

        ACTIVE_REQUESTS.dec()


def generate_text_stream(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
):

    REQUESTS_TOTAL.labels(
        endpoint="/generate/stream"
    ).inc()

    ACTIVE_REQUESTS.inc()

    start_time = time.time()

    try:

        response = generate_stream(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        REQUEST_LATENCY_SECONDS.observe(
            time.time() - start_time
        )

        return response

    except Exception:

        REQUEST_ERRORS_TOTAL.labels(
            endpoint="/generate/stream"
        ).inc()

        raise

    finally:

        ACTIVE_REQUESTS.dec()