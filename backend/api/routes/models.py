from backend.services.registry_service import (
    list_models,
    get_model,
    list_s3_models,
    s3_model_exists,
)
from backend.services.model_loader import load_model_from_s3
from backend.services.inference_engine import (
    build_model,
    is_model_built,
    load_model as load_model_local,
)
from backend.services.metrics_service import (
    REQUESTS_TOTAL,
    REQUEST_ERRORS_TOTAL,
)


def get_models():
    return list_models()


def get_model_by_id(model_id: str):
    return get_model(model_id)


def get_model_status(model_id: str):
    model = get_model(model_id)
    return {
        "model_id": model_id,
        "status": model["status"],
    }


def build_model_route(model_id: str):
    REQUESTS_TOTAL.labels(endpoint="/models/build").inc()
    get_model(model_id)
    if is_model_built(model_id):
        return {
            "message": f"Model '{model_id}' is already built.",
            "status": "READY",
        }
    try:
        build_model(model_id)
        return {
            "message": f"Model '{model_id}' built successfully.",
            "status": "READY",
        }
    except Exception as error:
        REQUEST_ERRORS_TOTAL.labels(endpoint="/models/build").inc()
        raise ValueError(f"Failed to build model '{model_id}': {str(error)}")


def list_s3_models_route(bucket: str = "adithya-medical-llm-dataset"):
    """List all models available in S3."""
    REQUESTS_TOTAL.labels(endpoint="/models/s3").inc()
    try:
        models = list_s3_models(bucket)
        return {"models": models, "count": len(models)}
    except Exception as error:
        REQUEST_ERRORS_TOTAL.labels(endpoint="/models/s3").inc()
        raise ValueError(f"Failed to list S3 models: {str(error)}")


def load_s3_model_route(model_name: str, bucket: str = "adithya-medical-llm-dataset"):
    """Load a model from S3 (download if not cached)."""
    REQUESTS_TOTAL.labels(endpoint="/models/load").inc()
    # Check if model exists in S3
    if not s3_model_exists(model_name, bucket):
        raise ValueError(f"Model '{model_name}' not found in S3.")
    try:
        model = load_model_from_s3(model_name, bucket)
        # Optionally update local registry status to "READY"
        return {
            "message": f"Model '{model_name}' loaded successfully from S3.",
            "status": "READY",
            "model_name": model_name,
        }
    except Exception as error:
        REQUEST_ERRORS_TOTAL.labels(endpoint="/models/load").inc()
        raise ValueError(f"Failed to load model '{model_name}': {str(error)}")