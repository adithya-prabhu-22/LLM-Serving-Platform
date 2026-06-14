from fastapi import APIRouter
from fastapi.responses import Response

from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from backend.services.metrics_service import (
    update_system_metrics,
)

from backend.services.gpu_metrics_service import (
    update_gpu_metrics,
)

router = APIRouter()


@router.get("/metrics")
def metrics():

    update_system_metrics()

    update_gpu_metrics()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )