from pynvml import *

from backend.services.metrics_service import (
    GPU_UTILIZATION_PERCENT,
    GPU_MEMORY_USED_MB,
    GPU_MEMORY_UTILIZATION_PERCENT,
    GPU_TEMPERATURE_CELSIUS,
    GPU_POWER_WATTS,
)

_initialized = False


def update_gpu_metrics():

    global _initialized

    try:

        if not _initialized:

            nvmlInit()

            _initialized = True

        handle = nvmlDeviceGetHandleByIndex(
            0
        )

        utilization = (
            nvmlDeviceGetUtilizationRates(
                handle
            )
        )

        memory = (
            nvmlDeviceGetMemoryInfo(
                handle
            )
        )

        temperature = (
            nvmlDeviceGetTemperature(
                handle,
                NVML_TEMPERATURE_GPU,
            )
        )

        power = (
            nvmlDeviceGetPowerUsage(
                handle
            )
            / 1000
        )

        GPU_UTILIZATION_PERCENT.set(
            utilization.gpu
        )

        GPU_MEMORY_USED_MB.set(
            memory.used
            / (1024 * 1024)
        )

        GPU_MEMORY_UTILIZATION_PERCENT.set(
            (
                memory.used
                / memory.total
            )
            * 100
        )

        GPU_TEMPERATURE_CELSIUS.set(
            temperature
        )

        GPU_POWER_WATTS.set(
            power
        )

    except Exception:

        GPU_UTILIZATION_PERCENT.set(
            0
        )

        GPU_MEMORY_USED_MB.set(
            0
        )

        GPU_MEMORY_UTILIZATION_PERCENT.set(
            0
        )

        GPU_TEMPERATURE_CELSIUS.set(
            0
        )

        GPU_POWER_WATTS.set(
            0
        )