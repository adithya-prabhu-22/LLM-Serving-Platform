from pathlib import Path

from safetensors.torch import (
    save_model,
)


def save_safetensor_checkpoint(
    model,
    save_path: str,
):

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model(
        model,
        str(save_path),
    )

    print(
        f"Saved checkpoint to "
        f"{save_path}"
    )