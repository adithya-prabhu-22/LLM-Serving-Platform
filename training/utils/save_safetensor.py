from pathlib import Path

from safetensors.torch import (
    save_file,
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

    save_file(
        model.state_dict(),
        str(save_path),
    )