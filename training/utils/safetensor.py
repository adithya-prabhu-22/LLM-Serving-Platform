from pathlib import Path
from safetensors.torch import save_model


def save_safetensor(model, output_path: str) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_path))
    return str(output_path)