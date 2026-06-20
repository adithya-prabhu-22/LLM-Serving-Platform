from pathlib import Path
from safetensors.torch import save_file

def save_safetensor(model, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    save_file(state_dict, str(output_path))
    return str(output_path)