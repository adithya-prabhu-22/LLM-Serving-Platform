from pathlib import Path
import json

def validate_config_file(filename: str):
    extension = Path(filename).suffix.lower()
    if extension != ".json":
        raise ValueError("Invalid config file. Expected .json")

def validate_weights_file(filename: str):
    extension = Path(filename).suffix.lower()
    if extension != ".safetensors":
        raise ValueError("Invalid weights file. Expected .safetensors")

def validate_config_content(content: bytes):
    try:
        config = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid JSON configuration.") from exc
    required_fields = {"vocab_size", "block_size", "d_model", "num_heads", "num_layers"}
    missing_fields = required_fields - set(config.keys())
    if missing_fields:
        raise ValueError(f"Missing config fields: {sorted(missing_fields)}")

def validate_weights_content(content: bytes):
    if len(content) == 0:
        raise ValueError("Weights file is empty.")