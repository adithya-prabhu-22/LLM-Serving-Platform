import json
import os
import boto3
from pathlib import Path
from safetensors.torch import load_file
from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel

def normalize_path(path: str | Path) -> Path:
    return Path(str(path).replace("\\", "/")).resolve()

def load_config(config_path: str | Path) -> GPTConfig:
    config_path = normalize_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = json.load(file)
    return GPTConfig(**config_data)

def build_model(config: GPTConfig) -> GPTModel:
    return GPTModel(config)

def load_model_structure(config_path: str | Path) -> GPTModel:
    config = load_config(config_path)
    model = build_model(config)
    return model

def load_model_weights(model: GPTModel, weights_path: str | Path) -> GPTModel:
    weights_path = normalize_path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    state_dict = load_file(str(weights_path))
    if "lm_head.weight" not in state_dict and "embeddings.token_embedding.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embeddings.token_embedding.weight"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def load_model(config_path: str | Path, weights_path: str | Path) -> GPTModel:
    model = load_model_structure(config_path)
    model = load_model_weights(model, weights_path)
    return model

def _get_s3_client():
    return boto3.client('s3')

def _download_model_from_s3(model_name: str, bucket: str, target_dir: Path):
    s3 = _get_s3_client()
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'models/{model_name}/'
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = os.path.basename(key)
            if filename in ['model.safetensors', 'config.json', 'metadata.json']:
                local_path = target_dir / filename
                s3.download_file(bucket, key, str(local_path))
                print(f"Downloaded {key} to {local_path}")

def ensure_model_cached(model_name: str, bucket: str = "adithya-medical-llm-dataset", cache_dir: str = "storage/deployed_models") -> Path:
    cache_dir = Path(cache_dir)
    model_dir = cache_dir / model_name
    model_path = model_dir / 'model.safetensors'
    if not model_path.exists():
        print(f"Model {model_name} not found locally. Downloading from S3...")
        _download_model_from_s3(model_name, bucket, model_dir)
    return model_dir

def load_model_from_s3(model_name: str, bucket: str = "adithya-medical-llm-dataset", cache_dir: str = "storage/deployed_models") -> GPTModel:
    model_dir = ensure_model_cached(model_name, bucket, cache_dir)
    config_path = model_dir / 'config.json'
    weights_path = model_dir / 'model.safetensors'
    return load_model(config_path, weights_path)