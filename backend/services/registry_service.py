import json
import boto3
from pathlib import Path
from typing import List, Dict, Any

REGISTRY_PATH = Path("storage/registry/models.json")
STATUS_VALUES = {"REGISTERED", "LOADING", "READY", "FAILED"}

def _get_s3_client():
    return boto3.client('s3')

def list_s3_models(bucket: str = "adithya-medical-llm-dataset") -> List[str]:
    s3 = _get_s3_client()
    model_names = set()
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='models/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            if len(parts) >= 3 and parts[2].endswith('.safetensors'):
                model_names.add(parts[1])
    return sorted(list(model_names))

def get_s3_model_metadata(model_name: str, bucket: str = "adithya-medical-llm-dataset") -> Dict[str, Any]:
    s3 = _get_s3_client()
    try:
        key = f'models/{model_name}/metadata.json'
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read())
    except s3.exceptions.NoSuchKey:
        pass
    try:
        key = f'models/{model_name}/config.json'
        response = s3.get_object(Bucket=bucket, Key=key)
        config = json.loads(response['Body'].read())
        return {
            "name": model_name,
            "model_type": "GPT",
            "d_model": config.get('d_model'),
            "num_layers": config.get('num_layers'),
            "block_size": config.get('block_size'),
            "description": "Model loaded from S3 (metadata inferred from config)"
        }
    except s3.exceptions.NoSuchKey:
        return {"name": model_name, "error": "Metadata and config not found"}

def s3_model_exists(model_name: str, bucket: str = "adithya-medical-llm-dataset") -> bool:
    s3 = _get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=f'models/{model_name}/model.safetensors')
        return True
    except s3.exceptions.ClientError:
        return False

def _initialize_registry():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)

def _load_registry() -> list[dict]:
    _initialize_registry()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Registry must contain a list.")
        return data
    except (json.JSONDecodeError, ValueError):
        _save_registry([])
        return []

def _save_registry(registry_data: list[dict]):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry_data, file, indent=4)

def register_model(model_id: str, name: str, architecture: str, config_path: str, weights_path: str):
    registry = _load_registry()
    for model in registry:
        if model["model_id"] == model_id:
            raise ValueError(f"Model '{model_id}' already exists.")
    registry.append({
        "model_id": model_id,
        "name": name,
        "architecture": architecture,
        "status": "REGISTERED",
        "config_path": config_path,
        "weights_path": weights_path,
    })
    _save_registry(registry)

def get_model(model_id: str) -> dict:
    registry = _load_registry()
    for model in registry:
        if model["model_id"] == model_id:
            return model
    raise ValueError(f"Model '{model_id}' not found.")

def list_models() -> list[dict]:
    return _load_registry()

def update_model_status(model_id: str, status: str):
    if status not in STATUS_VALUES:
        raise ValueError(f"Invalid status: {status}")
    registry = _load_registry()
    for model in registry:
        if model["model_id"] == model_id:
            model["status"] = status
            _save_registry(registry)
            return
    raise ValueError(f"Model '{model_id}' not found.")

def delete_model(model_id: str):
    registry = _load_registry()
    updated_registry = [model for model in registry if model["model_id"] != model_id]
    if len(updated_registry) == len(registry):
        raise ValueError(f"Model '{model_id}' not found.")
    _save_registry(updated_registry)

def is_model_ready(model_id: str) -> bool:
    model = get_model(model_id)
    return model["status"] == "READY"

def is_model_registered(model_id: str) -> bool:
    model = get_model(model_id)
    return model["status"] == "REGISTERED"