from pathlib import Path
import torch
import random
import numpy as np
import boto3
import os

def upload_checkpoint_to_s3(local_path: str, bucket: str = "adithya-medical-llm-dataset", s3_key: str = None):
    if s3_key is None:
        s3_key = os.path.basename(local_path)
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, f"checkpoints/{s3_key}")
        print(f"Uploaded checkpoint to s3://{bucket}/checkpoints/{s3_key}")
        os.remove(local_path)
        print(f"Deleted local checkpoint: {local_path}")
    except Exception as e:
        print(f"Failed to upload checkpoint to S3: {e}")

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    loss: float,
    best_val_loss: float,
    checkpoint_dir: str,
    chunk_index: int = 0,
    checkpoint_name: str = None,
    config=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_name is None:
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    else:
        checkpoint_path = checkpoint_dir / checkpoint_name

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "chunk_index": chunk_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
    }
    torch.save(checkpoint, checkpoint_path)
    upload_checkpoint_to_s3(str(checkpoint_path))
    return str(checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if checkpoint.get("random_rng_state") is not None:
        random.setstate(checkpoint["random_rng_state"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", None)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    chunk_index = checkpoint.get("chunk_index", 0)

    print(f"Loaded checkpoint (epoch={epoch}, step={step}, chunk_index={chunk_index})")
    return {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "chunk_index": chunk_index,
        "config": checkpoint.get("config", None),
    }

def latest_checkpoint(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    checkpoints = []
    for path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
            checkpoints.append((step, path))
        except ValueError:
            continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])

def best_checkpoint(checkpoint_dir: str):
    checkpoint_path = Path(checkpoint_dir) / "best_checkpoint.pt"
    if checkpoint_path.exists():
        return str(checkpoint_path)
    return None