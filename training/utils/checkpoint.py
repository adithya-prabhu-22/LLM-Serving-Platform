from pathlib import Path
import torch

def save_checkpoint(model, optimizer, scheduler, scaler, epoch: int, step: int, loss: float, best_val_loss: float, checkpoint_dir: str, checkpoint_name: str | None = None, config=None):
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)

def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
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
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", None)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Loaded checkpoint (epoch={epoch}, step={step})")
    return {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
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