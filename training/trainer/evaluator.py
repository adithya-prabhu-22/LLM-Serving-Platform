import math
import torch
from torch.amp import autocast
from training.trainer.loss import compute_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_cuda:
            with autocast(device_type="cuda"):
                logits = model(input_ids)
                loss = compute_loss(logits, targets)
        else:
            logits = model(input_ids)
            loss = compute_loss(logits, targets)
        total_loss += loss.item()
        total_batches += 1
    if total_batches == 0:
        raise ValueError("Evaluation dataloader contains no batches.")
    avg_loss = total_loss / total_batches
    try:
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    except OverflowError:
        perplexity = float("inf")
    if was_training:
        model.train()
    return {"loss": avg_loss, "perplexity": perplexity}