import torch
import torch.nn.functional as F

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits shape (batch_size, seq_len, vocab_size), but got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets shape (batch_size, seq_len), but got {tuple(targets.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating-point tensor.")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long.")
    batch_size, seq_len, vocab_size = logits.shape
    if targets.shape != (batch_size, seq_len):
        raise ValueError("Targets shape must match logits sequence dimensions.")
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss = F.cross_entropy(logits, targets, ignore_index=-100)
    return loss