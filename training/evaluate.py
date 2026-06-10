import math

import torch


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    for (
        input_ids,
        targets,
    ) in dataloader:

        input_ids = input_ids.to(
            device
        )

        targets = targets.to(
            device
        )

        logits = model(
            input_ids
        )

        loss = criterion(
            logits,
            targets,
        )

        total_loss += (
            loss.item()
        )

    avg_loss = (
        total_loss
        / len(dataloader)
    )

    perplexity = math.exp(
        avg_loss
    )

    return (
        avg_loss,
        perplexity,
    )