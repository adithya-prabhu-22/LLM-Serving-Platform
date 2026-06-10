import torch.nn as nn


class GPTLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = (
            nn.CrossEntropyLoss()
        )

    def forward(
        self,
        logits,
        targets,
    ):

        vocab_size = (
            logits.size(-1)
        )

        logits = logits.view(
            -1,
            vocab_size,
        )

        targets = targets.view(
            -1
        )

        return self.loss_fn(
            logits,
            targets,
        )