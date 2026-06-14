import torch


@torch.no_grad()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
):

    model.eval()

    for _ in range(
        max_new_tokens
    ):

        context = input_ids[
            :,
            -model.max_len:,
        ]

        logits = model(
            context
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token_logits = (
            next_token_logits
            / temperature
        )

        effective_top_k = min(
            top_k,
            next_token_logits.size(-1),
        )

        top_k_logits, top_k_indices = (
            torch.topk(
                next_token_logits,
                k=effective_top_k,
                dim=-1,
            )
        )

        probabilities = (
            torch.softmax(
                top_k_logits,
                dim=-1,
            )
        )

        sampled_index = (
            torch.multinomial(
                probabilities,
                num_samples=1,
            )
        )

        next_token = (
            top_k_indices.gather(
                -1,
                sampled_index,
            )
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    return input_ids


@torch.no_grad()
def generate_tokens_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
):

    model.eval()

    for _ in range(
        max_new_tokens
    ):

        context = input_ids[
            :,
            -model.max_len:
        ]

        logits = model(
            context
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token_logits = (
            next_token_logits
            / temperature
        )

        effective_top_k = min(
            top_k,
            next_token_logits.size(-1),
        )

        top_k_logits, top_k_indices = (
            torch.topk(
                next_token_logits,
                k=effective_top_k,
                dim=-1,
            )
        )

        probabilities = (
            torch.softmax(
                top_k_logits,
                dim=-1,
            )
        )

        sampled_index = (
            torch.multinomial(
                probabilities,
                num_samples=1,
            )
        )

        next_token = (
            top_k_indices.gather(
                -1,
                sampled_index,
            )
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

        yield next_token.item()