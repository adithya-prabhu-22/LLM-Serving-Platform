import torch

def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    next_token_logits = logits[:, -1, :]
    next_token_logits = next_token_logits / temperature
    effective_top_k = min(top_k, next_token_logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=effective_top_k, dim=-1)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, num_samples=1)
    next_token = top_k_indices.gather(-1, sampled_index)
    return next_token

@torch.no_grad()
def generate_tokens(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
    model.eval()
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch_size, seq_len).")
    if max_new_tokens <= 0:
        return input_ids
    logits, past_kv = model(input_ids, use_cache=True)
    next_token = _sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
    generated_tokens = [next_token]
    current_token = next_token
    for _ in range(max_new_tokens - 1):
        logits, past_kv = model(current_token, past_kv=past_kv, use_cache=True)
        next_token = _sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
        generated_tokens.append(next_token)
        current_token = next_token
    generated_tokens = torch.cat(generated_tokens, dim=1)
    return torch.cat((input_ids, generated_tokens), dim=1)

@torch.no_grad()
def generate_tokens_stream(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
    model.eval()
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch_size, seq_len).")
    if max_new_tokens <= 0:
        return
    logits, past_kv = model(input_ids, use_cache=True)
    next_token = _sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
    yield next_token.item()
    current_token = next_token
    for _ in range(max_new_tokens - 1):
        logits, past_kv = model(current_token, past_kv=past_kv, use_cache=True)
        next_token = _sample_next_token(logits=logits, temperature=temperature, top_k=top_k)
        yield next_token.item()
        current_token = next_token