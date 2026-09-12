import tiktoken
from langsmith import traceable

TOKENIZER = tiktoken.get_encoding("gpt2")


@traceable(run_type="tool", name="Tokenizer-Encode")
def encode(text: str) -> list[int]:
    return TOKENIZER.encode(text)


@traceable(run_type="tool", name="Tokenizer-Decode")
def decode(token_ids: list[int]) -> str:
    return TOKENIZER.decode(token_ids)


def vocab_size() -> int:
    return TOKENIZER.n_vocab