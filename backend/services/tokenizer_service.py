import tiktoken


TOKENIZER = tiktoken.get_encoding(
    "gpt2"
)


def encode(
    text: str,
) -> list[int]:

    return TOKENIZER.encode(
        text
    )


def decode(
    token_ids: list[int],
) -> str:

    return TOKENIZER.decode(
        token_ids
    )


def vocab_size() -> int:

    return TOKENIZER.n_vocab