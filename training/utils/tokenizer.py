import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = TOKENIZER.n_vocab

def encode(text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return TOKENIZER.encode(text)

def decode(token_ids: list[int]) -> str:
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list.")
    return TOKENIZER.decode(token_ids)

def vocab_size() -> int:
    return VOCAB_SIZE

def get_tokenizer() -> tiktoken.Encoding:
    return TOKENIZER