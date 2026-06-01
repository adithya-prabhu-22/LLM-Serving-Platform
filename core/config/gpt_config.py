from dataclasses import dataclass


@dataclass
class GPTConfig:

    vocab_size: int
    block_size: int

    d_model: int
    num_heads: int
    num_layers: int

    dropout: float = 0.1

    ff_dim: int | None = None
    activation: str = "gelu"

    qkv_bias: bool = False

    use_flash_attention: bool = False

    def __post_init__(self):

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if self.block_size <= 0:
            raise ValueError(
                "block_size must be positive."
            )

        if self.d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        if (
            self.ff_dim is not None
            and self.ff_dim <= 0
        ):
            raise ValueError(
                "ff_dim must be positive."
            )

        if not self.activation.strip():
            raise ValueError(
                "activation cannot be empty."
            )

        if self.ff_dim is None:
            self.ff_dim = (
                4 * self.d_model
            )