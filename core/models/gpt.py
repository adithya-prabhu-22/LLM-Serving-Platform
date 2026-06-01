import torch
import torch.nn as nn

from core.models.embeddings import GPTEmbeddings
from core.models.normalization import LayerNorm
from core.models.transformer_block import (
    TransformerDecoderBlock,
)


class GPTModel(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()

        required_fields = [
            "vocab_size",
            "block_size",
            "d_model",
            "num_heads",
            "num_layers",
            "dropout",
        ]

        for field in required_fields:

            if not hasattr(
                config,
                field,
            ):
                raise ValueError(
                    f"Missing required config field: {field}"
                )

        self.config = config

        self.vocab_size = (
            config.vocab_size
        )

        self.max_len = (
            config.block_size
        )

        self.d_model = (
            config.d_model
        )

        self.num_heads = (
            config.num_heads
        )

        self.num_layers = (
            config.num_layers
        )

        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            block_size=config.block_size,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    ff_dim=getattr(
                        config,
                        "ff_dim",
                        None,
                    ),
                    activation=getattr(
                        config,
                        "activation",
                        "gelu",
                    ),
                    dropout=config.dropout,
                    bias=getattr(
                        config,
                        "qkv_bias",
                        False,
                    ),
                    use_flash_attention=getattr(
                        config,
                        "use_flash_attention",
                        False,
                    ),
                )
                for _ in range(
                    config.num_layers
                )
            ]
        )

        self.final_norm = LayerNorm(
            d_model=config.d_model,
            bias=getattr(
                config,
                "qkv_bias",
                False,
            ),
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = (
            self.embeddings.token_embedding.weight
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids shape "
                f"(batch_size, seq_len), "
                f"but got {tuple(input_ids.shape)}"
            )

        _, seq_len = input_ids.shape

        if past_kv is None:

            past_length = 0

        else:

            if len(past_kv) != self.num_layers:
                raise ValueError(
                    f"Expected past_kv for "
                    f"{self.num_layers} layers, "
                    f"but got {len(past_kv)}."
                )

            past_length = (
                past_kv[0][0].shape[2]
            )

        total_length = (
            past_length + seq_len
        )

        if total_length > self.max_len:
            raise ValueError(
                f"Sequence length "
                f"{total_length} exceeds "
                f"block_size {self.max_len}."
            )

        x = self.embeddings(
            input_ids,
            start_pos=past_length,
        )

        present_kv = (
            []
            if use_cache
            else None
        )

        for idx, block in enumerate(
            self.blocks
        ):

            layer_past = (
                None
                if past_kv is None
                else past_kv[idx]
            )

            if use_cache:

                x, layer_present = block(
                    x,
                    past_kv=layer_past,
                    use_cache=True,
                )

                present_kv.append(
                    layer_present
                )

            else:

                x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        if use_cache:
            return logits, present_kv

        return logits

    def num_parameters(
        self,
        trainable_only: bool = False,
    ) -> int:

        parameters = (
            self.parameters()
        )

        if trainable_only:

            parameters = (
                p
                for p in parameters
                if p.requires_grad
            )

        return sum(
            p.numel()
            for p in parameters
        )