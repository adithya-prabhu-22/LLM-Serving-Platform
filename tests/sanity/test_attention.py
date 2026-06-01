import torch

from core.models.attention import (
    MultiHeadCausalSelfAttention,
)


def test_attention_forward():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
        dropout=0.1,
    )

    x = torch.randn(
        2,
        16,
        768,
    )

    output = attention(x)

    assert output.shape == (
        2,
        16,
        768,
    )

    print("✓ Standard attention forward pass")


def test_flash_attention_forward():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
        dropout=0.1,
        use_flash_attention=True,
    )

    x = torch.randn(
        2,
        16,
        768,
    )

    output = attention(x)

    assert output.shape == (
        2,
        16,
        768,
    )

    print("✓ Flash attention forward pass")


def test_attention_kv_cache():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
        dropout=0.1,
    )

    x = torch.randn(
        2,
        8,
        768,
    )

    output, present_kv = attention(
        x,
        use_cache=True,
    )

    assert output.shape == (
        2,
        8,
        768,
    )

    assert present_kv is not None

    keys, values = present_kv

    assert keys.shape == (
        2,
        12,
        8,
        64,
    )

    assert values.shape == (
        2,
        12,
        8,
        64,
    )

    print("✓ KV cache generation")


def test_attention_kv_cache_reuse():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
        dropout=0.1,
    )

    first_input = torch.randn(
        2,
        8,
        768,
    )

    _, past_kv = attention(
        first_input,
        use_cache=True,
    )

    next_token = torch.randn(
        2,
        1,
        768,
    )

    output, new_kv = attention(
        next_token,
        past_kv=past_kv,
        use_cache=True,
    )

    assert output.shape == (
        2,
        1,
        768,
    )

    keys, values = new_kv

    assert keys.shape[2] == 9
    assert values.shape[2] == 9

    print("✓ KV cache reuse")


def test_invalid_input_shape():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
    )

    invalid_input = torch.randn(
        16,
        768,
    )

    try:

        attention(invalid_input)

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        print("✓ Input shape validation")


def test_invalid_embedding_dimension():

    attention = MultiHeadCausalSelfAttention(
        d_model=768,
        num_heads=12,
    )

    invalid_input = torch.randn(
        2,
        16,
        512,
    )

    try:

        attention(invalid_input)

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        print("✓ Embedding dimension validation")


def run_all_tests():

    print(
        "\nRunning MultiHeadCausalSelfAttention tests...\n"
    )

    test_attention_forward()
    test_flash_attention_forward()
    test_attention_kv_cache()
    test_attention_kv_cache_reuse()
    test_invalid_input_shape()
    test_invalid_embedding_dimension()

    print(
        "\n✓ All attention tests passed"
    )


if __name__ == "__main__":
    run_all_tests()