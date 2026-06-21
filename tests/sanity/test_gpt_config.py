import pytest

from core.config.gpt_config import GPTConfig


def test_valid_config():
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        d_model=768,
        num_heads=12,
        num_layers=12,
    )

    assert config.vocab_size == 50257
    assert config.block_size == 1024
    assert config.ff_dim == 3072
    assert config.batch_size == 8
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 0.01


def test_auto_ff_dim():
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        d_model=512,
        num_heads=8,
        num_layers=6,
    )
    assert config.ff_dim == 2048


def test_custom_training_params():
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        d_model=768,
        num_heads=12,
        num_layers=12,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.1,
    )
    assert config.batch_size == 32
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.1


def test_invalid_vocab_size():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=0,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_block_size():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=0,
            d_model=768,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_d_model():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=0,
            num_heads=12,
            num_layers=12,
        )


def test_invalid_num_heads():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=0,
            num_layers=12,
        )


def test_invalid_num_layers():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=0,
        )


def test_invalid_head_divisibility():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=7,
            num_layers=12,
        )


def test_invalid_dropout():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            dropout=1.5,
        )


def test_invalid_ff_dim():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            ff_dim=-1,
        )


def test_empty_activation():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            activation="",
        )


def test_invalid_activation():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            activation="invalid",
        )


def test_invalid_temperature():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            temperature=0,
        )


def test_invalid_top_k():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            top_k=0,
        )


def test_top_k_exceeds_vocab_size():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=100,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            top_k=200,
        )


def test_invalid_cache_type():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            cache_type="invalid",
        )


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            batch_size=0,
        )


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            learning_rate=0,
        )


def test_invalid_weight_decay():
    with pytest.raises(ValueError):
        GPTConfig(
            vocab_size=50257,
            block_size=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
            weight_decay=-1,
        )


def run_all_tests():
    print("\nRunning GPTConfig tests...\n")

    test_valid_config()
    test_auto_ff_dim()
    test_custom_training_params()

    test_invalid_vocab_size()
    test_invalid_block_size()
    test_invalid_d_model()
    test_invalid_num_heads()
    test_invalid_num_layers()
    test_invalid_head_divisibility()
    test_invalid_dropout()
    test_invalid_ff_dim()
    test_empty_activation()
    test_invalid_activation()
    test_invalid_temperature()
    test_invalid_top_k()
    test_top_k_exceeds_vocab_size()
    test_invalid_cache_type()
    test_invalid_batch_size()
    test_invalid_learning_rate()
    test_invalid_weight_decay()

    print("\n✓ All GPTConfig tests passed")


if __name__ == "__main__":
    run_all_tests()