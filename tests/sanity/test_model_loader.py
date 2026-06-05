from backend.services.model_loader import (
    load_config,
    build_model,
    load_model_structure,
)


def test_load_config():

    config = load_config(
        "tests/resources/sample_config.json"
    )

    assert config.vocab_size == 52000
    assert config.block_size == 512
    assert config.d_model == 512
    assert config.num_heads == 8
    assert config.num_layers == 8


def test_build_model():

    config = load_config(
        "tests/resources/sample_config.json"
    )

    model = build_model(
        config
    )

    assert model.vocab_size == 52000
    assert model.max_len == 512
    assert model.d_model == 512
    assert model.num_heads == 8
    assert model.num_layers == 8


def test_load_model_structure():

    model = load_model_structure(
        "tests/resources/sample_config.json"
    )

    assert model.vocab_size == 52000
    assert model.max_len == 512
    assert model.d_model == 512
    assert model.num_heads == 8
    assert model.num_layers == 8