import pytest
import torch
from core.config.gpt_config import GPTConfig
from core.models.gpt import GPTModel
from backend.services.text_generation import generate_tokens

def get_model():
    config = GPTConfig(vocab_size=100, block_size=32, d_model=64, num_heads=4, num_layers=2, cache_type="ring")
    return GPTModel(config)

def test_generate_tokens():
    model = get_model()
    input_ids = torch.randint(low=0, high=100, size=(1, 5))
    output_ids = generate_tokens(model=model, input_ids=input_ids, max_new_tokens=10)
    assert output_ids.shape == (1, 15)
    print("✓ Token generation")

def test_generate_tokens_respects_length():
    model = get_model()
    input_ids = torch.randint(low=0, high=100, size=(1, 8))
    output_ids = generate_tokens(model=model, input_ids=input_ids, max_new_tokens=4)
    assert output_ids.shape == (1, 12)
    print("✓ Output length")

def test_zero_new_tokens():
    model = get_model()
    input_ids = torch.randint(low=0, high=100, size=(1, 5))
    output_ids = generate_tokens(model=model, input_ids=input_ids, max_new_tokens=0)
    assert torch.equal(output_ids, input_ids)
    print("✓ Zero token generation")

def test_invalid_input_shape():
    model = get_model()
    invalid_input = torch.randint(low=0, high=100, size=(5,))
    with pytest.raises(ValueError):
        generate_tokens(model=model, input_ids=invalid_input, max_new_tokens=5)
    print("✓ Input shape validation")

def test_invalid_temperature():
    model = get_model()
    input_ids = torch.randint(low=0, high=100, size=(1, 5))
    with pytest.raises(ValueError):
        generate_tokens(model=model, input_ids=input_ids, max_new_tokens=5, temperature=0.0)
    print("✓ Temperature validation")

def test_invalid_top_k():
    model = get_model()
    input_ids = torch.randint(low=0, high=100, size=(1, 5))
    with pytest.raises(ValueError):
        generate_tokens(model=model, input_ids=input_ids, max_new_tokens=5, top_k=0)
    print("✓ Top-k validation")

def run_all_tests():
    print("\nRunning Text Generation tests...\n")
    test_generate_tokens()
    test_generate_tokens_respects_length()
    test_zero_new_tokens()
    test_invalid_input_shape()
    test_invalid_temperature()
    test_invalid_top_k()
    print("\n✓ All Text Generation tests passed")

if __name__ == "__main__":
    run_all_tests()