import pytest
import torch
from core.models.embeddings import GPTEmbeddings

def test_embedding_forward():
    embeddings = GPTEmbeddings(vocab_size=50257, d_model=768, block_size=1024, dropout=0.1)
    input_ids = torch.randint(low=0, high=50257, size=(2, 16))
    output = embeddings(input_ids)
    assert output.shape == (2, 16, 768)
    print("✓ Forward pass")

def test_embedding_dropout():
    embeddings = GPTEmbeddings(vocab_size=50257, d_model=768, block_size=1024, dropout=0.1)
    input_ids = torch.randint(low=0, high=50257, size=(2, 16))
    output = embeddings(input_ids)
    assert output.shape == (2, 16, 768)
    print("✓ Dropout path")

def test_invalid_input_shape():
    embeddings = GPTEmbeddings(vocab_size=50257, d_model=768, block_size=1024)
    invalid_input = torch.randint(low=0, high=50257, size=(16,))
    with pytest.raises(ValueError):
        embeddings(invalid_input)
    print("✓ Input shape validation")

def test_invalid_vocab_size():
    with pytest.raises(ValueError):
        GPTEmbeddings(vocab_size=0, d_model=768, block_size=1024)
    print("✓ vocab_size validation")

def test_invalid_d_model():
    with pytest.raises(ValueError):
        GPTEmbeddings(vocab_size=50257, d_model=0, block_size=1024)
    print("✓ d_model validation")

def test_invalid_block_size():
    with pytest.raises(ValueError):
        GPTEmbeddings(vocab_size=50257, d_model=768, block_size=0)
    print("✓ block_size validation")

def test_invalid_dropout():
    with pytest.raises(ValueError):
        GPTEmbeddings(vocab_size=50257, d_model=768, block_size=1024, dropout=1.5)
    print("✓ dropout validation")

def run_all_tests():
    print("\nRunning GPTEmbeddings tests...\n")
    test_embedding_forward()
    test_embedding_dropout()
    test_invalid_input_shape()
    test_invalid_vocab_size()
    test_invalid_d_model()
    test_invalid_block_size()
    test_invalid_dropout()
    print("\n✓ All GPTEmbeddings tests passed")

if __name__ == "__main__":
    run_all_tests()