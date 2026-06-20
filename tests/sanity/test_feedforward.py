import pytest
import torch
from core.models.feedforward import FeedForward

def test_feedforward_forward():
    model = FeedForward(d_model=768, ff_dim=3072, activation="gelu", dropout=0.1)
    x = torch.randn(2, 8, 768)
    output = model(x)
    assert output.shape == (2, 8, 768)
    print("✓ Forward pass")

def test_all_activations():
    activations = ["gelu", "relu", "silu", "tanh"]
    x = torch.randn(2, 8, 768)
    for activation in activations:
        model = FeedForward(d_model=768, ff_dim=3072, activation=activation, dropout=0.1)
        output = model(x)
        assert output.shape == (2, 8, 768)
    print("✓ Activation support")

def test_default_ff_dim():
    model = FeedForward(d_model=768, activation="gelu")
    assert model.ff_dim == 3072
    print("✓ Default ff_dim")

def test_invalid_activation():
    with pytest.raises(ValueError):
        FeedForward(d_model=768, activation="invalid")
    print("✓ Activation validation")

def test_invalid_input_shape():
    model = FeedForward(d_model=768)
    x = torch.randn(8, 768)
    with pytest.raises(ValueError):
        model(x)
    print("✓ Input shape validation")

def test_invalid_embedding_dimension():
    model = FeedForward(d_model=768)
    x = torch.randn(2, 8, 512)
    with pytest.raises(ValueError):
        model(x)
    print("✓ Embedding dimension validation")

def test_invalid_d_model():
    with pytest.raises(ValueError):
        FeedForward(d_model=0)
    print("✓ d_model validation")

def test_invalid_ff_dim():
    with pytest.raises(ValueError):
        FeedForward(d_model=768, ff_dim=0)
    print("✓ ff_dim validation")

def test_invalid_dropout():
    with pytest.raises(ValueError):
        FeedForward(d_model=768, dropout=1.5)
    print("✓ dropout validation")

def run_all_tests():
    print("\nRunning FeedForward tests...\n")
    test_feedforward_forward()
    test_all_activations()
    test_default_ff_dim()
    test_invalid_activation()
    test_invalid_input_shape()
    test_invalid_embedding_dimension()
    test_invalid_d_model()
    test_invalid_ff_dim()
    test_invalid_dropout()
    print("\n✓ All FeedForward tests passed")

if __name__ == "__main__":
    run_all_tests()