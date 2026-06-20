import pytest
import torch
from core.models.activations import GELU, ReLU, SiLU, Tanh, get_activation

def test_gelu():
    activation = GELU()
    x = torch.randn(2, 8, 768)
    output = activation(x)
    assert output.shape == x.shape
    print("✓ GELU")

def test_relu():
    activation = ReLU()
    x = torch.randn(2, 8, 768)
    output = activation(x)
    assert output.shape == x.shape
    print("✓ ReLU")

def test_silu():
    activation = SiLU()
    x = torch.randn(2, 8, 768)
    output = activation(x)
    assert output.shape == x.shape
    print("✓ SiLU")

def test_tanh():
    activation = Tanh()
    x = torch.randn(2, 8, 768)
    output = activation(x)
    assert output.shape == x.shape
    print("✓ Tanh")

def test_get_activation():
    activations = ["gelu", "relu", "silu", "tanh"]
    for name in activations:
        activation = get_activation(name)
        assert activation is not None
    print("✓ get_activation")

def test_invalid_activation():
    with pytest.raises(ValueError):
        get_activation("invalid_activation")
    print("✓ Invalid activation validation")

def run_all_tests():
    print("\nRunning Activation tests...\n")
    test_gelu()
    test_relu()
    test_silu()
    test_tanh()
    test_get_activation()
    test_invalid_activation()
    print("\n✓ All activation tests passed")

if __name__ == "__main__":
    run_all_tests()