import pytest
import torch
from core.models.normalization import LayerNorm

def test_layernorm_forward():
    layer_norm = LayerNorm(d_model=768)
    x = torch.randn(2, 8, 768)
    output = layer_norm(x)
    assert output.shape == (2, 8, 768)
    print("✓ Forward pass")

def test_layernorm_without_bias():
    layer_norm = LayerNorm(d_model=768, bias=False)
    x = torch.randn(2, 8, 768)
    output = layer_norm(x)
    assert output.shape == (2, 8, 768)
    assert layer_norm.beta is None
    print("✓ No bias mode")

def test_layernorm_with_bias():
    layer_norm = LayerNorm(d_model=768, bias=True)
    assert layer_norm.beta is not None
    assert layer_norm.beta.shape == (768,)
    print("✓ Bias mode")

def test_gamma_shape():
    layer_norm = LayerNorm(d_model=768)
    assert layer_norm.gamma.shape == (768,)
    print("✓ Gamma shape")

def test_output_mean_close_to_zero():
    layer_norm = LayerNorm(d_model=768)
    x = torch.randn(2, 8, 768)
    output = layer_norm(x)
    mean = output.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    print("✓ Mean normalization")

def test_output_variance_close_to_one():
    layer_norm = LayerNorm(d_model=768)
    x = torch.randn(2, 8, 768)
    output = layer_norm(x)
    variance = output.var(dim=-1, unbiased=False)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-3)
    print("✓ Variance normalization")

def test_invalid_d_model():
    with pytest.raises(ValueError):
        LayerNorm(d_model=0)
    print("✓ d_model validation")

def run_all_tests():
    print("\nRunning LayerNorm tests...\n")
    test_layernorm_forward()
    test_layernorm_without_bias()
    test_layernorm_with_bias()
    test_gamma_shape()
    test_output_mean_close_to_zero()
    test_output_variance_close_to_one()
    test_invalid_d_model()
    print("\n✓ All LayerNorm tests passed")

if __name__ == "__main__":
    run_all_tests()