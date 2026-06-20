import pytest
import torch
from core.cache.ring_kv_cache import RingKVCache
from core.models.transformer_block import TransformerDecoderBlock

def test_transformer_block_forward():
    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )
    x = torch.randn(2, 8, 768)
    output = block(x)
    assert output.shape == (2, 8, 768)
    print("✓ Transformer block forward pass")

def test_transformer_block_flash_attention():
    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
        use_flash_attention=True,
    )
    x = torch.randn(2, 8, 768)
    output = block(x)
    assert output.shape == (2, 8, 768)
    print("✓ Transformer block flash attention")

def test_transformer_block_with_ring_cache():
    block = TransformerDecoderBlock(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        activation="gelu",
        dropout=0.1,
    )
    cache = RingKVCache(
        batch_size=2,
        num_heads=12,
        head_dim=64,
        max_seq_len=32,
    )
    x = torch.randn(2, 8, 768)
    output, present_kv = block(x, past_kv=cache, use_cache=True)
    assert output.shape == (2, 8, 768)
    assert present_kv is cache
    assert len(present_kv) == 8
    print("✓ Transformer block cache support")

def test_invalid_input_shape():
    block = TransformerDecoderBlock(d_model=768, num_heads=12)
    x = torch.randn(8, 768)
    with pytest.raises(ValueError):
        block(x)
    print("✓ Input shape validation")

def test_invalid_embedding_dimension():
    block = TransformerDecoderBlock(d_model=768, num_heads=12)
    x = torch.randn(2, 8, 512)
    with pytest.raises(ValueError):
        block(x)
    print("✓ Embedding dimension validation")

def test_invalid_d_model():
    with pytest.raises(ValueError):
        TransformerDecoderBlock(d_model=0, num_heads=12)
    print("✓ d_model validation")

def test_invalid_num_heads():
    with pytest.raises(ValueError):
        TransformerDecoderBlock(d_model=768, num_heads=0)
    print("✓ num_heads validation")

def test_invalid_head_divisibility():
    with pytest.raises(ValueError):
        TransformerDecoderBlock(d_model=768, num_heads=7)
    print("✓ head divisibility validation")

def run_all_tests():
    print("\nRunning TransformerDecoderBlock tests...\n")
    test_transformer_block_forward()
    test_transformer_block_flash_attention()
    test_transformer_block_with_ring_cache()
    test_invalid_input_shape()
    test_invalid_embedding_dimension()
    test_invalid_d_model()
    test_invalid_num_heads()
    test_invalid_head_divisibility()
    print("\n✓ All transformer block tests passed")

if __name__ == "__main__":
    run_all_tests()