import torch
from core.cache.ring_kv_cache import RingKVCache
from core.models.attention import MultiHeadCausalSelfAttention

def test_attention_forward():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12, dropout=0.1)
    x = torch.randn(2, 16, 768)
    output = attention(x)
    assert output.shape == (2, 16, 768)
    print("✓ Standard attention forward pass")

def test_flash_attention_forward():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12, dropout=0.1, use_flash_attention=True)
    x = torch.randn(2, 16, 768)
    output = attention(x)
    assert output.shape == (2, 16, 768)
    print("✓ Flash attention forward pass")

def test_attention_use_cache_without_cache_object():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12)
    x = torch.randn(2, 8, 768)
    output, present_kv = attention(x, use_cache=True)
    assert output.shape == (2, 8, 768)
    assert present_kv is None
    print("✓ Attention cache mode without cache object")

def test_attention_with_ring_cache():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12)
    cache = RingKVCache(batch_size=2, num_heads=12, head_dim=64, max_seq_len=32)
    x = torch.randn(2, 8, 768)
    output, returned_cache = attention(x, past_kv=cache, use_cache=True)
    assert output.shape == (2, 8, 768)
    assert returned_cache is cache
    assert cache.get_keys().shape == (2, 12, 8, 64)
    assert cache.get_values().shape == (2, 12, 8, 64)
    print("✓ Ring cache attention")

def test_invalid_input_shape():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12)
    invalid_input = torch.randn(16, 768)
    try:
        attention(invalid_input)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        print("✓ Input shape validation")

def test_invalid_embedding_dimension():
    attention = MultiHeadCausalSelfAttention(d_model=768, num_heads=12)
    invalid_input = torch.randn(2, 16, 512)
    try:
        attention(invalid_input)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        print("✓ Embedding dimension validation")

def run_all_tests():
    print("\nRunning MultiHeadCausalSelfAttention tests...\n")
    test_attention_forward()
    test_flash_attention_forward()
    test_attention_use_cache_without_cache_object()
    test_attention_with_ring_cache()
    test_invalid_input_shape()
    test_invalid_embedding_dimension()
    print("\n✓ All attention tests passed")

if __name__ == "__main__":
    run_all_tests()