"""
VERIFICATION TESTS FOR KV-CACHE IMPLEMENTATION

Test 1: forward_with_kv produces identical logits to forward
Test 2: forward_incremental produces identical logits to forward at same positions
Test 3: KV cache shapes are correct
Test 4: Incremental extends cache by 4 positions
Test 5: embed_partial matches full embedding at same positions
Test 6: Hierarchical mask rows are correctly extracted
Test 7: No NaN or Inf in cached outputs

Run: pytest tests/test_kv_cache.py -v --timeout=60
"""

import torch
import pytest

from world_model import (
    HierarchicalWorldModel,
    hierarchical_causal_mask,
)


@pytest.fixture
def model_and_data(tiny_config):
    """Create model and test data. Function-scoped for isolation."""
    torch.manual_seed(42)
    
    model = HierarchicalWorldModel(tiny_config)
    model.eval()
    
    B, T = 2, 4  # 4 timesteps × 4 positions = 16, fits in max_seq_len=128
    tokens = torch.randint(0, tiny_config.num_codes, (B, T, 3))
    actions = torch.randint(0, tiny_config.num_actions, (B, T))
    
    return model, tokens, actions


# TEST 1: forward_with_kv matches forward

def test_forward_with_kv_matches_forward(model_and_data):
    """forward_with_kv must produce IDENTICAL logits to forward."""
    model, tokens, actions = model_and_data
    
    with torch.no_grad():
        l0_orig, l1_orig, l2_orig = model(tokens, actions)
        l0_kv, l1_kv, l2_kv, kv_cache, x_hidden = model.forward_with_kv(tokens, actions)
    
    assert torch.allclose(l0_orig, l0_kv, atol=1e-5), \
        f"L0 logits differ! Max diff: {(l0_orig - l0_kv).abs().max():.6f}"
    assert torch.allclose(l1_orig, l1_kv, atol=1e-5), \
        f"L1 logits differ! Max diff: {(l1_orig - l1_kv).abs().max():.6f}"
    assert torch.allclose(l2_orig, l2_kv, atol=1e-5), \
        f"L2 logits differ! Max diff: {(l2_orig - l2_kv).abs().max():.6f}"


# TEST 2: forward_incremental matches forward at new positions

def test_forward_incremental_matches_forward(model_and_data, tiny_config):
    """Incremental forward for last timestep must match full forward."""
    model, tokens, actions = model_and_data
    B, T = tokens.shape[:2]
    
    with torch.no_grad():
        tokens_prefix = tokens[:, :-1, :]
        actions_prefix = actions[:, :-1]
        tokens_last = tokens[:, -1:, :]
        actions_last = actions[:, -1:]
        
        # Full forward over entire sequence
        l0_full, l1_full, l2_full = model(tokens, actions)
        
        # KV-cached: prefix then incremental
        _, _, _, kv_cache, x_hidden = model.forward_with_kv(tokens_prefix, actions_prefix)
        prefix_seq_len = (T - 1) * 4
        
        x_new, _ = model.forward_incremental(
            tokens_last, actions_last, kv_cache, prefix_seq_len
        )
        
        # Extract logits from incremental output
        logits_l0_inc = model.headl0(x_new[:, 3, :])
        logits_l1_inc = model.headl1(x_new[:, 0, :])
        logits_l2_inc = model.headl2(x_new[:, 1, :])
    
    assert torch.allclose(logits_l0_inc, l0_full[:, -1, :], atol=1e-4), \
        f"L0 incremental mismatch! Max diff: {(logits_l0_inc - l0_full[:, -1, :]).abs().max():.6f}"
    assert torch.allclose(logits_l1_inc, l1_full[:, -1, :], atol=1e-4), \
        f"L1 incremental mismatch! Max diff: {(logits_l1_inc - l1_full[:, -1, :]).abs().max():.6f}"
    assert torch.allclose(logits_l2_inc, l2_full[:, -1, :], atol=1e-4), \
        f"L2 incremental mismatch! Max diff: {(logits_l2_inc - l2_full[:, -1, :]).abs().max():.6f}"


# TEST 3: KV cache shapes

def test_kv_cache_shapes(model_and_data, tiny_config):
    """KV cache should have correct shapes."""
    model, tokens, actions = model_and_data
    B, T = tokens.shape[:2]
    
    with torch.no_grad():
        _, _, _, kv_cache, x_hidden = model.forward_with_kv(tokens, actions)
    
    assert len(kv_cache) == tiny_config.n_layers
    
    d_head = tiny_config.d_model // tiny_config.n_heads
    expected_len = T * 4
    
    for i, (K, V) in enumerate(kv_cache):
        assert K.shape == (B, tiny_config.n_heads, expected_len, d_head), \
            f"Layer {i} K shape wrong: {K.shape}"
        assert V.shape == (B, tiny_config.n_heads, expected_len, d_head), \
            f"Layer {i} V shape wrong: {V.shape}"


# TEST 4: Incremental extends cache

def test_incremental_extends_cache(model_and_data, tiny_config):
    """After incremental forward, cache should grow by 4 positions."""
    model, tokens, actions = model_and_data
    B, T = tokens.shape[:2]
    
    with torch.no_grad():
        _, _, _, kv_cache, x_hidden = model.forward_with_kv(tokens[:, :-1, :], actions[:, :-1])
        prefix_len = (T - 1) * 4
        
        _, updated_cache = model.forward_incremental(
            tokens[:, -1:, :], actions[:, -1:], kv_cache, prefix_len
        )
    
    expected_new_len = T * 4
    
    for i, (K, V) in enumerate(updated_cache):
        assert K.shape[2] == expected_new_len, \
            f"Layer {i} K not extended: {K.shape[2]} != {expected_new_len}"
        assert V.shape[2] == expected_new_len, \
            f"Layer {i} V not extended: {V.shape[2]} != {expected_new_len}"


# TEST 5: embed_partial matches full embedding

def test_embed_partial_matches_full(model_and_data):
    """embed_partial for last timestep should match full embedding at same positions."""
    model, tokens, actions = model_and_data
    B, T = tokens.shape[:2]
    
    with torch.no_grad():
        full_emb = model.embedding(tokens, actions)
        
        last_emb = model.embedding.embed_partial(
            tokens[:, -1:, :], actions[:, -1:], start_pos=(T-1)*4
        )
    
    assert torch.allclose(full_emb[:, -4:, :], last_emb, atol=1e-6), \
        f"embed_partial mismatch! Max diff: {(full_emb[:, -4:, :] - last_emb).abs().max():.6f}"



# TEST 6: Mask row extraction

def test_mask_row_extraction():
    """Mask rows extracted for incremental forward should match full mask."""
    seq_len = 32  # 8 timesteps × 4
    device = torch.device('cpu')
    
    full_mask = hierarchical_causal_mask(seq_len, device)
    
    new_start = seq_len - 4
    mask_rows = full_mask[new_start:seq_len, :seq_len]
    
    assert torch.allclose(mask_rows, full_mask[new_start:, :])
    
    # Verify L1 position of last timestep blocks past L1/L2
    l1_row = mask_rows[1, :]
    
    for t in range(seq_len // 4 - 1):
        l1_pos = t * 4 + 1
        l2_pos = t * 4 + 2
        assert l1_row[l1_pos] == float('-inf'), \
            f"L1 should block past L1 at pos {l1_pos}"
        assert l1_row[l2_pos] == float('-inf'), \
            f"L1 should block past L2 at pos {l2_pos}"


# TEST 7: No NaN or Inf

def test_no_nan_inf_in_cached_outputs(model_and_data):
    """KV-cached forward should produce no NaN or Inf values."""
    model, tokens, actions = model_and_data
    
    with torch.no_grad():
        l0, l1, l2, kv_cache, x_hidden = model.forward_with_kv(tokens, actions)
    
    for name, t in [("l0", l0), ("l1", l1), ("l2", l2)]:
        assert not torch.isnan(t).any(), f"NaN in {name} logits"
        assert not torch.isinf(t).any(), f"Inf in {name} logits"
    
    for i, (K, V) in enumerate(kv_cache):
        assert not torch.isnan(K).any(), f"NaN in layer {i} K cache"
        assert not torch.isnan(V).any(), f"NaN in layer {i} V cache"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])