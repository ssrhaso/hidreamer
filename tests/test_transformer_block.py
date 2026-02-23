"""
Pytest conversion of testing/validate_transformer_block.py
Original logic preserved exactly — wrapped into individual test functions.
"""

import torch
import pytest

from world_model import (
    WorldModelConfig,
    TokenEmbedding,
    TransformerBlock,
    hierarchical_causal_mask,
)


# module-level helpers

@pytest.fixture(scope="module")
def block_inputs(tiny_config):
    """Embed a random batch and return (seq, mask, block) ready for forward."""
    B, T = 2, 4
    tokens  = torch.randint(0, tiny_config.num_codes,   (B, T, 3))
    actions = torch.randint(0, tiny_config.num_actions, (B, T))

    embed = TokenEmbedding(tiny_config)
    seq   = embed(tokens, actions)                            # (B, T*4, d_model)
    mask  = hierarchical_causal_mask(T * 4, torch.device("cpu"))

    block = TransformerBlock(tiny_config)
    return seq, mask, block


""" TEST A  —  Shape preserved """

def test_transformer_block_shape_preserved(block_inputs):
    """Output shape must equal input shape."""
    seq, mask, block = block_inputs
    out = block(seq, mask)
    assert out.shape == seq.shape, f"FAIL: shape {out.shape} != {seq.shape}"


""" TEST B  —  Block actually transforms the input """

def test_transformer_block_output_differs_from_input(block_inputs):
    """Output must differ from input — block must do work."""
    seq, mask, block = block_inputs
    out = block(seq, mask)
    assert not torch.allclose(out, seq, atol=1e-6), (
        "FAIL: TransformerBlock output is identical to input — block did nothing"
    )


""" TEST C  —  No NaN / Inf """

def test_transformer_block_no_nan(block_inputs):
    """No NaN values in output."""
    seq, mask, block = block_inputs
    out = block(seq, mask)
    assert not torch.isnan(out).any(), "FAIL: NaN in TransformerBlock output"


def test_transformer_block_no_inf(block_inputs):
    """No Inf values in output."""
    seq, mask, block = block_inputs
    out = block(seq, mask)
    assert not torch.isinf(out).any(), "FAIL: Inf in TransformerBlock output"


""" TEST D  —  Parameter count (smoke check) """

def test_transformer_block_has_parameters(tiny_config):
    """Block must have a positive number of trainable parameters."""
    block = TransformerBlock(tiny_config)
    num_params = sum(p.numel() for p in block.parameters())
    assert num_params > 0, "FAIL: TransformerBlock has no parameters"


""" TEST E  —  Gradients flow """

def test_transformer_block_gradients_flow(tiny_config):
    """All parameters must receive gradients after a backward pass."""
    B, T = 2, 4
    tokens  = torch.randint(0, tiny_config.num_codes,   (B, T, 3))
    actions = torch.randint(0, tiny_config.num_actions, (B, T))

    embed = TokenEmbedding(tiny_config)
    seq   = embed(tokens, actions)
    mask  = hierarchical_causal_mask(T * 4, torch.device("cpu"))
    block = TransformerBlock(tiny_config)

    out  = block(seq, mask)
    loss = out.sum()
    loss.backward()

    no_grad = [
        name for name, p in block.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"FAIL: parameters missing gradients: {no_grad}"
