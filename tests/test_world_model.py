"""
Pytest conversion of testing/validate_world_model.py
Original logic preserved exactly — wrapped into individual test functions.
Uses tiny_model / tiny_config / dummy_batch fixtures from conftest.py.
"""

import torch
import pytest

from world_model import hierarchical_loss


""" TEST A  —  Output shapes """

def test_world_model_output_shapes(tiny_model, tiny_config, dummy_batch):
    """All three logit tensors must have shape (B, T, num_codes)."""
    tokens, actions = dummy_batch
    B, T, _ = tokens.shape
    expected = (B, T, tiny_config.num_codes)

    logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)

    assert logits_l0.shape == expected, f"FAIL L0: {logits_l0.shape} != {expected}"
    assert logits_l1.shape == expected, f"FAIL L1: {logits_l1.shape} != {expected}"
    assert logits_l2.shape == expected, f"FAIL L2: {logits_l2.shape} != {expected}"


""" TEST B  —  No NaN / Inf in outputs """

def test_world_model_no_nan_in_logits(tiny_model, dummy_batch):
    """Logit tensors must contain no NaN values."""
    tokens, actions = dummy_batch
    logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)

    assert not torch.isnan(logits_l0).any(), "FAIL: NaN in L0 logits"
    assert not torch.isnan(logits_l1).any(), "FAIL: NaN in L1 logits"
    assert not torch.isnan(logits_l2).any(), "FAIL: NaN in L2 logits"


def test_world_model_no_inf_in_logits(tiny_model, dummy_batch):
    """Logit tensors must contain no Inf values."""
    tokens, actions = dummy_batch
    logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)

    assert not torch.isinf(logits_l0).any(), "FAIL: Inf in L0 logits"
    assert not torch.isinf(logits_l1).any(), "FAIL: Inf in L1 logits"
    assert not torch.isinf(logits_l2).any(), "FAIL: Inf in L2 logits"


""" TEST C  —  Gradients flow end-to-end """

def test_world_model_gradients_flow(tiny_model, dummy_batch):
    """Gradients must reach: embeddings → transformer blocks → output heads."""
    tokens, actions = dummy_batch

    tiny_model.train()
    tiny_model.zero_grad()
    logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)

    dummy_loss = logits_l0.sum() + logits_l1.sum() + logits_l2.sum()
    dummy_loss.backward()

    assert tiny_model.embedding.token_embeds[0].weight.grad is not None, (
        "FAIL: no gradient reaching token embeddings"
    )
    assert list(tiny_model.blocks[0].parameters())[0].grad is not None, (
        "FAIL: no gradient reaching transformer blocks"
    )
    assert tiny_model.headl0.weight.grad is not None, (
        "FAIL: no gradient reaching output head l0"
    )
    assert tiny_model.headl1.weight.grad is not None, (
        "FAIL: no gradient reaching output head l1"
    )
    assert tiny_model.headl2.weight.grad is not None, (
        "FAIL: no gradient reaching output head l2"
    )


""" TEST D  —  Parameter count sanity """

def test_world_model_has_parameters(tiny_model):
    """Model must have a positive number of trainable parameters."""
    total = sum(p.numel() for p in tiny_model.parameters())
    assert total > 0, "FAIL: model has no parameters"


""" BONUS  —  Loss + accuracy metrics are finite """

def test_world_model_loss_and_metrics_finite(tiny_model, tiny_config, dummy_batch):
    """hierarchical_loss must return a finite scalar and valid accuracy keys."""
    tokens, actions = dummy_batch
    logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)

    loss, metrics = hierarchical_loss(
        logits_l0=logits_l0,
        logits_l1=logits_l1,
        logits_l2=logits_l2,
        tokens=tokens,
        layer_weights=tiny_config.layer_weights,
    )

    assert not torch.isnan(loss),  "FAIL: loss is NaN"
    assert not torch.isinf(loss),  "FAIL: loss is Inf"
    assert loss.item() > 0,        "FAIL: loss is non-positive (unexpected for random model)"

    for key in ("accuracy_l0", "accuracy_l1", "accuracy_l2"):
        assert key in metrics,                       f"FAIL: missing metric key '{key}'"
        assert 0.0 <= metrics[key] <= 1.0,           f"FAIL: {key}={metrics[key]} out of [0,1]"
