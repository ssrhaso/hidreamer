"""
Pytest conversion of testing/validate_mask.py
Original logic preserved exactly — wrapped into individual test functions.
"""

import torch
import pytest

from world_model import hierarchical_causal_mask

# shared fixture: compute mask once for the whole module

@pytest.fixture(scope="module")
def mask_8():
    """Hierarchical causal mask for 8 positions (2 timesteps: t0, t1)."""
    return hierarchical_causal_mask(8, torch.device("cpu"))


""" CROSS-TIMESTEP HIERARCHICAL BLOCKING """

def test_l1_t1_cannot_see_l1_t0(mask_8):
    """L1 at t1 must NOT see L1 from t0 (hierarchical block)."""
    assert mask_8[5, 1] == float("-inf"), (
        f"L1_t1 should NOT see L1_t0 (hierarchical block); got {mask_8[5, 1]}"
    )


def test_l1_t1_cannot_see_l2_t0(mask_8):
    """L1 at t1 must NOT see L2 from t0 (hierarchical block)."""
    assert mask_8[5, 2] == float("-inf"), (
        f"L1_t1 should NOT see L2_t0 (hierarchical block); got {mask_8[5, 2]}"
    )


def test_l2_t1_cannot_see_l1_t0(mask_8):
    """L2 at t1 must NOT see L1 from t0 (hierarchical block)."""
    assert mask_8[6, 1] == float("-inf"), (
        f"L2_t1 should NOT see L1_t0 (hierarchical block); got {mask_8[6, 1]}"
    )


def test_l2_t1_cannot_see_l2_t0(mask_8):
    """L2 at t1 must NOT see L2 from t0 (hierarchical block)."""
    assert mask_8[6, 2] == float("-inf"), (
        f"L2_t1 should NOT see L2_t0 (hierarchical block); got {mask_8[6, 2]}"
    )


""" COARSE PAST STILL VISIBLE """

def test_l1_t1_can_see_l0_t0(mask_8):
    """L1 at t1 SHOULD see L0 from t0 (coarse/physics layer from past)."""
    assert mask_8[5, 0] == 0, (
        f"L1_t1 SHOULD see L0_t0 (physics from past); got {mask_8[5, 0]}"
    )


def test_l2_t1_can_see_l0_t0(mask_8):
    """L2 at t1 SHOULD see L0 from t0 (coarse/physics layer from past)."""
    assert mask_8[6, 0] == 0, (
        f"L2_t1 SHOULD see L0_t0 (physics from past); got {mask_8[6, 0]}"
    )


""" WITHIN-TIMESTEP HIERARCHY """

def test_l1_t1_can_see_l0_t1(mask_8):
    """L1 at t1 SHOULD see L0 at t1 (current-timestep physics)."""
    assert mask_8[5, 4] == 0, (
        f"L1_t1 SHOULD see L0_t1 (current physics); got {mask_8[5, 4]}"
    )


def test_l2_t1_can_see_l0_t1(mask_8):
    """L2 at t1 SHOULD see L0 at t1."""
    assert mask_8[6, 4] == 0, (
        f"L2_t1 SHOULD see L0_t1 (current physics); got {mask_8[6, 4]}"
    )


def test_l2_t1_can_see_l1_t1(mask_8):
    """L2 at t1 SHOULD see L1 at t1 (current-timestep mechanics)."""
    assert mask_8[6, 5] == 0, (
        f"L2_t1 SHOULD see L1_t1 (current mechanics); got {mask_8[6, 5]}"
    )
