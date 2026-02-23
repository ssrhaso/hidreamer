"""
Smoke integration test for the full training pipeline.

Covers (all on CPU with tiny config, mocked wandb):
  1. Two forward passes — no NaN, loss > 0
  2. Single gradient step — parameters are updated
  3. Checkpoint save → reload → model outputs identical
  4. Full 2-epoch train() call via world_model_train.train()
     - wandb.init / wandb.log / wandb.finish / wandb.watch all mocked
     - stats dict returned and has expected keys
     - final_metrics.json written to disk
"""

import json
import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from world_model import HierarchicalWorldModel, hierarchical_loss
from torch import GradScaler


""" 1. TWO FORWARD PASSES  —  no NaN, finite positive loss """

def test_smoke_two_forward_passes_no_nan(tiny_model, tiny_config, dummy_batch):
    """Two consecutive forward passes must produce no NaN/Inf, positive finite loss."""
    tokens, actions = dummy_batch

    for pass_idx in range(2):
        logits_l0, logits_l1, logits_l2 = tiny_model(tokens, actions)
        loss, metrics = hierarchical_loss(
            logits_l0=logits_l0,
            logits_l1=logits_l1,
            logits_l2=logits_l2,
            tokens=tokens,
            layer_weights=tiny_config.layer_weights,
        )

        assert not torch.isnan(loss), f"NaN loss on forward pass {pass_idx + 1}"
        assert not torch.isinf(loss), f"Inf loss on forward pass {pass_idx + 1}"
        assert loss.item() > 0,       f"Non-positive loss on forward pass {pass_idx + 1}"


def test_smoke_all_logits_finite(tiny_model, dummy_batch):
    """All output logit tensors must be finite on both forward passes."""
    tokens, actions = dummy_batch

    for _ in range(2):
        l0, l1, l2 = tiny_model(tokens, actions)
        for name, logits in [("l0", l0), ("l1", l1), ("l2", l2)]:
            assert torch.isfinite(logits).all(), f"Non-finite values in {name} logits"


""" 2. SINGLE GRADIENT STEP  —  parameters change """

def test_smoke_gradient_step_updates_parameters(tiny_config, dummy_batch):
    """After one AdamW step the model parameters must have changed."""
    model = HierarchicalWorldModel(config=tiny_config)
    model.train()

    tokens, actions = dummy_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # snapshot initial weights
    params_before = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    # forward + backward + step
    logits_l0, logits_l1, logits_l2 = model(tokens, actions)
    loss, _ = hierarchical_loss(
        logits_l0=logits_l0,
        logits_l1=logits_l1,
        logits_l2=logits_l2,
        tokens=tokens,
        layer_weights=tiny_config.layer_weights,
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    changed = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and not torch.equal(p.data, params_before[name])
    ]
    assert changed, "No model parameters were updated after one gradient step"


""" 3. CHECKPOINT ROUND-TRIP  —  save → reload → identical outputs """

def test_smoke_checkpoint_roundtrip(tiny_config, dummy_batch, tmp_path):
    """Save a checkpoint; reload into a fresh model; forward pass must be bit-identical."""
    tokens, actions = dummy_batch

    # original model
    model_a = HierarchicalWorldModel(config=tiny_config)
    model_a.eval()

    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    scaler    = GradScaler(enabled=False)

    with torch.no_grad():
        logits_before = model_a(tokens, actions)

    ckpt_path = tmp_path / "smoke_test.pt"
    torch.save(
        {
            "model_state_dict":     model_a.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "epoch":                0,
            "global_step":          1,
            "best_val_loss":        9.999,
        },
        ckpt_path,
    )

    # reload into fresh model
    model_b = HierarchicalWorldModel(config=tiny_config)
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_b.load_state_dict(ckpt["model_state_dict"])
    model_b.eval()

    assert ckpt["epoch"]       == 0
    assert ckpt["global_step"] == 1

    with torch.no_grad():
        logits_after = model_b(tokens, actions)

    for name, before, after in zip(
        ["l0", "l1", "l2"], logits_before, logits_after
    ):
        assert torch.allclose(before, after, atol=1e-6), (
            f"Checkpoint round-trip failed: {name} logits differ after reload"
        )


""" 4. FULL TRAIN() LOOP  —  mocked wandb, 2 epochs, stats dict valid """

@patch("wandb.init",   return_value=MagicMock())
@patch("wandb.log",    return_value=None)
@patch("wandb.finish", return_value=None)
@patch("wandb.watch",  return_value=None)
def test_smoke_full_train_loop(
    mock_watch, mock_finish, mock_log, mock_init,
    tiny_dataset_paths,
):
    """Run the 2-epoch train() from world_model_train.py end-to-end on CPU.

    Assertions:
      - wandb.init called once
      - wandb.finish called once
      - Returned stats dict has expected keys
      - final_metrics.json written to save_dir
      - final_val_loss is a finite float
    """
    from world_model_train import train

    model, stats = train(
        config_path=tiny_dataset_paths["config_path"],
        resume_from=None,
        use_wandb=True,         # wandb calls are mocked above
    )

    # wandb integration
    mock_init.assert_called_once()
    mock_finish.assert_called_once()

    # stats dict
    assert stats is not None, "train() returned None stats"
    for key in (
        "final_train_loss", "final_val_loss",
        "final_train_accuracy_l0", "final_val_accuracy_l0",
        "final_train_accuracy_l1", "final_val_accuracy_l1",
        "final_train_accuracy_l2", "final_val_accuracy_l2",
    ):
        assert key in stats, f"Missing key in stats: {key}"

    val_loss = stats["final_val_loss"]
    assert isinstance(val_loss, float), f"final_val_loss is not float: {type(val_loss)}"
    assert math.isfinite(val_loss),     f"final_val_loss is not finite: {val_loss}"

    # JSON on disk
    save_dir = tiny_dataset_paths["config"]["logging"]["save_dir"]
    json_path = Path(save_dir) / "final_metrics.json"
    assert json_path.exists(), f"final_metrics.json not written to {save_dir}"

    with open(json_path) as f:
        on_disk = json.load(f)
    assert "final_val_loss" in on_disk, "final_val_loss missing from saved JSON"

    # best_model.pt must exist (checked here to avoid false-positive from a
    # separate test running on the same session-scoped tmp dir)
    best_path = Path(save_dir) / "best_model.pt"
    assert best_path.exists(), f"best_model.pt not found in {save_dir}"


# NOTE: best_model.pt existence is asserted inside test_smoke_full_train_loop
# to avoid a false-positive from the session-scoped tmp dir already containing
# that file by the time a standalone test runs.
