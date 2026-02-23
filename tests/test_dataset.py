"""
Pytest conversion of testing/validate_dataset.py
Original logic preserved exactly — wrapped into individual test functions.

Real .npy / .npz files are NOT required: the tiny_dataset_paths fixture in
conftest.py generates all necessary dummy files in a tmp directory.
"""

import numpy as np
import pytest
import torch

from world_model import HierarchicalWorldModel, WorldModelConfig, hierarchical_loss
from world_model_dataset import WorldModelDataset, create_dataloaders


# module-scoped loaders so we only build the dataset once per test file

@pytest.fixture(scope="module")
def loaders(tiny_dataset_paths):
    train_loader, val_loader, info = create_dataloaders(
        config_path=tiny_dataset_paths["config_path"],
        seed=42,
    )
    return train_loader, val_loader, info


@pytest.fixture(scope="module")
def dataset(loaders):
    train_loader, _, _ = loaders
    return train_loader.dataset.dataset   # unwrap RandomSubset → WorldModelDataset


@pytest.fixture(scope="module")
def first_batch(loaders):
    train_loader, _, _ = loaders
    return next(iter(train_loader))


""" TEST A  —  Batch shapes and dtypes """

def test_dataset_token_shape(first_batch, loaders):
    """tokens batch must have shape (batch_size, seq_len, 3)."""
    tokens, _ = first_batch
    _, _, info = loaders
    expected = (info["batch_size"], info["seq_len"], 3)
    assert tokens.shape == expected, (
        f"FAIL tokens shape: {tokens.shape}, expected {expected}"
    )


def test_dataset_action_shape(first_batch, loaders):
    """actions batch must have shape (batch_size, seq_len)."""
    _, actions = first_batch
    _, _, info = loaders
    expected = (info["batch_size"], info["seq_len"])
    assert actions.shape == expected, (
        f"FAIL actions shape: {actions.shape}, expected {expected}"
    )


def test_dataset_token_dtype(first_batch):
    """tokens must be int64."""
    tokens, _ = first_batch
    assert tokens.dtype == torch.int64, f"FAIL tokens dtype: {tokens.dtype}"


def test_dataset_action_dtype(first_batch):
    """actions must be int64."""
    _, actions = first_batch
    assert actions.dtype == torch.int64, f"FAIL actions dtype: {actions.dtype}"


def test_dataset_token_range(first_batch):
    """Token indices must lie in [0, 255]."""
    tokens, _ = first_batch
    assert tokens.min() >= 0 and tokens.max() <= 255, (
        f"FAIL token range: [{tokens.min()}, {tokens.max()}]"
    )


def test_dataset_action_range(first_batch):
    """Action indices must lie in [0, 8]."""
    _, actions = first_batch
    assert actions.min() >= 0 and actions.max() <= 8, (
        f"FAIL action range: [{actions.min()}, {actions.max()}]"
    )


""" TEST B  —  Episode boundary integrity """

def test_dataset_no_episode_boundary_violations(dataset, loaders, tiny_dataset_paths):
    """No sampled window may cross an episode boundary (done=True)."""
    _, _, info = loaders

    # Load ground-truth dones for each game
    game    = tiny_dataset_paths["game"]
    buf     = np.load(
        str(tiny_dataset_paths["replay_dir"] / f"replay_buffer_ALE_{game}.npz")
    )
    dones_per_game = [buf["dones"].astype(bool)]    # only 1 game in fixture

    num_checked = min(500, len(dataset.valid_starts))
    violations  = 0

    for game_idx, start in dataset.valid_starts[:num_checked]:
        window_dones = dones_per_game[game_idx][start : start + info["seq_len"] - 1]
        if window_dones.any():
            violations += 1

    assert violations == 0, (
        f"FAIL: {violations} boundary violations across {num_checked} samples"
    )


""" TEST C  —  Multi-game mixing (all configured games present) """

def test_dataset_all_games_present(dataset, tiny_dataset_paths):
    """Every game in the config must contribute at least one valid window."""
    game_counts: dict = {}
    for game_idx, _ in dataset.valid_starts:
        game_counts[game_idx] = game_counts.get(game_idx, 0) + 1

    num_games = len(tiny_dataset_paths["config"]["data"]["games"])

    assert len(game_counts) == num_games, (
        f"FAIL: {len(game_counts)} games found, expected {num_games}"
    )
    for gidx in range(num_games):
        assert game_counts.get(gidx, 0) > 0, (
            f"FAIL: game index {gidx} has 0 valid windows"
        )


""" TEST D  —  Forward pass + loss integration """

def test_dataset_forward_pass_and_loss(first_batch, tiny_config):
    """A batch from the real DataLoader must pass through the model without NaN."""
    tokens, actions = first_batch
    model = HierarchicalWorldModel(config=tiny_config)
    model.eval()

    logits_l0, logits_l1, logits_l2 = model(tokens, actions)

    loss, metrics = hierarchical_loss(
        logits_l0=logits_l0,
        logits_l1=logits_l1,
        logits_l2=logits_l2,
        tokens=tokens,
        layer_weights=tiny_config.layer_weights,
    )

    assert not torch.isnan(loss), "FAIL: loss is NaN after dataset forward pass"
    assert not torch.isinf(loss), "FAIL: loss is Inf after dataset forward pass"


def test_dataset_gradients_flow_after_real_batch(first_batch, tiny_config):
    """Gradients must reach embeddings and heads when using a real DataLoader batch."""
    tokens, actions = first_batch
    model = HierarchicalWorldModel(config=tiny_config)
    model.train()
    model.zero_grad()

    logits_l0, logits_l1, logits_l2 = model(tokens, actions)
    loss, _ = hierarchical_loss(
        logits_l0=logits_l0,
        logits_l1=logits_l1,
        logits_l2=logits_l2,
        tokens=tokens,
        layer_weights=tiny_config.layer_weights,
    )
    loss.backward()

    assert model.embedding.token_embeds[0].weight.grad is not None, (
        "FAIL: no gradient reaching token embeddings"
    )
    assert model.headl0.weight.grad is not None, (
        "FAIL: no gradient reaching output heads"
    )
