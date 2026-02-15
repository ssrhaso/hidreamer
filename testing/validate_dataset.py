""" WORLD MODEL DATASET VALIDATION """

import sys
sys.path.append('.')

import numpy as np
import torch
from src.world_model_dataset import WorldModelDataset, create_dataloaders
from src.world_model import WorldModelConfig, HierarchicalWorldModel, hierarchical_loss


if __name__ == "__main__":
    # LOAD DATASET
    train_loader, val_loader, info = create_dataloaders()
    dataset = train_loader.dataset.dataset  # unwrap RandomSubset

    print(f"\nDataset info: {info}")
    print(f"Games loaded: {len(dataset.all_tokens)}")
    for i, (tok, act) in enumerate(zip(dataset.all_tokens, dataset.all_actions)):
        print(f"  Game {i}: tokens {tok.shape}, actions {act.shape}")


    # TEST A: Batch shapes and dtypes
    tokens, actions = next(iter(train_loader))

    assert tokens.shape == (info['batch_size'], info['seq_len'], 3), \
        f"FAIL tokens shape: {tokens.shape}, expected ({info['batch_size']}, {info['seq_len']}, 3)"
    assert actions.shape == (info['batch_size'], info['seq_len']), \
        f"FAIL actions shape: {actions.shape}, expected ({info['batch_size']}, {info['seq_len']})"
    print(f" Shapes OK: tokens {tokens.shape}, actions {actions.shape}")

    assert tokens.dtype == torch.int64, f"FAIL tokens dtype: {tokens.dtype}"
    assert actions.dtype == torch.int64, f"FAIL actions dtype: {actions.dtype}"
    print(f" Dtypes OK: tokens {tokens.dtype}, actions {actions.dtype}")

    assert tokens.min() >= 0 and tokens.max() <= 255, \
        f"FAIL token range: [{tokens.min()}, {tokens.max()}]"
    assert actions.min() >= 0 and actions.max() <= 8, \
        f"FAIL action range: [{actions.min()}, {actions.max()}]"
    print(f" Ranges OK: tokens [{tokens.min()}, {tokens.max()}], actions [{actions.min()}, {actions.max()}]")


    # TEST B: Episode boundary integrity
    games = ['Pong-v5', 'Breakout-v5', 'MsPacman-v5']
    dones_per_game = []
    for game in games:
        buf = np.load(f"data/replay_buffer_ALE_{game}.npz")
        dones_per_game.append(buf['dones'].astype(bool))

    num_checked = min(5000, len(dataset.valid_starts))
    violations = 0
    for game_idx, start in dataset.valid_starts[:num_checked]:
        window_dones = dones_per_game[game_idx][start : start + info['seq_len'] - 1]
        if window_dones.any():
            violations += 1

    assert violations == 0, f"FAIL: {violations} boundary violations in {num_checked} samples!"
    print(f" Boundary check: {num_checked} windows, 0 violations")


    # TEST C: Multi-game mixing
    game_counts = {}
    for game_idx, _ in dataset.valid_starts:
        game_counts[game_idx] = game_counts.get(game_idx, 0) + 1

    for gidx, count in sorted(game_counts.items()):
        pct = 100.0 * count / len(dataset.valid_starts)
        print(f" Game {gidx} ({games[gidx]}): {count} windows ({pct:.1f}%)")

    assert len(game_counts) == 3, f"FAIL: only {len(game_counts)} games found, expected 3"
    for gidx in range(3):
        assert game_counts.get(gidx, 0) > 0, f"FAIL: game {gidx} has 0 valid windows"
    print(f" All 3 games present")


    # TEST D: Forward pass + loss integration
    config = WorldModelConfig()
    model = HierarchicalWorldModel(config)

    logits_l0, logits_l1, logits_l2 = model(tokens, actions)
    print(f" Logits shapes: L0={logits_l0.shape}, L1={logits_l1.shape}, L2={logits_l2.shape}")

    loss, metrics = hierarchical_loss(logits_l0, logits_l1, logits_l2, tokens)
    assert not torch.isnan(loss), "FAIL: loss is NaN"
    assert not torch.isinf(loss), "FAIL: loss is Inf"
    print(f" Loss: {loss.item():.4f}")
    print(f" Acc L0: {metrics['accuracy_l0']:.4f}, L1: {metrics['accuracy_l1']:.4f}, L2: {metrics['accuracy_l2']:.4f}")

    loss.backward()
    assert model.embedding.token_embeds[0].weight.grad is not None, "FAIL: no grad to embeddings"
    assert model.headl0.weight.grad is not None, "FAIL: no grad to output heads"
    print(f" Gradients flow: OK")

    print("\nALL DATASET TESTS PASSED")
