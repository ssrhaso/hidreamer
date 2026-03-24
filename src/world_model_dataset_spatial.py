"""
SPATIAL WORLD MODEL DATASET

Loads pre-extracted spatial token sequences produced by extract_spatial_tokens.py.

Expected file layout per game:
    {tokens_dir}/{game}/spatial_tokens_l0.npy  shape (N, 4)   int16
    {tokens_dir}/{game}/spatial_tokens_l1.npy  shape (N, 16)  int16
    {tokens_dir}/{game}/spatial_tokens_l2.npy  shape (N, 16)  int16
    {tokens_dir}/{game}/actions.npy            shape (N,)     int16
    {tokens_dir}/{game}/dones.npy              shape (N,)     bool

__getitem__ returns (tokens_l0, tokens_l1, tokens_l2, actions) for a window
of length seq_len that does not cross any episode boundary (done=True event).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import yaml


class SpatialWorldModelDataset(Dataset):
    def __init__(
        self,
        games: List[str],
        tokens_dir: str = "checkpoints/spatial_tokens",
        seq_len: int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.all_l0: List[np.ndarray] = []
        self.all_l1: List[np.ndarray] = []
        self.all_l2: List[np.ndarray] = []
        self.all_actions: List[np.ndarray] = []
        self.valid_starts: List[Tuple[int, int]] = []  # (game_idx, start_idx)

        for game_idx, game in enumerate(games):
            print(f"LOADING GAME {game}...")
            base = f"{tokens_dir}/{game}"

            l0      = np.load(f"{base}/spatial_tokens_l0.npy").astype(np.int64)  # (N, 4)
            l1      = np.load(f"{base}/spatial_tokens_l1.npy").astype(np.int64)  # (N, 16)
            l2      = np.load(f"{base}/spatial_tokens_l2.npy").astype(np.int64)  # (N, 16)
            actions = np.load(f"{base}/actions.npy").astype(np.int64)             # (N,)
            dones   = np.load(f"{base}/dones.npy").astype(bool)                   # (N,)

            N = len(actions)
            assert l0.shape == (N, 4),  f"{game}: l0 shape {l0.shape}, expected ({N}, 4)"
            assert l1.shape == (N, 16), f"{game}: l1 shape {l1.shape}, expected ({N}, 16)"
            assert l2.shape == (N, 16), f"{game}: l2 shape {l2.shape}, expected ({N}, 16)"

            self.all_l0.append(l0)
            self.all_l1.append(l1)
            self.all_l2.append(l2)
            self.all_actions.append(actions)

            # Valid windows must not cross episode boundaries.
            # boundaries[s] = number of done events in positions [s, s+seq_len-2].
            # This checks that no episode ends WITHIN the window (last position allowed).
            cumsum = np.concatenate([[0], np.cumsum(dones.astype(int))])  # (N+1,)
            num_candidates = N - seq_len + 1
            if num_candidates <= 0:
                print(f"  WARNING: {game} has {N} frames but seq_len={seq_len} — skipping")
                continue

            boundaries = cumsum[seq_len - 1:N] - cumsum[:num_candidates]
            valid_idx = np.where(boundaries == 0)[0]

            for s in valid_idx:
                self.valid_starts.append((game_idx, int(s)))

            print(f"  {N} TIMESTEPS, {len(valid_idx)} VALID STARTS")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        game_idx, start = self.valid_starts[idx]
        end = start + self.seq_len

        l0      = torch.from_numpy(self.all_l0[game_idx][start:end])      # (seq_len, 4)
        l1      = torch.from_numpy(self.all_l1[game_idx][start:end])      # (seq_len, 16)
        l2      = torch.from_numpy(self.all_l2[game_idx][start:end])      # (seq_len, 16)
        actions = torch.from_numpy(self.all_actions[game_idx][start:end])  # (seq_len,)

        return l0, l1, l2, actions


def create_spatial_dataloaders(
    config_path: str = "configs/worldmodel_spatial.yaml",
    seed: int = 42,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = SpatialWorldModelDataset(
        games=config['data']['games'],
        tokens_dir=config['data']['tokens_dir'],
        seq_len=config['training']['seq_len'],
    )

    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    info = {
        'total': len(dataset),
        'train': train_size,
        'val': val_size,
        'seq_len': config['training']['seq_len'],
        'batch_size': config['training']['batch_size'],
    }

    return train_loader, val_loader, info


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    config_path = "configs/worldmodel_spatial.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Creating spatial dataloaders...")
    train_loader, val_loader, info = create_spatial_dataloaders(
        config_path=config_path,
        seed=config['training']['seed'],
    )

    print(f"\nDataset info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nSampling one batch from train loader...")
    batch = next(iter(train_loader))
    tokens_l0, tokens_l1, tokens_l2, actions = batch

    B = tokens_l0.size(0)
    T = tokens_l0.size(1)

    print(f"\nBatch shapes:")
    print(f"  tokens_l0 : {list(tokens_l0.shape)}   expected [{B}, {T}, 4]")
    print(f"  tokens_l1 : {list(tokens_l1.shape)}   expected [{B}, {T}, 16]")
    print(f"  tokens_l2 : {list(tokens_l2.shape)}   expected [{B}, {T}, 16]")
    print(f"  actions   : {list(actions.shape)}   expected [{B}, {T}]")

    assert tokens_l0.shape == (B, T, 4),  f"FAIL tokens_l0: {tokens_l0.shape}"
    assert tokens_l1.shape == (B, T, 16), f"FAIL tokens_l1: {tokens_l1.shape}"
    assert tokens_l2.shape == (B, T, 16), f"FAIL tokens_l2: {tokens_l2.shape}"
    assert actions.shape   == (B, T),     f"FAIL actions:   {actions.shape}"

    print("\nSpatialWorldModelDataset: PASSED")
