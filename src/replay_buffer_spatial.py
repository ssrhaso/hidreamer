"""
REPLAY BUFFER FOR SPATIAL WORLD MODEL POLICY TRAINING

Stores 37-token spatial transitions:
  L0: (capacity, 4)   coarse tokens  (codebook 16)
  L1: (capacity, 16)  mid tokens     (codebook 64)
  L2: (capacity, 16)  fine tokens    (codebook 64)

sample()             → tokens_l0/l1/l2, actions, rewards, dones
sample_seed_context() → tokens_l0/l1/l2, actions  (no rewards/dones)
sample_reward_biased_seed() → same, oversampled near non-zero reward
"""

import numpy as np
import torch
from typing import Dict


class SpatialTokenReplayBuffer:
    """
    Fixed-capacity ring buffer for 37-token spatial transitions.
    Pre-allocated tensors on CPU; returned tensors moved to device on demand.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        seq_len:  int = 16,
        device:   torch.device = None,
    ):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.device   = device or torch.device('cpu')

        self._tokens_l0 = torch.zeros(capacity, 4,  dtype=torch.long)
        self._tokens_l1 = torch.zeros(capacity, 16, dtype=torch.long)
        self._tokens_l2 = torch.zeros(capacity, 16, dtype=torch.long)
        self._actions   = torch.zeros(capacity,     dtype=torch.long)
        self._rewards   = torch.zeros(capacity,     dtype=torch.float32)
        self._dones     = torch.zeros(capacity,     dtype=torch.bool)

        self._ptr  = 0
        self._size = 0

    def push(
        self,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
        action:    int,
        reward:    float,
        done:      bool,
    ):
        """Store single transition."""
        self._tokens_l0[self._ptr] = tokens_l0
        self._tokens_l1[self._ptr] = tokens_l1
        self._tokens_l2[self._ptr] = tokens_l2
        self._actions[self._ptr]   = action
        self._rewards[self._ptr]   = reward
        self._dones[self._ptr]     = done

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _build_indices(self, starts: torch.Tensor, length: int) -> torch.Tensor:
        """Vectorised index matrix: (B, length)."""
        offsets = torch.arange(length, dtype=torch.long)
        return starts.unsqueeze(1) + offsets.unsqueeze(0)

    def _gather(self, starts: torch.Tensor, length: int) -> Dict[str, torch.Tensor]:
        """Gather sequences for all fields."""
        idx = self._build_indices(starts, length)
        dev = self.device
        return {
            'tokens_l0': self._tokens_l0[idx].to(dev),
            'tokens_l1': self._tokens_l1[idx].to(dev),
            'tokens_l2': self._tokens_l2[idx].to(dev),
            'actions':   self._actions[idx].to(dev),
            'rewards':   self._rewards[idx].to(dev),
            'dones':     self._dones[idx].to(dev),
        }

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Random sequences of length seq_len for reward/continue network training."""
        assert self._size >= self.seq_len, "Buffer too small for sampling"
        max_start = self._size - self.seq_len
        starts = torch.randint(0, max(max_start, 1), (batch_size,))
        return self._gather(starts, self.seq_len)

    def sample_seed_context(
        self,
        batch_size:  int,
        context_len: int = 16,
    ) -> Dict[str, torch.Tensor]:
        """Short context windows (tokens + actions only) for seeding imagination."""
        max_start = max(1, self._size - context_len)
        starts    = torch.randint(0, max_start, (batch_size,))
        idx       = self._build_indices(starts, context_len)
        dev       = self.device
        return {
            'tokens_l0': self._tokens_l0[idx].to(dev),
            'tokens_l1': self._tokens_l1[idx].to(dev),
            'tokens_l2': self._tokens_l2[idx].to(dev),
            'actions':   self._actions[idx].to(dev),
        }

    def sample_reward_biased_seed(
        self,
        batch_size:       int,
        context_len:      int   = 16,
        nonzero_fraction: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Half of seeds from near-reward events, rest random."""
        nonzero_idx = torch.where(self._rewards[:self._size].abs() > 1e-4)[0]

        nonzero_n = int(batch_size * nonzero_fraction) if len(nonzero_idx) > 0 else 0
        random_n  = batch_size - nonzero_n

        parts = []

        if nonzero_n > 0:
            chosen = nonzero_idx[torch.randint(len(nonzero_idx), (nonzero_n,))]
            offset = max(1, int(context_len * 0.75))
            starts = (chosen.long() - offset).clamp(0, self._size - context_len)
            parts.append(starts)

        if random_n > 0:
            max_start = max(1, self._size - context_len)
            parts.append(torch.randint(0, max_start, (random_n,)))

        starts = torch.cat(parts, dim=0)[:batch_size]
        idx    = self._build_indices(starts, context_len)
        dev    = self.device
        return {
            'tokens_l0': self._tokens_l0[idx].to(dev),
            'tokens_l1': self._tokens_l1[idx].to(dev),
            'tokens_l2': self._tokens_l2[idx].to(dev),
            'actions':   self._actions[idx].to(dev),
        }

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int = 1000) -> bool:
        return self._size >= min_size

    @classmethod
    def from_numpy_data(
        cls,
        tokens_dir: str,
        game:       str,
        capacity:   int          = 100_000,
        seq_len:    int          = 16,
        device:     torch.device = None,
    ) -> 'SpatialTokenReplayBuffer':
        """Load pre-extracted spatial token files for one game."""
        base = f"{tokens_dir}/{game}"

        l0      = np.load(f"{base}/spatial_tokens_l0.npy").astype(np.int64)
        l1      = np.load(f"{base}/spatial_tokens_l1.npy").astype(np.int64)
        l2      = np.load(f"{base}/spatial_tokens_l2.npy").astype(np.int64)
        actions = np.load(f"{base}/actions.npy").astype(np.int64)
        rewards = np.load(f"{base}/rewards.npy").astype(np.float32)
        dones   = np.load(f"{base}/dones.npy").astype(bool)

        N = min(len(actions), capacity)

        buf = cls(capacity=capacity, seq_len=seq_len, device=device)
        buf._tokens_l0[:N] = torch.from_numpy(l0[:N])
        buf._tokens_l1[:N] = torch.from_numpy(l1[:N])
        buf._tokens_l2[:N] = torch.from_numpy(l2[:N])
        buf._actions[:N]   = torch.from_numpy(actions[:N])
        buf._rewards[:N]   = torch.from_numpy(rewards[:N])
        buf._dones[:N]     = torch.from_numpy(dones[:N])
        buf._size          = N
        buf._ptr           = N % capacity

        print(f"    Loaded {N} spatial transitions for {game}")
        return buf
