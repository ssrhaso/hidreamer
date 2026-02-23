"""
Shared pytest fixtures for the DREAMER test suite.

Path setup:  adds `src/` to sys.path so all tests can import world_model,
             world_model_dataset, encoder_v1, vq, etc. without a package install.
"""

import sys
from pathlib import Path

# Make `src/` importable from any test file
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))          # allows `from src.world_model import …` too

import json

import numpy as np
import pytest
import torch
import yaml

from world_model import HierarchicalWorldModel, WorldModelConfig


""" TINY MODEL FIXTURES """

@pytest.fixture(scope="session")
def tiny_config() -> WorldModelConfig:
    """Small in-memory WorldModelConfig that runs fast on CPU.

    Constraints kept from the real config:
      - d_model divisible by n_heads
      - max_seq_len divisible by 4
      - num_codes / num_actions match production values so token ranges stay valid
    """
    return WorldModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,        # deterministic for tests
        max_seq_len=32,
        num_codes=256,
        num_actions=9,
        layer_weights=[1.0, 0.5, 0.1],
    )


@pytest.fixture                     # function-scoped: gradient tests mutate .grad
def tiny_model(tiny_config) -> HierarchicalWorldModel:
    """Fresh HierarchicalWorldModel on CPU for each test."""
    model = HierarchicalWorldModel(config=tiny_config)
    model.eval()
    return model


@pytest.fixture
def dummy_batch(tiny_config):
    """Random (tokens, actions) tensors on CPU with shape (B=2, T=4, …)."""
    B, T = 2, 4
    tokens  = torch.randint(0, tiny_config.num_codes,   (B, T, 3))
    actions = torch.randint(0, tiny_config.num_actions, (B, T))
    return tokens, actions


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Temporary directory for checkpoint saves / loads."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    return ckpt


""" TINY DATASET FIXTURE  (session-scoped via tmp_path_factory) """

@pytest.fixture(scope="session")
def tiny_dataset_paths(tmp_path_factory):
    """Generate minimal dummy .npy / .npz dataset files and a matching YAML config.

    Files created (all tiny so the test suite stays under 60 s on CPU):
      <tmp>/rsvq_tokens/vq_tokens_ALE_TestGame-v0_layer{0,1,2}.npy   shape (300, 1, 1)
      <tmp>/replay/replay_buffer_ALE_TestGame-v0.npz                  N=300 frames
      <tmp>/worldmodel.yaml

    Returns a dict with:
      config_path  – str path to the YAML
      config       – the raw dict
      game         – game name string
      N            – number of frames
      tokens_dir   – Path to token directory
      replay_dir   – Path to replay directory
    """
    tmp = tmp_path_factory.mktemp("dataset")

    N        = 300
    SEQ_LEN  = 8
    BATCH    = 2
    GAME     = "TestGame-v0"

    tokens_dir = tmp / "rsvq_tokens"
    replay_dir = tmp / "replay"
    tokens_dir.mkdir()
    replay_dir.mkdir()

    # RSVQ token files (3 layers)
    rng = np.random.default_rng(0)
    for layer in range(3):
        arr = rng.integers(0, 256, size=(N, 1, 1), dtype=np.int64)
        np.save(tokens_dir / f"vq_tokens_ALE_{GAME}_layer{layer}.npy", arr)

    # Replay buffer
    actions = rng.integers(0, 9, size=N, dtype=np.int64)
    dones   = np.zeros(N, dtype=bool)
    dones[::50] = True                          # episode boundary every 50 steps
    states  = rng.integers(0, 256, size=(N, 4, 84, 84), dtype=np.uint8)
    np.savez(
        replay_dir / f"replay_buffer_ALE_{GAME}.npz",
        actions=actions,
        dones=dones,
        states=states,
    )

    # Config YAML
    cfg = {
        "model": {
            "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256,
            "dropout": 0.0, "max_seq_len": 32, "num_codes": 256,
            "num_actions": 9, "layer_weights": [1.0, 0.5, 0.1],
        },
        "training": {
            "seed": 42,
            "batch_size": BATCH,
            "seq_len": SEQ_LEN,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "num_epochs": 2,
            "warmup_steps": 2,
            "betas": [0.9, 0.95],
            "mixed_precision": False,
            "accumulation_steps": 1,
        },
        "data": {
            "games": [GAME],
            "tokens_dir": str(tokens_dir),
            "replay_dir": str(replay_dir),
            "val_split": 0.1,
        },
        "logging": {
            "save_dir": str(tmp / "checkpoints"),
            "save_every": 1,
        },
    }

    config_path = tmp / "worldmodel.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    return {
        "config_path": str(config_path),
        "config":      cfg,
        "game":        GAME,
        "N":           N,
        "tokens_dir":  tokens_dir,
        "replay_dir":  replay_dir,
    }


""" TINY RSVQ TOKEN FILES  (for test_shared_tokens) """

@pytest.fixture(scope="session")
def tiny_rsvq_token_paths(tmp_path_factory):
    """Three game token files with a known number of shared tokens at layer 0.

    Shared tokens are guaranteed by construction: codes [0..49] appear in all
    three games, everything else is game-specific.
    """
    tmp = tmp_path_factory.mktemp("rsvq")

    N      = 10_000
    SHARED = list(range(50))        # 50 tokens shared across all games
    GAMES  = ["Pong-v5", "Breakout-v5", "MsPacman-v5"]

    rng = np.random.default_rng(1)
    paths = {}
    for g_idx, game in enumerate(GAMES):
        # Half frames from shared pool, half from game-specific pool
        shared_part    = rng.choice(SHARED, size=N // 2)
        specific_start = 50 + g_idx * 68          # non-overlapping ranges
        specific_part  = rng.integers(specific_start, specific_start + 68, size=N // 2)
        tokens = np.concatenate([shared_part, specific_part]).astype(np.int64)
        rng.shuffle(tokens)
        path = tmp / f"vq_tokens_ALE_{game}_layer0.npy"
        np.save(path, tokens)
        paths[game] = str(path)

    return {"paths": paths, "games": GAMES, "expected_min_shared": 45}
