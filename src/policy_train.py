"""
TRAIN POLICY — Entry point for Hi-Dreamer actor-critic training.

Usage:
    python train_policy.py --config configs/policy.yaml --game Pong-v5
    python train_policy.py --config configs/policy.yaml --offline  # no env needed

Requires pre-trained frozen models:
    - World model checkpoint
    - HRVQ tokenizer checkpoint  
    - CNN encoder checkpoint
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_model import HierarchicalWorldModel, WorldModelConfig
from vq import HRVQTokenizer
from encoder_v1 import AtariCNNEncoder

from policy import (
    ActorNetwork, CriticNetwork, RewardNetwork, ContinueNetwork,
    HierarchicalFeatureExtractor, count_policy_params,
)
from imagination import ImagineRollout
from replay_buffer import TokenReplayBuffer
from actor_critic_train import ActorCriticTrainer


def parse_args():
    """ Parse command-line arguments for policy training."""
    
    parser = argparse.ArgumentParser(description = "Hi-Dreamer Policy Training")
    parser.add_argument("--config", type = str, default = "configs/policy.yaml")
    parser.add_argument("--game", type = str, default = None, help="Override game from config")
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--wandb", action = "store_true", default = False)
    parser.add_argument("--offline", action = "store_true", default = False, help = "Force offline mode")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Resume from checkpoint")
    parser.add_argument("--seed", type = int, default = 42)
    return parser.parse_args()

def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_offline_buffer(config: dict, game: str, device: torch.device) -> TokenReplayBuffer:
    """Load pre-collected data into replay buffer."""
    
    print(f"\nLoading offline buffer for {game}...")
    buffer = TokenReplayBuffer.from_numpy_data(
        tokens_dir=config['data']['tokens_dir'],
        replay_dir=config['data']['replay_dir'],
        game=game,
        capacity=100_000,
        seq_len=64,
        device=device,
    )
    print(f"  Buffer ready: {len(buffer)} transitions")
    return buffer
