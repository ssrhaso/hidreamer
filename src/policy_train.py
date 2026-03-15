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

