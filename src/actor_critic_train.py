""" 
ACTOR-CRITIC TRAINER FOR DREAMER AGENT

Alertnates between:
    1. Collecting real env transitions -> storing in replay buffer
    2. Training RewardNetwork and ContinueNetwork on real data from replay buffer
    3. Training ActorNetwork and CriticNetwork on imagined data from world model
    
Inspiration from:
- DreamerV3 (Hafner et al.)
- TWISTER (Burchi & Timofte, ICLR 2025)
"""

import os
import time
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import wandb

from policy import (
    ActorNetwork,
    CriticNetwork,
    RewardNetwork,
    ContinueNetwork,
    
    HierarchicalFeatureExtractor,
    compute_lambda_returns,
    ReturnNormalizer,
    symlog,
    symexp,
    count_policy_params,
    get_horizon,
)

from imagination import ImagineRollout, Trajectory
from replay_buffer import TokenReplayBuffer


class ActorCriticTrainer:
    """ 
    Full Training Loop for POLICY LEARNING
    """
    def __init__():
        pass
    
    def _train_aux():
        pass
    
    def _train_actor_critic():
        pass
    
    @torch.no_grad()
    def _encode_observation():
        pass
    
    def collect_real_episode():
        pass
    
    @torch.no_grad()
    def evaluate():
        pass
    
    """  MAIN TRAINING LOOP """
    def train():
        pass
    
    """ SAVE AND LOGGING """
    def _save_checkpoint():
        pass