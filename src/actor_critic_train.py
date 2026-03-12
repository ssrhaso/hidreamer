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
from torch.distributions import Bernoulli

import wandb

from policy import (
    ActorNetwork,
    CriticMovingAverage,
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
    def __init__(
        self,
        
        # FROZEN MODELS
        world_model           : nn.Module,
        hrvq_tokenizer        : nn.Module,
        encoder               : nn.Module,
        
        # TRAINABLE NETWORKS
        feature_extractor     : HierarchicalFeatureExtractor,
        policy                : ActorNetwork,
        critic                : CriticNetwork,
        reward_network        : RewardNetwork,
        continue_network      : ContinueNetwork,
        
        # INFRASTRUCTURE
        imagination           : ImagineRollout,
        replay_buffer         : TokenReplayBuffer,
        config                : dict,
        device                : torch.device = None,
        env = None,
    ):
        
        self.world_model = world_model
        self.hrvq_tokenizer = hrvq_tokenizer
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.policy = policy
        self.critic = critic
        self.reward_network = reward_network
        self.continue_network = continue_network
        self.imagination = imagination
        self.replay_buffer = replay_buffer
        self.config = config
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        """ SEPERATE OPTIMIZERS FOR ACTOR-CRITIC AND AUXILIARY NETWORKS - DreamerV3 Inspired """
        
        policy_config = config['policy']
        
        self.actor_optimizer = Adam(
            params = policy.parameters(),
            lr = policy_config['actor_lr'],
            eps = 1e-5,
        )
        
        self.critic_optimizer = Adam(
            params = critic.parameters(),
            lr = policy_config['critic_lr'],
            eps = 1e-5,
        )
        
        self.aux_optimizer = Adam(
            params = list(reward_network.parameters()) + list(continue_network.parameters()),
            lr = policy_config['aux_lr'],
            eps = 1e-5,
        )
        
        # Slow Target for Critic (EMA)
        self.slow_target = CriticMovingAverage(
            critic = self.critic, 
            tau = policy_config['critic_slow_target_tau']
        )
        
        # Return Normalizer
        self.return_normalizer = ReturnNormalizer(
            decay = policy_config['return_normalizer_decay'],
        )
        
        # Training State Variables
        self.global_step = 0
        self.episodes_collected = 0
        
        # Continue Loss - Bernoulli Dist
        self.Bernoulli = Bernoulli
        
        
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