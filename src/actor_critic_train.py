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
        
        
    def _train_aux(
        self,
        num_batches : int = 1
    ) -> dict:
        """
        Train RewardNetwork and ContinueNetwork on REAL DATA 
        (Replay buffer)
        
        Supervised labels, used during imagination to provide reward signal """
        
        # INIT
        self.reward_network.train()
        self.continue_network.train()
        total_reward_loss = 0.0
        total_continue_loss = 0.0
        
        # BATCH TRAINING LOOP
        for _ in range(num_batches):
            
            # SAMPLE BATCH FROM REPLAY BUFFER
            batch = self.replay_buffer.sample(
                batch_size = self.config['policy']['aux_batch_size'],
            )
            
            tokens = batch['tokens']         # (B, L, 3) (32, 64, 3)
            rewards = batch['rewards']       # (B, L)    (32, 64)
            dones = batch['dones']           # (B, L)    (32, 64)
            
            # FEATURE EXTRACTION (for each timestep)
            B, L, _ = tokens.shape
            tokens_flat = tokens.reshape(B * L, 3)           # (B*L, 3)           (2048, 3)
            features = self.feature_extractor(tokens_flat)   # (B*L, feature_dim) (2048, 512)
            
            # CONTINUE NETWORK LOSS (Bernoulli) - Binary Classification 
            continue_logits = self.continue_network(features)       # (B*L, 1) , raw logits
            continue_target = (~dones).float().reshape(-1, 1)       # (B*L, 1) , 1 if not done, 0 if done
            continue_loss = -self.Bernoulli(logits = continue_logits).log_prob(continue_target).mean()
            continue_loss = continue_loss.mean()
            
            # REWARD NETWORK LOSS (MSE) - SymLog of rewards (range handling)
            reward_prediction = self.reward_network(features) 
            reward_target = symlog(rewards.reshape(-1))
            reward_loss = F.mse_loss(input = reward_prediction, target = reward_target)
            
            # BACKPROP AND OPTIMIZE
            aux_loss = reward_loss + continue_loss
            self.aux_optimizer.zero_grad()
            aux_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.reward_network.parameters()) + list(self.continue_network.parameters()),
                max_norm = self.config['policy']['aux_max_norm'],
            )
            self.aux_optimizer.step()
            total_reward_loss += reward_loss.item()
            total_continue_loss += continue_loss.item()
            
        # RETURNS
        return {
            'aux/reward_loss': total_reward_loss / num_batches,
            'aux/continue_loss': total_continue_loss / num_batches,
        }
        
    
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