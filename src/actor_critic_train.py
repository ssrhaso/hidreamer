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
            
            with torch.no_grad():
                features = self.feature_extractor(tokens_flat)   # (B*L, feature_dim) (2048, 512)
            
            # CONTINUE NETWORK LOSS (Bernoulli) - Binary Classification 
            continue_logits = self.continue_network(features)       # (B*L, 1) , raw logits
            continue_target = (~dones).float().reshape(-1)       # (2048, ) , 1 if not done, 0 if done
            continue_loss = -self.Bernoulli(logits = continue_logits).log_prob(continue_target)
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
        
    
    def _train_actor_critic(
        self,
        trajectory : Trajectory
    ) -> dict :
        """ 
        Train ActorNetwork and CriticNetwork on IMAGINED TRAJECTORY from world model.
        
        ACTOR:  Reinforce with λ-return advantages + entropy regularisation
        CRITIC: MSE on symlog λ-return targets (vs slow target)
        """
        # INIT - Load config scalars
        policy_config = self.config['policy']
        gamma = policy_config['gamma']
        lam = policy_config['lambda']
        entropy_scale = policy_config['entropy_scale']
        
        # COMPUTE λ RETURNS 
        with torch.no_grad():
            
            # USE Slow Target for stable value estimates
            slow_values = torch.zeros_like(trajectory.values)                  # (B, H) empty

            for h in range(trajectory.feats.size(1)):                          # loop over horizon
                slow_values[:, h] = self.slow_target(trajectory.feats[:, h])   # fill column h , slice to (B, 1152)
            slow_last = self.slow_target(
                self.feature_extractor(trajectory.tokens[:, -1])
            ) # (B, H) one value per step
        
        # CONVERT REWARDS from SymLog space -> real space for λ returns
        rewards_real = symexp(trajectory.rewards)  # (B, H)
        
        # Update RETURN NORMALIZER
        lambda_returns = compute_lambda_returns(
            rewards = rewards_real,
            values = slow_values,
            continues = trajectory.continues,
            last_value = slow_last,
            gamma = gamma,
            lam = lam,
        ) # (B, H)
        
        # CRITIC Update
        self.return_normalizer.update(lambda_returns)
        self.critic.train()
        
        # RECOMPUTE Values 
        values = torch.zeros_like(trajectory.values)                       # (B, H) empty
        for h in range(trajectory.feats.size(1)):                          # loop over horizon
            values[:, h] = self.critic(trajectory.feats[:, h].detach())    # fill column h , slice to (B, 1152)
        
        # CRITIC Loss
        value_targets = symlog(lambda_returns).detach()
        critic_loss = 0.5 * F.mse_loss(input = values, target = value_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = self.config['policy']['critic_max_norm'])
        self.critic_optimizer.step()
        
        # UPDATE SLOW TARGET
        self.slow_target.update(self.critic)
        
        # ACTOR UPDATE
        self.policy.train()
        
        with torch.no_grad():
            advantages = lambda_returns - slow_values
            advantages = self.return_normalizer.normalize(advantages)
            
        # REINFORCE Loss
        actor_loss = -(trajectory.log_probs * advantages.detach()).mean()
        
        # ENTROPY REGULARIZATION 
        entropy_loss = -trajectory.entropies.mean()
        
        total_actor_loss = actor_loss + entropy_scale * entropy_loss
        
        # BACKPROP AND OPTIMIZE
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.feature_extractor.parameters()), 
            max_norm = self.config['policy']['actor_max_norm']
        )
        self.actor_optimizer.step()
        
        # RETURNS
        return {
            'actor/loss': actor_loss.item(),
            'actor/entropy': trajectory.entropies.mean().item(),
            'critic/loss': critic_loss.item(),
            'critic/value_mean': values.mean().item(),
            'returns/mean': lambda_returns.mean().item(),
            'returns/std': lambda_returns.std().item(),
            'advantages/mean': advantages.mean().item(),
        }
        
    @torch.no_grad()
    def _encode_observation(
        self,
        obs : np.ndarray
    )-> torch.Tensor:
        """
        Encode raw Atari frame to HRVQ Tokens 
        (using pretrained CNN encoder and HRVQ tokenizer)
        """
        
        # NORMALISE 
        frame = obs.astype(np.float32)            # uint8 -> float32
        frame = frame / 255.0                     # normalise to [0, 1]
        frame = torch.from_numpy(frame)           # numpy array -> torch tensor (C, H, W)
        frame = frame.unsqueeze(0)                # add batch dimension (1, C, H, W)
        frame = frame.to(self.device)             # move to gpu -> (1, 4, 84, 84) fp32
    
        # ENCODER (CNN feature extractor)
        embedding = self.encoder(frame)           # -> (1, 384) fp32
        
        # HRVQ ENCODE (Tokenization + Quantization)
        embedding = embedding.unsqueeze(1).unsqueeze(2)   # Reshape -> (1, 1, 1, 384) for HRVQ
        token_list = self.hrvq_tokenizer.encode(embedding)       # HRVQ Tokenizer -> 3 codebook indices - L0, L1, L2 
        
        # Remove extra dimensions and concatenate tokens -> (1, 3)
        
        token_0 = token_list[0].squeeze(2).squeeze(1).squeeze(0)   # (1, 1, 1) -> scalar tensor 
        token_1 = token_list[1].squeeze(2).squeeze(1).squeeze(0)   # (1, 1, 1) -> scalar tensor
        token_2 = token_list[2].squeeze(2).squeeze(1).squeeze(0)   # (1, 1, 1) -> scalar tensor
        
        # CONCATENATE TOKENS INTO SINGLE TENSOR
        tokens = torch.stack(tensors = [token_0, token_1, token_2], dim = -1)  # (3,)
    

        return tokens
    
    def collect_real_episode():
        pass
    
    @torch.no_grad()
    def evaluate():
        pass
    
    
    def train():
        """  MAIN TRAINING LOOP """
        pass
    
    
    def _save_checkpoint():
        """ SAVE AND LOGGING """
        pass