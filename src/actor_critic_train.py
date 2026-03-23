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
from torch.optim import Adam, AdamW
from torch import autocast, GradScaler
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

from world_model import hierarchical_loss
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
        
        # BUG FIX: feature_extractor was being gradient-clipped but NEVER stepped,
        # because it wasn't in any optimizer.  Include it here so its projection
        # layer (hidden-state mode) or visual_encoder (visual mode) actually trains.
        # Filter by requires_grad so frozen decoder params (visual mode) are excluded.
        self.actor_optimizer = Adam(
            params = (
                [p for p in policy.parameters()            if p.requires_grad]
                + [p for p in feature_extractor.parameters() if p.requires_grad]
            ),
            lr  = policy_config['actor_lr'],
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
        
        # WORLD MODEL OPTIMIZER (JOINT TRAINING)
        jt_config = config.get('joint_training', {})
        self.joint_training_enabled = jt_config.get('enabled', False)

        if self.joint_training_enabled:
            # AdamW matches Phase 2 training (world_model_train.py)
            # Lower LR (1e-4 vs 3e-4) because we're fine-tuning, not training from scratch
            self.wm_optimizer = AdamW(
                params = [p for p in world_model.parameters() if p.requires_grad],
                lr = jt_config.get('wm_lr', 1e-4),
                weight_decay = jt_config.get('wm_weight_decay', 0.01),
                betas = (0.9, 0.95),   # Same as Phase 2 (GPT/STORM convention)
            )

            use_amp = jt_config.get('use_amp', True) and self.device.type == 'cuda'
            self.wm_scaler = GradScaler(enabled = use_amp)
            self.wm_use_amp = use_amp
            self.wm_grad_clip = jt_config.get('wm_grad_clip', 1.0)
            self.wm_train_ratio = jt_config.get('wm_train_ratio', 1)
            self.wm_batch_size = jt_config.get('wm_batch_size', 32)
            self.wm_seq_len = jt_config.get('wm_seq_len', 64)
            self.wm_layer_weights = config.get('model', {}).get(
                'layer_weights', [1.0, 0.5, 0.1]
            )
            # If model config not in policy yaml, pull from worldmodel config
            if 'model' not in config:
                self.wm_layer_weights = self.world_model.config.layer_weights


        # AC-CPC head (optional) — temporal contrastive loss that trains
        # feature_extractor to produce position/velocity-aware representations.
        # Without this, the WM features are never shaped to encode the continuous
        # quantities (paddle pos, ball pos) that the policy needs for control.
        from policy import CPCHead
        if policy_config.get('use_cpc', False):
            self.cpc_head = CPCHead(
                feat_dim    = feature_extractor.feat_dim,
                proj_dim    = policy_config.get('cpc_proj_dim', 256),
                k_steps     = policy_config.get('cpc_k_steps', 3),
                temperature = policy_config.get('cpc_temperature', 0.1),
            ).to(self.device)
            # CPC head is a tiny MLP; stepped together with actor since its loss
            # backprops through feature_extractor (which is now in actor_optimizer).
            self.cpc_optimizer = Adam(
                params = self.cpc_head.parameters(),
                lr = policy_config.get('cpc_lr', 1e-4),
                eps = 1e-5,
            )
        else:
            self.cpc_head = None
            self.cpc_optimizer = None

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
        
    def _train_world_model(
        self,
        num_steps : int = 1,
    ) -> dict:
        """
        Train the (unfrozen) world model on sequences from the replay buffer.
        
        Uses the SAME hierarchical loss as Phase 2 training (world_model_train.py).
        Called inside the main training loop when joint_training.enabled = true.
        """
        self.world_model.train()

        total_loss = 0.0
        total_acc_l0 = 0.0
        total_acc_l1 = 0.0
        total_acc_l2 = 0.0
        
        for _ in range(num_steps):
            
            # Sample a sequence batch from the replay buffer
            batch = self.replay_buffer.sample_wm_batch(
                batch_size = self.wm_batch_size,
                seq_len = self.wm_seq_len,
            )
            
            tokens = batch['tokens']    # (B, L, 3) 
            actions = batch['actions']  # (B, L)
            
            # Forward pass with AMP (matches Phase 2 training)
            with autocast(device_type = self.device.type, enabled = self.wm_use_amp):
                logits_l0, logits_l1, logits_l2 = self.world_model(tokens, actions)
                
                loss, metrics = hierarchical_loss(
                    logits_l0 = logits_l0,
                    logits_l1 = logits_l1,
                    logits_l2 = logits_l2,
                    tokens = tokens,
                    layer_weights = self.wm_layer_weights,
                )
            
            # Backward pass with GradScaler
            self.wm_optimizer.zero_grad()
            self.wm_scaler.scale(loss).backward()
            self.wm_scaler.unscale_(self.wm_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(),
                max_norm = self.wm_grad_clip,
            )
            self.wm_scaler.step(self.wm_optimizer)
            self.wm_scaler.update()
            
            total_loss += metrics['loss_total']
            total_acc_l0 += metrics['accuracy_l0']
            total_acc_l1 += metrics['accuracy_l1']
            total_acc_l2 += metrics['accuracy_l2']
        
        # Switch back to eval for imagination rollouts
        self.world_model.eval()
        
        return {
            'wm/loss': total_loss / num_steps,
            'wm/acc_l0': total_acc_l0 / num_steps,
            'wm/acc_l1': total_acc_l1 / num_steps,
            'wm/acc_l2': total_acc_l2 / num_steps,
        }
    
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

            actions = batch['actions']         # (B, L)

            is_visual = getattr(self.feature_extractor, 'is_visual_mode', False)

            if is_visual:
                # VISUAL MODE: decode directly from replay tokens — NO WM forward needed.
                # Reward/continue networks use the same visual features as the actor.
                tokens_flat = tokens.reshape(B * L, 3)                       # (B*L, 3)
                features    = self.feature_extractor(tokens_flat)             # (B*L, feat_dim)
            else:
                # HIDDEN STATE MODE: run frozen WM to get hidden states.
                with torch.no_grad():
                    x = self.world_model.embedding(tokens, actions)
                    mask = self.world_model._get_mask(x.size(1), x.device)
                    for block in self.world_model.blocks:
                        x = block(x, mask=mask)
                    x = self.world_model.ln_final(x)                         # (B, L*4, d_model)

                h_l0 = x[:, 0::4, :]   # (B, L, d_model)
                h_l1 = x[:, 1::4, :]
                h_l2 = x[:, 2::4, :]
                hidden_states = torch.cat([h_l0, h_l1, h_l2], dim=-1)       # (B, L, d_model*3)
                features = self.feature_extractor(hidden_states.reshape(B * L, -1))
            
            # CONTINUE NETWORK LOSS (Bernoulli) - Binary Classification 
            continue_logits = self.continue_network(features).squeeze(-1)       # (B*L) , raw logits
            continue_target = (~dones).float().reshape(-1)       # (2048, ) , 1 if not done, 0 if done
            continue_loss = -self.Bernoulli(logits = continue_logits).log_prob(continue_target)
            continue_loss = continue_loss.mean()
            
            # REWARD NETWORK LOSS (WEIGHTED MSE) - SymLog of rewards (range handling)
            # WHY WEIGHTED: Pong has ~1% non-zero reward steps. Plain MSE collapses to
            # predicting 0 everywhere (loss=0.0002 is the degenerate zero-prediction regime).
            # Upweighting rare events by reward_nonzero_weight forces the network to actually
            # learn the reward structure instead of ignoring it.
            reward_prediction = self.reward_network(features).squeeze(-1)
            reward_target = symlog(rewards.reshape(-1))

            nz_weight = self.config['policy'].get('reward_nonzero_weight', 20.0)
            reward_weights = torch.ones_like(reward_target)
            reward_weights[reward_target.abs() > 1e-4] = nz_weight
            reward_loss = (reward_weights * (reward_prediction - reward_target) ** 2).mean()
            
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
            
        # RETURNS — include reward-network health diagnostics
        # reward_pred_std near-zero → degenerate collapse (predicting ~0 everywhere)
        # reward_pred_nonzero_frac near-zero → not detecting reward events
        return {
            'aux/reward_loss': total_reward_loss / num_batches,
            'aux/continue_loss': total_continue_loss / num_batches,
            'aux/reward_pred_std': reward_prediction.std().item(),
            'aux/reward_pred_nonzero_frac': (reward_prediction.abs() > 0.1).float().mean().item(),
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
        
        B, H, feat_dim = trajectory.feats.shape
        
        # COMPUTE λ RETURNS 
        with torch.no_grad():
            
            # BATCHED slow-target values: (B, H, feat_dim) -> (B*H, feat_dim) -> (B, H)
            feats_flat = trajectory.feats.reshape(B * H, feat_dim)       # (B*H, feat_dim)
            slow_values = self.slow_target(feats_flat).reshape(B, H)     # (B, H)
            
            # Bootstrap value for last state in trajectory
            slow_last = self.slow_target(trajectory.last_feat)                       # (B,)
        
        # CONVERT REWARDS from SymLog space -> real space for λ returns
        rewards_real = symexp(trajectory.rewards)  # (B, H)
        
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
        
        # BATCHED critic values: single forward pass over all (B*H) states
        values = self.critic(feats_flat.detach()).reshape(B, H)   # (B, H)
        
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
            # NORMALIZED RETURNS AND ADVANTAGES (stability, DreamerV3)
            normalized_returns = self.return_normalizer.normalize(lambda_returns)
            normalized_values = self.return_normalizer.normalize(slow_values)
            
            advantages = normalized_returns - normalized_values
            
        # REINFORCE Loss
        actor_loss = -(trajectory.log_probs * advantages.detach()).mean()

        # ENTROPY REGULARIZATION
        entropy_loss = -trajectory.entropies.mean()

        total_actor_loss = actor_loss + entropy_scale * entropy_loss

        # AC-CPC AUXILIARY LOSS (if enabled)
        # trajectory.feats: (B, H, feat_dim) — in computation graph because
        # feature_extractor is trainable (now in actor_optimizer after bug fix).
        # CPC loss trains feature_extractor to produce temporally predictive
        # representations (encoding ball/paddle position and velocity).
        cpc_loss_val = 0.0
        if self.cpc_head is not None:
            cpc_scale = self.config['policy'].get('cpc_scale', 0.5)
            cpc_loss = self.cpc_head(trajectory.feats)
            total_actor_loss = total_actor_loss + cpc_scale * cpc_loss
            cpc_loss_val = cpc_loss.item()

        # BACKPROP AND OPTIMIZE (actor + feature_extractor + cpc_head in one pass)
        self.actor_optimizer.zero_grad()
        if self.cpc_optimizer is not None:
            self.cpc_optimizer.zero_grad()

        total_actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.feature_extractor.parameters()),
            max_norm = self.config['policy']['actor_max_norm']
        )
        self.actor_optimizer.step()

        if self.cpc_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.cpc_head.parameters(), max_norm=1.0)
            self.cpc_optimizer.step()
        
        # RETURNS
        return {
            'actor/loss': actor_loss.item(),
            'actor/entropy': trajectory.entropies.mean().item(),
            'actor/cpc_loss': cpc_loss_val,
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
    
    def collect_real_episode(
        self
    ) -> dict:
        """
        Collect one full episode of real environment interaction using current policy.
        Push transition to replay buffer.
        """
        
        # ERROR HANDLING
        if self.env is None:
            raise ValueError("Environment not provided for real episode collection.")
        
        # RESET ENV (Initial Obs)
        obs , _ = self.env.reset()  # (4, 84, 84) uint8 
        episode_return = 0.0
        episode_length = 0
        done = False
        
        token_history = []
        action_history = []
        
        # INTERACTION LOOP
        while not done:
            
            
            # POLICY ACTION SELECTION
            with torch.no_grad():
                # ENCODE OBSERVATION TO HRVQ TOKENS (28224 raw values -> 3 discrete tokens)
                tokens = self._encode_observation(obs)                      # (3, ) [L0, L1, L2] 
            
                # FEATURE EXTRACTION
                features, token_history, action_history = self._get_online_features(
                    tokens, action_history, token_history
                )
            
                distribution = self.policy(features)    #  logits
                action = distribution.sample().item()   #  sampled action index
            
            # STEP ENV
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            action_history.append(action)
            done = terminated or truncated
            
            # PUSH TO BUFFER
            self.replay_buffer.push(
                tokens = tokens.cpu(),
                action = action,
                reward = reward, 
                done = done,
            )
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
        
        self.episodes_collected += 1
        
        return {
            'episode_return': episode_return,
            'episode_length': episode_length,
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        num_episodes : int = 5,
    ) -> dict:
        """ Evaluate policy on real environment """
        
        # ERROR HANDLING
        if self.env is None:
            return {}
        
        # EVAL LOOP - Setup
        self.policy.eval()
        returns = []
        lengths = []
        
        # EVAL LOOP
        for _ in range(num_episodes):
            # RESET ENV (Initial Obs)
            obs, _ = self.env.reset()  
            
            # INIT EPISODE VARIABLES
            episode_return = 0.0
            episode_length = 0
            done = False
            
            action_history = []
            token_history = []
            
            # INTERACTION LOOP
            while not done:
                # ENCODE OBSERVATION TO HRVQ TOKENS (28224 raw values -> 3 discrete tokens)
                tokens = self._encode_observation(obs)                      # (3, ) [L0, L1, L2] 
            
                # DENSE FLOAT VECTOR (3 tokens -> 512 dim feature vector) for policy input
                features, token_history, action_history = self._get_online_features(
                    tokens, action_history, token_history
                )
                
                
                # GREEDY ACTION SELECTION (for evaluation)
                distribution = self.policy(features)                  #  prob logits
                action = distribution.probs.argmax(dim = -1).item()   #  greedily take highest prob
                
                # STEP ENV and ACCUMULATE
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                action_history.append(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
                obs = next_obs
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        self.policy.train()
        return {
            'eval/return_mean': np.mean(returns),
            'eval/return_std': np.std(returns),
            'eval/length_mean': np.mean(lengths),
        }
                
            
    
    
    def train(
        self,
        total_steps : int = 200_000,
        use_wandb : bool = True,
        save_directory : str = 'checkpoints/policy',
        eval_interval : int = 10_000,
        log_interval : int = 100,
        prefill_steps : int = 0,
        offline_mode : bool = True,
    ):
        """  MAIN TRAINING LOOP 
        
        When JOINT TRAINING, world model unfreezes and gets gradient steps on the growing buffer"""
        
        """ SETUP """
        joint_active = self.joint_training_enabled and not offline_mode

        if joint_active:
            # World model is TRAINABLE — keep it unfrozen
            # (already unfrozen by load_frozen_models when joint_training.enabled)
            print("  JOINT TRAINING: World model is UNFROZEN and will receive gradient updates")
            print(f"  WM train ratio: {self.wm_train_ratio} steps per collect_interval")
            print(f"  WM learning rate: {self.config['joint_training']['wm_lr']}")
        else:
            # FREEZE world model (standard offline mode)
            self.world_model.eval()
            for param in self.world_model.parameters():
                param.requires_grad = False
        
        # HRVQ + Encoder are ALWAYS frozen
        for model in [self.hrvq_tokenizer, self.encoder]:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        os.makedirs(name = save_directory, exist_ok = True)
        policy_config = self.config['policy']
        
        # WANDB
        if use_wandb:
            wandb.init(
                project = "hi-dreamer-policy",
                config = self.config,
                resume = "allow",
            )
        
        """ PREFILL REPLAY BUFFER """
        
        if not offline_mode and prefill_steps > 0:
            print(f"Prefilling replay buffer with {prefill_steps} random steps...")
            while len(self.replay_buffer) < prefill_steps:
                self.collect_real_episode()
            print(f"  Buffer size: {len(self.replay_buffer)}")
        
        wm_trainable = sum(p.numel() for p in self.world_model.parameters() if p.requires_grad)
        
        print(f"STARTING POLICY TRAINING")
        print(f"  Parameters: {count_policy_params(
            critic=self.critic, actor=self.policy,
            reward_net=self.reward_network, continue_net=self.continue_network,
            feature_extractor=self.feature_extractor
        )}")
        print()
        print(f"  World model params: {wm_trainable:,} ({'TRAINABLE' if joint_active else 'FROZEN'})")
        print()
        print(f"  Total steps: {total_steps}")
        print()
        print(f"  Horizon schedule: mode={policy_config.get('horizon_mode', 'decay')}  H=[{policy_config.get('min_horizon', 5)}, {policy_config.get('max_horizon', 30)}]")
        print()
        print(f"  Batch size: {policy_config['batch_size']}")
        print(f"  Mode: {'JOINT (online + WM updates)' if joint_active else 'OFFLINE (frozen WM)' if offline_mode else 'ONLINE (frozen WM)'}")
        print()
        print(f"  Buffer size: {len(self.replay_buffer)}")
        print()
        
        start_time = time.time()
        best_eval_return = -float('inf')
        
        # HORIZON SCHEDULE PARAMETERS 
        horizon_max = policy_config.get('max_horizon', 30)
        horizon_min = policy_config.get('min_horizon', 5)
        horizon_mode = policy_config.get('horizon_mode', 'decay')      # 'flat', 'decay', 'bell'
        
        """ MAIN LOOP """
        for step in range(total_steps):
            
            self.global_step = step
            
            current_horizon = get_horizon(
                current_step = step,
                total_steps = total_steps,
                min_horizon = horizon_min,
                max_horizon = horizon_max,
                flat_horizon = policy_config.get('flat_horizon', 15),
                mode = horizon_mode,
            )
            
            """ Collect Real Data """        
            if not offline_mode and step % policy_config.get('collect_interval', 1) == 0:
                episode_info = self.collect_real_episode()
                
                if use_wandb and episode_info:
                    wandb.log({
                        'env/episode_return': episode_info['episode_return'],
                        'env/episode_length': episode_info['episode_length'],
                    } , step = step)
            
            """ Joint Mode Training """
            wm_metrics = {}

            if joint_active and len(self.replay_buffer) >= self.wm_seq_len:
                wm_metrics = self._train_world_model(
                    num_steps = self.wm_train_ratio,
                )
            
            """ Train Aux Networks (REAL DATA)"""
            aux_metrics = self._train_aux(
                num_batches = policy_config.get('aux_batches', 1),
            )
            
            """ Imagination Rollout (WORLD MODEL latent rollout)  """
            self.world_model.eval()  # Ensure world model is in eval mode for imagination

            # Use reward-biased seeding when configured — ensures imagination rollouts
            # encounter scoring events instead of all-zero-reward trajectories (see diagnosis).
            use_reward_biased = policy_config.get('reward_biased_seed', False)
            if use_reward_biased:
                seed_context = self.replay_buffer.sample_reward_biased_seed(
                    batch_size = policy_config['batch_size'],
                    context_len = policy_config['seed_context_len'],
                    nonzero_fraction = policy_config.get('reward_biased_fraction', 0.5),
                )
            else:
                seed_context = self.replay_buffer.sample_seed_context(
                    batch_size = policy_config['batch_size'],
                    context_len = policy_config['seed_context_len']
                )

            imagined_trajectory = self.imagination.rollout(
                seed_tokens = seed_context['tokens'],
                seed_actions = seed_context['actions'],
                horizon = current_horizon,
            )
            
            """ Train Actor Critic Networks (IMAGINED DATA) """
            actor_critic_metrics = self._train_actor_critic(imagined_trajectory)
            
            """ Logging """
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_second = (step + 1) / max(elapsed, 1)
                
                # Build log string
                cpc_str = (f" | cpc={actor_critic_metrics['actor/cpc_loss']:.3f}"
                           if self.cpc_head is not None else "")
                log_str = (
                    f"Step {step:>6d}/{total_steps} | "
                    f"H={current_horizon} | "
                    f"actor={actor_critic_metrics['actor/loss']:.4f} | "
                    f"critic={actor_critic_metrics['critic/loss']:.4f} | "
                    f"ent={actor_critic_metrics['actor/entropy']:.3f} | "
                    f"ret={actor_critic_metrics['returns/mean']:.3f} | "
                    f"rew_loss={aux_metrics['aux/reward_loss']:.4f} | "
                    f"rew_std={aux_metrics['aux/reward_pred_std']:.4f} | "
                    f"rew_nz={aux_metrics['aux/reward_pred_nonzero_frac']:.3f}"
                    + cpc_str
                )
                
                # Append WM metrics if joint training
                if wm_metrics:
                    log_str += (
                        f" | wm={wm_metrics['wm/loss']:.4f}"
                        f" L0={wm_metrics['wm/acc_l0']:.3f}"
                    )
                
                log_str += f" | sps={steps_per_second:.1f}"
                print(log_str)
                
                if use_wandb:
                    metrics = {
                        **aux_metrics,
                        **actor_critic_metrics,
                        **wm_metrics,                                   
                        'speed/sps': steps_per_second,
                        'horizon/current': current_horizon,
                    }
                    wandb.log(metrics, step = step)
        
            """ Evaluation """
            if step % eval_interval == 0 and step > 0 and not offline_mode:
                eval_metrics = self.evaluate(num_episodes=5)
                print(f"\n  EVAL @ step {step}: return={eval_metrics['eval/return_mean']:.1f}")
                
                if use_wandb:
                    wandb.log(eval_metrics, step=step)
                
                # Save best
                if eval_metrics['eval/return_mean'] > best_eval_return:
                    best_eval_return = eval_metrics['eval/return_mean']
                    self._save_checkpoint(
                        os.path.join(save_directory, "best_policy.pt"),
                        step, best_eval_return,
                    )
            
            """ Periodic Checkpoint """
            if step % eval_interval == 0 and step > 0:
                self._save_checkpoint(
                    os.path.join(save_directory, f"policy_step_{step}.pt"),
                    step, best_eval_return,
                )
        
        """ Final Save """
        self._save_checkpoint(
            os.path.join(save_directory, "final_policy.pt"),
            total_steps, best_eval_return,
        )
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 3600:.1f} hours")
        print(f"Best eval return: {best_eval_return:.1f}")
        
        if use_wandb:
            wandb.finish()
    
    def _save_checkpoint(
        self,
        path : str,
        step : int,
        best_return : float,
    ):
        """ SAVE AND LOGGING """
        checkpoint = {
            'step': step,
            'best_return': best_return,
            'game': self.config['policy'].get('game', 'unknown'),
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'reward_net_state_dict': self.reward_network.state_dict(),
            'continue_net_state_dict': self.continue_network.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'aux_optim_state_dict': self.aux_optimizer.state_dict(),
        }

        if self.cpc_head is not None:
            checkpoint['cpc_head_state_dict']  = self.cpc_head.state_dict()
            checkpoint['cpc_optim_state_dict'] = self.cpc_optimizer.state_dict()

        # Save WM state if joint training (critical for Colab resume)
        if self.joint_training_enabled:
            checkpoint['world_model_state_dict'] = self.world_model.state_dict()
            checkpoint['wm_optim_state_dict'] = self.wm_optimizer.state_dict()
            checkpoint['wm_scaler_state_dict'] = self.wm_scaler.state_dict()

        torch.save(checkpoint, path)
        
        
        
        # Upload to W&B Artifacts
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"hidreamer-actor-critic-{self.config['policy'].get('game', 'unknown')}-step{step}",
                type="model",
                metadata={'step': step, 'best_return': best_return},
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        print(f"    Saved Checkpoint: {path}")
    
    def _get_online_features(self, tokens, action_history, token_history):
        """Get features for online action selection.

        Visual mode:       O(1) — decode current tokens via frozen decoder + CNN.
                           No WM forward needed, no context history required.
        Hidden-state mode: O(n) — run WM on recent token context to get hidden states.

        Both modes maintain token_history / action_history so the replay buffer
        receives proper sequences for imagination seeding.
        """
        # Always maintain history for replay buffer seeding
        token_history.append(tokens.cpu())
        max_ctx = self.config['policy']['seed_context_len']
        if len(token_history) > max_ctx:
            token_history  = token_history[-max_ctx:]
            action_history = action_history[-max_ctx:]
        while len(action_history) < len(token_history):
            action_history.append(0)

        is_visual = getattr(self.feature_extractor, 'is_visual_mode', False)

        if is_visual:
            # Direct decode — no WM forward, no context needed for the feature
            tok     = tokens.unsqueeze(0).to(self.device)     # (1, 3)
            feature = self.feature_extractor(tok)             # (1, feat_dim)
        else:
            # Run frozen WM on recent context to get transformer hidden states
            ctx_tokens  = torch.stack(token_history).unsqueeze(0).to(self.device)   # (1, T, 3)
            ctx_actions = torch.tensor(action_history).unsqueeze(0).to(self.device) # (1, T)
            with torch.no_grad():
                x    = self.world_model.embedding(ctx_tokens, ctx_actions)
                mask = self.world_model._get_mask(x.size(1), x.device)
                for block in self.world_model.blocks:
                    x = block(x, mask=mask)
                x = self.world_model.ln_final(x)
                t          = ctx_tokens.size(1)
                last_start = (t - 1) * 4
                hidden     = x[:, last_start:last_start + 3, :]  # (1, 3, d_model)
            feature = self.feature_extractor(hidden)              # (1, feat_dim)

        return feature, token_history, action_history