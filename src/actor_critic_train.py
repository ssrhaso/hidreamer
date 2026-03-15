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
            continue_logits = self.continue_network(features).squeeze(-1)       # (B*L) , raw logits
            continue_target = (~dones).float().reshape(-1)       # (2048, ) , 1 if not done, 0 if done
            continue_loss = -self.Bernoulli(logits = continue_logits).log_prob(continue_target)
            continue_loss = continue_loss.mean()
            
            # REWARD NETWORK LOSS (MSE) - SymLog of rewards (range handling)
            reward_prediction = self.reward_network(features).squeeze(-1)
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
            ) 
        
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
        
        # INTERACTION LOOP
        while not done:
            
            
            # POLICY ACTION SELECTION
            with torch.no_grad():
                # ENCODE OBSERVATION TO HRVQ TOKENS (28224 raw values -> 3 discrete tokens)
                tokens = self._encode_observation(obs)                      # (3, ) [L0, L1, L2] 
            
                # DENSE FLOAT VECTOR (3 tokens -> 512 dim feature vector) for policy input
                features = self.feature_extractor(tokens.unsqueeze(0))      # (1, feature_dim ) 
            
                distribution = self.policy(features)    #  logits
                action = distribution.sample().item()   #  sampled action index
            
            # STEP ENV
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
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
            
            # INTERACTION LOOP
            while not done:
                # ENCODE OBSERVATION TO HRVQ TOKENS (28224 raw values -> 3 discrete tokens)
                tokens = self._encode_observation(obs)                      # (3, ) [L0, L1, L2] 
            
                # DENSE FLOAT VECTOR (3 tokens -> 512 dim feature vector) for policy input
                features = self.feature_extractor(tokens.unsqueeze(0))      # (1, feature_dim ) 
                
                # GREEDY ACTION SELECTION (for evaluation)
                distribution = self.policy(features)                  #  prob logits
                action = distribution.probs.argmax(dim = -1).item()   #  greedily take highest prob
                
                # STEP ENV and ACCUMULATE
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
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
        """  MAIN TRAINING LOOP """
        
        """ SETUP """
        # FREEZE MODELS
        for model in [self.world_model, self.hrvq_tokenizer, self.encoder]:
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
        
        print(f"STARTING POLICY TRAINING")
        print(f"  Parameters: {count_policy_params(self.policy)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Horizon schedule: mode={policy_config.get('horizon_mode', 'decay')}  H=[{policy_config.get('min_horizon', 5)}, {policy_config.get('max_horizon', 30)}]")
        print(f"  Batch size: {policy_config['batch_size']}")
        print(f"  Offline mode: {offline_mode}")
        print(f"  Buffer size: {len(self.replay_buffer)}")
        
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
            
            """ Train Aux Networks (REAL DATA)"""
            aux_metrics = self._train_aux(
                num_batches = policy_config.get('aux_batches', 1),
            )
            
            """ Imagination Rollout (WORLD MODEL latent rollout)  """
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
                
                print(
                    f"Step {step:>6d}/{total_steps} | "
                    f"H={current_horizon} | "
                    f"actor_loss={actor_critic_metrics['actor/loss']:.4f} | "
                    f"critic_loss={actor_critic_metrics['critic/loss']:.4f} | "
                    f"entropy={actor_critic_metrics['actor/entropy']:.3f} | "
                    f"returns={actor_critic_metrics['returns/mean']:.3f} | "
                    f"reward_loss={aux_metrics['aux/reward_loss']:.4f} | "
                    f"sps={steps_per_second:.1f}"
                )
                
                if use_wandb:
                    metrics = {
                        **aux_metrics,
                        **actor_critic_metrics,
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
        torch.save({
            'step': step,
            'best_return': best_return,
            'game': self.config.get('game', 'unknown'),   
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'reward_net_state_dict': self.reward_network.state_dict(),
            'continue_net_state_dict': self.continue_network.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'aux_optim_state_dict': self.aux_optimizer.state_dict(),
        }, path)
        
        
        
        # Upload to W&B Artifacts
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"hidreamer-actor-critic-{self.config.get('game', 'unknown')}-step{step}",
                type="model",
                metadata={'step': step, 'best_return': best_return},
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        print(f"    Saved Checkpoint: {path}")