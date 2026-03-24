"""
ACTOR-CRITIC TRAINER FOR SPATIAL HI-DREAMER

Differences from actor_critic_train.py:
  - No hrvq_tokenizer or encoder params (spatial WM is the only frozen model)
  - _train_aux uses stride-37 level pooling on spatial hidden states
  - seed context dict uses tokens_l0/l1/l2 keys (not 'tokens')
  - Online collection / _encode_observation stubbed (offline-only)
"""

import os
import time
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
    HiddenStateFeatureExtractor,
    compute_lambda_returns,
    ReturnNormalizer,
    symlog,
    symexp,
    count_policy_params,
    get_horizon,
)

from imagination_spatial import SpatialImagineRollout, SpatialTrajectory
from replay_buffer_spatial import SpatialTokenReplayBuffer


class SpatialActorCriticTrainer:
    """Full training loop for spatial policy learning (offline mode)."""

    def __init__(
        self,
        world_model:       nn.Module,
        feature_extractor: HiddenStateFeatureExtractor,
        policy:            ActorNetwork,
        critic:            CriticNetwork,
        reward_network:    RewardNetwork,
        continue_network:  ContinueNetwork,
        imagination:       SpatialImagineRollout,
        replay_buffer:     SpatialTokenReplayBuffer,
        config:            dict,
        device:            torch.device = None,
    ):
        self.world_model      = world_model
        self.feature_extractor = feature_extractor
        self.policy           = policy
        self.critic           = critic
        self.reward_network   = reward_network
        self.continue_network = continue_network
        self.imagination      = imagination
        self.replay_buffer    = replay_buffer
        self.config           = config
        self.device           = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        policy_config = config['policy']

        self.actor_optimizer = Adam(
            params=(
                [p for p in policy.parameters()             if p.requires_grad]
                + [p for p in feature_extractor.parameters() if p.requires_grad]
            ),
            lr=policy_config['actor_lr'],
            eps=1e-5,
        )

        self.critic_optimizer = Adam(
            params=critic.parameters(),
            lr=policy_config['critic_lr'],
            eps=1e-5,
        )

        self.aux_optimizer = Adam(
            params=list(reward_network.parameters()) + list(continue_network.parameters()),
            lr=policy_config['aux_lr'],
            eps=1e-5,
        )

        from policy import CPCHead
        if policy_config.get('use_cpc', False):
            self.cpc_head = CPCHead(
                feat_dim    = feature_extractor.feat_dim,
                proj_dim    = policy_config.get('cpc_proj_dim', 256),
                k_steps     = policy_config.get('cpc_k_steps', 3),
                temperature = policy_config.get('cpc_temperature', 0.1),
            ).to(self.device)
            self.cpc_optimizer = Adam(
                params=self.cpc_head.parameters(),
                lr=policy_config.get('cpc_lr', 1e-4),
                eps=1e-5,
            )
        else:
            self.cpc_head      = None
            self.cpc_optimizer = None

        self.slow_target = CriticMovingAverage(
            critic=self.critic,
            tau=policy_config['critic_slow_target_tau'],
        )

        self.return_normalizer = ReturnNormalizer(
            decay=policy_config['return_normalizer_decay'],
        )

        self.global_step = 0
        self.Bernoulli   = Bernoulli

    def _train_aux(self, num_batches: int = 1) -> dict:
        """Train reward and continue networks on real spatial data."""
        self.reward_network.train()
        self.continue_network.train()
        total_reward_loss   = 0.0
        total_continue_loss = 0.0

        for _ in range(num_batches):
            batch   = self.replay_buffer.sample(self.config['policy']['aux_batch_size'])
            tokens_l0 = batch['tokens_l0']   # (B, L, 4)
            tokens_l1 = batch['tokens_l1']   # (B, L, 16)
            tokens_l2 = batch['tokens_l2']   # (B, L, 16)
            actions   = batch['actions']     # (B, L)
            rewards   = batch['rewards']     # (B, L)
            dones     = batch['dones']       # (B, L)

            B, L = tokens_l0.shape[:2]

            with torch.no_grad():
                out = self.world_model(tokens_l0, tokens_l1, tokens_l2, actions)
                x   = out['hidden']          # (B, L*37, D)
                D   = x.size(-1)

                x_ts    = x.reshape(B, L, 37, D)
                mean_l0 = x_ts[:, :, 0:4,   :].mean(dim=2)   # (B, L, D)
                mean_l1 = x_ts[:, :, 4:20,  :].mean(dim=2)
                mean_l2 = x_ts[:, :, 20:36, :].mean(dim=2)

                hidden_3 = torch.stack([mean_l0, mean_l1, mean_l2], dim=2)  # (B, L, 3, D)
                features = self.feature_extractor(
                    hidden_3.reshape(B * L, 3, D)
                )                                                            # (B*L, feat_dim)

            continue_logits = self.continue_network(features).squeeze(-1)
            continue_target = (~dones).float().reshape(-1)
            continue_loss   = -self.Bernoulli(logits=continue_logits).log_prob(continue_target).mean()

            reward_pred   = self.reward_network(features).squeeze(-1)
            reward_target = symlog(rewards.reshape(-1))
            nz_weight     = self.config['policy'].get('reward_nonzero_weight', 20.0)
            weights       = torch.ones_like(reward_target)
            weights[reward_target.abs() > 1e-4] = nz_weight
            reward_loss   = (weights * (reward_pred - reward_target) ** 2).mean()

            aux_loss = reward_loss + continue_loss
            self.aux_optimizer.zero_grad()
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.reward_network.parameters()) + list(self.continue_network.parameters()),
                max_norm=self.config['policy']['aux_max_norm'],
            )
            self.aux_optimizer.step()
            total_reward_loss   += reward_loss.item()
            total_continue_loss += continue_loss.item()

        return {
            'aux/reward_loss':             total_reward_loss / num_batches,
            'aux/continue_loss':           total_continue_loss / num_batches,
            'aux/reward_pred_std':         reward_pred.std().item(),
            'aux/reward_pred_nonzero_frac': (reward_pred.abs() > 0.1).float().mean().item(),
        }

    def _train_actor_critic(self, trajectory: SpatialTrajectory) -> dict:
        """Train actor and critic on imagined trajectory."""
        policy_config = self.config['policy']
        gamma         = policy_config['gamma']
        lam           = policy_config['lambda']
        entropy_scale = policy_config['entropy_scale']

        B, H, feat_dim = trajectory.feats.shape

        with torch.no_grad():
            feats_flat  = trajectory.feats.reshape(B * H, feat_dim)
            slow_values = self.slow_target(feats_flat).reshape(B, H)
            slow_last   = self.slow_target(trajectory.last_feat)

        rewards_real   = symexp(trajectory.rewards)
        lambda_returns = compute_lambda_returns(
            rewards     = rewards_real,
            values      = slow_values,
            continues   = trajectory.continues,
            last_value  = slow_last,
            gamma       = gamma,
            lam         = lam,
        )

        self.return_normalizer.update(lambda_returns)
        self.critic.train()
        values       = self.critic(feats_flat.detach()).reshape(B, H)
        value_targets = symlog(lambda_returns).detach()
        critic_loss   = 0.5 * F.mse_loss(values, value_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config['policy']['critic_max_norm'])
        self.critic_optimizer.step()
        self.slow_target.update(self.critic)

        self.policy.train()
        with torch.no_grad():
            normalized_returns = self.return_normalizer.normalize(lambda_returns)
            normalized_values  = self.return_normalizer.normalize(slow_values)
            advantages         = normalized_returns - normalized_values

        actor_loss       = -(trajectory.log_probs * advantages.detach()).mean()
        entropy_loss     = -trajectory.entropies.mean()
        total_actor_loss = actor_loss + entropy_scale * entropy_loss

        cpc_loss_val = 0.0
        if self.cpc_head is not None:
            cpc_scale        = self.config['policy'].get('cpc_scale', 0.5)
            cpc_loss         = self.cpc_head(trajectory.feats)
            total_actor_loss = total_actor_loss + cpc_scale * cpc_loss
            cpc_loss_val     = cpc_loss.item()

        self.actor_optimizer.zero_grad()
        if self.cpc_optimizer is not None:
            self.cpc_optimizer.zero_grad()

        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.feature_extractor.parameters()),
            max_norm=self.config['policy']['actor_max_norm'],
        )
        self.actor_optimizer.step()

        if self.cpc_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.cpc_head.parameters(), max_norm=1.0)
            self.cpc_optimizer.step()

        return {
            'actor/loss':       actor_loss.item(),
            'actor/entropy':    trajectory.entropies.mean().item(),
            'actor/cpc_loss':   cpc_loss_val,
            'critic/loss':      critic_loss.item(),
            'critic/value_mean': values.mean().item(),
            'returns/mean':     lambda_returns.mean().item(),
            'returns/std':      lambda_returns.std().item(),
            'advantages/mean':  advantages.mean().item(),
        }

    def train(
        self,
        total_steps:    int  = 200_000,
        use_wandb:      bool = True,
        save_directory: str  = 'checkpoints/policy_spatial',
        log_interval:   int  = 100,
    ):
        """Main offline training loop."""
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad = False

        os.makedirs(save_directory, exist_ok=True)
        policy_config = self.config['policy']

        if use_wandb:
            wandb.init(
                project=self.config['logging'].get('wandb_project', 'hi-dreamer-spatial-policy'),
                config=self.config,
                resume='allow',
            )

        print(f"STARTING SPATIAL POLICY TRAINING")
        print(f"  Parameters: {count_policy_params(critic=self.critic, actor=self.policy, reward_net=self.reward_network, continue_net=self.continue_network, feature_extractor=self.feature_extractor)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Buffer size: {len(self.replay_buffer)}")
        print()

        start_time = time.time()
        best_metric = float('inf')

        horizon_max  = policy_config.get('max_horizon', 15)
        horizon_min  = policy_config.get('min_horizon', 5)
        horizon_mode = policy_config.get('horizon_mode', 'flat')

        for step in range(total_steps):
            self.global_step = step

            current_horizon = get_horizon(
                current_step = step,
                total_steps  = total_steps,
                min_horizon  = horizon_min,
                max_horizon  = horizon_max,
                flat_horizon = policy_config.get('flat_horizon', 15),
                mode         = horizon_mode,
            )

            aux_metrics = self._train_aux(
                num_batches=policy_config.get('aux_batches', 1),
            )

            self.world_model.eval()

            use_reward_biased = policy_config.get('reward_biased_seed', False)
            if use_reward_biased:
                seed_context = self.replay_buffer.sample_reward_biased_seed(
                    batch_size=policy_config['batch_size'],
                    context_len=policy_config['seed_context_len'],
                    nonzero_fraction=policy_config.get('reward_biased_fraction', 0.5),
                )
            else:
                seed_context = self.replay_buffer.sample_seed_context(
                    batch_size=policy_config['batch_size'],
                    context_len=policy_config['seed_context_len'],
                )

            imagined_trajectory = self.imagination.rollout(
                seed_context=seed_context,
                horizon=current_horizon,
            )

            ac_metrics = self._train_actor_critic(imagined_trajectory)

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                sps     = (step + 1) / max(elapsed, 1)
                cpc_str = (f" | cpc={ac_metrics['actor/cpc_loss']:.3f}" if self.cpc_head is not None else "")
                print(
                    f"Step {step:>6d}/{total_steps} | "
                    f"H={current_horizon} | "
                    f"actor={ac_metrics['actor/loss']:.4f} | "
                    f"critic={ac_metrics['critic/loss']:.4f} | "
                    f"ent={ac_metrics['actor/entropy']:.3f} | "
                    f"ret={ac_metrics['returns/mean']:.3f} | "
                    f"rew_loss={aux_metrics['aux/reward_loss']:.4f} | "
                    f"rew_std={aux_metrics['aux/reward_pred_std']:.4f}"
                    + cpc_str
                    + f" | sps={sps:.1f}"
                )
                if use_wandb:
                    wandb.log({**aux_metrics, **ac_metrics, 'speed/sps': sps, 'horizon/current': current_horizon}, step=step)

            if step % policy_config.get('eval_every', 5000) == 0 and step > 0:
                self._save_checkpoint(os.path.join(save_directory, f"policy_step_{step}.pt"), step)

        self._save_checkpoint(os.path.join(save_directory, "final_policy.pt"), total_steps)
        print(f"\nTraining complete in {(time.time() - start_time) / 3600:.1f} hours")
        if use_wandb:
            wandb.finish()

    def _save_checkpoint(self, path: str, step: int):
        checkpoint = {
            'step':                       step,
            'policy_state_dict':          self.policy.state_dict(),
            'critic_state_dict':          self.critic.state_dict(),
            'reward_net_state_dict':      self.reward_network.state_dict(),
            'continue_net_state_dict':    self.continue_network.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_optim_state_dict':     self.actor_optimizer.state_dict(),
            'critic_optim_state_dict':    self.critic_optimizer.state_dict(),
            'aux_optim_state_dict':       self.aux_optimizer.state_dict(),
        }
        if self.cpc_head is not None:
            checkpoint['cpc_head_state_dict']  = self.cpc_head.state_dict()
            checkpoint['cpc_optim_state_dict'] = self.cpc_optimizer.state_dict()

        torch.save(checkpoint, path)
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"hidreamer-spatial-policy-step{step}",
                type="model",
                metadata={'step': step},
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        print(f"    Saved checkpoint: {path}")
