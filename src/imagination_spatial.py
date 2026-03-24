"""
SPATIAL IMAGINATION ROLLOUT FOR POLICY TRAINING

37-token layout per timestep:
  positions 0-3   : L0 patches  (coarse, codebook 16)
  positions 4-19  : L1 patches  (mid,    codebook 64)
  positions 20-35 : L2 patches  (fine,   codebook 64)
  position  36    : action

Generation uses a 3-pass cascade per new timestep:
  Pass 1: forward on current context → sample all 4  L0 patches from logits_l0[:,-1]
  Pass 2: append new L0, forward      → sample all 16 L1 patches from logits_l1[:,-1]
  Pass 3: fill real L1, forward       → sample all 16 L2 patches from logits_l2[:,-1]

Context is capped at MAX_CONTEXT_T=15 timesteps (15*37=555 < max_seq_len=592).

Feature extraction (level-mean-pool → 1152D):
  x shape (B, T*37, D) → reshape to (B, T, 37, D)
  mean over positions [0:4]  → mean_l0 (B, T, D)
  mean over positions [4:20] → mean_l1 (B, T, D)
  mean over positions [20:36]→ mean_l2 (B, T, D)
  stack → (B, T, 3, D) → HiddenStateFeatureExtractor → (B*T, feat_dim)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from world_model_spatial import SpatialHierarchicalWorldModel

MAX_CONTEXT_T = 15


@dataclass
class SpatialTrajectory:
    """Container for one imagination rollout batch."""
    tokens_l0:  torch.Tensor   # (B, H, 4)
    tokens_l1:  torch.Tensor   # (B, H, 16)
    tokens_l2:  torch.Tensor   # (B, H, 16)
    actions:    torch.Tensor   # (B, H)
    log_probs:  torch.Tensor   # (B, H)
    feats:      torch.Tensor   # (B, H, feat_dim)
    values:     torch.Tensor   # (B, H)
    rewards:    torch.Tensor   # (B, H)
    continues:  torch.Tensor   # (B, H)
    last_value: torch.Tensor   # (B,)
    entropies:  torch.Tensor   # (B, H)
    last_feat:  torch.Tensor   # (B, feat_dim)


class SpatialImagineRollout:
    """Run horizon-step imagination inside frozen SpatialHierarchicalWorldModel."""

    def __init__(
        self,
        world_model:      SpatialHierarchicalWorldModel,
        feature_extractor,
        actor_network,
        critic_network,
        reward_network,
        continue_network,
        max_horizon: int   = 15,
        temperature: float = 1.0,
        device:      torch.device = None,
    ):
        self.world_model      = world_model
        self.feature_extractor = feature_extractor
        self.actor_network    = actor_network
        self.critic_network   = critic_network
        self.reward_network   = reward_network
        self.continue_network = continue_network
        self.max_horizon      = max_horizon
        self.temperature      = temperature
        self.device           = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def _sample_patches(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample independently from each patch position.
        logits: (B, P, vocab)
        returns: (B, P)
        """
        B, P, V = logits.shape
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        flat  = probs.reshape(B * P, V)
        samp  = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return samp.reshape(B, P)

    def _extract_features(
        self,
        context_l0:      torch.Tensor,  # (B, T, 4)
        context_l1:      torch.Tensor,  # (B, T, 16)
        context_l2:      torch.Tensor,  # (B, T, 16)
        context_actions: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        Run frozen WM, pool per level, pass through feature extractor.
        Returns (B, feat_dim) for the LAST timestep.
        """
        def _trim(t, max_t=MAX_CONTEXT_T):
            return t[:, -max_t:] if t.size(1) > max_t else t

        context_l0      = _trim(context_l0)
        context_l1      = _trim(context_l1)
        context_l2      = _trim(context_l2)
        context_actions = _trim(context_actions)

        out = self.world_model(context_l0, context_l1, context_l2, context_actions)
        x   = out['hidden']            # (B, T*37, D)
        B   = x.size(0)
        T   = context_l0.size(1)
        D   = x.size(-1)

        x_ts   = x.reshape(B, T, 37, D)
        mean_l0 = x_ts[:, -1, 0:4,   :].mean(dim=1)   # (B, D)
        mean_l1 = x_ts[:, -1, 4:20,  :].mean(dim=1)   # (B, D)
        mean_l2 = x_ts[:, -1, 20:36, :].mean(dim=1)   # (B, D)

        hidden_3 = torch.stack([mean_l0, mean_l1, mean_l2], dim=1)  # (B, 3, D)
        return self.feature_extractor(hidden_3)                      # (B, feat_dim)

    @torch.no_grad()
    def _cascade_predict_step(
        self,
        context_l0:      torch.Tensor,  # (B, T_ctx, 4)
        context_l1:      torch.Tensor,  # (B, T_ctx, 16)
        context_l2:      torch.Tensor,  # (B, T_ctx, 16)
        context_actions: torch.Tensor,  # (B, T_ctx)
    ):
        """
        3-pass cascade to generate one new (l0, l1, l2) timestep.
        Returns new_l0 (B,4), new_l1 (B,16), new_l2 (B,16).
        """
        B = context_l0.size(0)

        def _trim(t, max_t=MAX_CONTEXT_T):
            return t[:, -max_t:] if t.size(1) > max_t else t

        ctx_l0 = _trim(context_l0)
        ctx_l1 = _trim(context_l1)
        ctx_l2 = _trim(context_l2)
        ctx_a  = _trim(context_actions)

        # Pass 1: predict L0
        out1   = self.world_model(ctx_l0, ctx_l1, ctx_l2, ctx_a)
        new_l0 = self._sample_patches(out1['logits_l0'][:, -1])   # (B, 4)

        # Pass 2: extend context with new L0 (L1/L2 zeroed), predict L1
        ext_l0 = torch.cat([ctx_l0, new_l0.unsqueeze(1)], dim=1)
        ext_l1 = torch.cat([ctx_l1, torch.zeros(B, 1, 16, dtype=torch.long, device=self.device)], dim=1)
        ext_l2 = torch.cat([ctx_l2, torch.zeros(B, 1, 16, dtype=torch.long, device=self.device)], dim=1)
        ext_a  = torch.cat([ctx_a,  torch.zeros(B, 1,     dtype=torch.long, device=self.device)], dim=1)

        ext_l0 = _trim(ext_l0)
        ext_l1 = _trim(ext_l1)
        ext_l2 = _trim(ext_l2)
        ext_a  = _trim(ext_a)

        out2   = self.world_model(ext_l0, ext_l1, ext_l2, ext_a)
        new_l1 = self._sample_patches(out2['logits_l1'][:, -1])   # (B, 16)

        # Pass 3: fill real L1, predict L2
        ext_l1[:, -1] = new_l1
        out3   = self.world_model(ext_l0, ext_l1, ext_l2, ext_a)
        new_l2 = self._sample_patches(out3['logits_l2'][:, -1])   # (B, 16)

        return new_l0, new_l1, new_l2

    def rollout(
        self,
        seed_context: dict,
        horizon:      Optional[int] = None,
    ) -> SpatialTrajectory:
        """
        Run imagination rollout seeded from real context.

        seed_context keys: tokens_l0 (B,T,4), tokens_l1 (B,T,16),
                           tokens_l2 (B,T,16), actions (B,T)
        """
        H = horizon if horizon is not None else self.max_horizon

        ctx_l0 = seed_context['tokens_l0'].to(self.device)   # (B, T_seed, 4)
        ctx_l1 = seed_context['tokens_l1'].to(self.device)
        ctx_l2 = seed_context['tokens_l2'].to(self.device)
        ctx_a  = seed_context['actions'].to(self.device)

        B = ctx_l0.size(0)

        traj_l0        = torch.zeros(B, H, 4,  dtype=torch.long,  device=self.device)
        traj_l1        = torch.zeros(B, H, 16, dtype=torch.long,  device=self.device)
        traj_l2        = torch.zeros(B, H, 16, dtype=torch.long,  device=self.device)
        traj_actions   = torch.zeros(B, H,     dtype=torch.long,  device=self.device)
        traj_log_probs = torch.zeros(B, H,     dtype=torch.float, device=self.device)
        traj_entropies = torch.zeros(B, H,     dtype=torch.float, device=self.device)
        traj_features  = []
        traj_values    = torch.zeros(B, H,     dtype=torch.float, device=self.device)
        traj_rewards   = torch.zeros(B, H,     dtype=torch.float, device=self.device)
        traj_continues = torch.zeros(B, H,     dtype=torch.float, device=self.device)

        for h in range(H):
            # Feature for current last timestep
            feature = self._extract_features(ctx_l0, ctx_l1, ctx_l2, ctx_a)   # (B, feat_dim)

            # Actor picks action
            distribution = self.actor_network(feature)
            action       = distribution.sample()
            log_probs    = distribution.log_prob(action)
            entropy      = distribution.entropy()

            # Critic, reward, continue predictions (no grad — only actor needs grad)
            with torch.no_grad():
                value          = self.critic_network(feature)
                reward         = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob  = torch.sigmoid(continue_logit)

            # Write the action we actually took into the last context position
            ctx_a = ctx_a.clone()
            ctx_a[:, -1] = action

            # 3-pass cascade to predict next tokens
            with torch.no_grad():
                new_l0, new_l1, new_l2 = self._cascade_predict_step(
                    ctx_l0, ctx_l1, ctx_l2, ctx_a
                )

            # Extend context
            ctx_l0 = torch.cat([ctx_l0, new_l0.unsqueeze(1)], dim=1)
            ctx_l1 = torch.cat([ctx_l1, new_l1.unsqueeze(1)], dim=1)
            ctx_l2 = torch.cat([ctx_l2, new_l2.unsqueeze(1)], dim=1)
            ctx_a  = torch.cat([ctx_a,  torch.zeros(B, 1, dtype=torch.long, device=self.device)], dim=1)

            traj_l0[:, h]        = new_l0
            traj_l1[:, h]        = new_l1
            traj_l2[:, h]        = new_l2
            traj_actions[:, h]   = action
            traj_log_probs[:, h] = log_probs
            traj_entropies[:, h] = entropy
            traj_features.append(feature)
            traj_values[:, h]    = value
            traj_rewards[:, h]   = reward
            traj_continues[:, h] = continue_prob

        # Bootstrap value for last state
        last_feat  = self._extract_features(ctx_l0, ctx_l1, ctx_l2, ctx_a)
        last_value = self.critic_network(last_feat)

        return SpatialTrajectory(
            tokens_l0  = traj_l0,
            tokens_l1  = traj_l1,
            tokens_l2  = traj_l2,
            actions    = traj_actions,
            log_probs  = traj_log_probs,
            feats      = torch.stack(traj_features, dim=1),
            values     = traj_values,
            rewards    = traj_rewards,
            continues  = traj_continues,
            last_value = last_value,
            entropies  = traj_entropies,
            last_feat  = last_feat,
        )
