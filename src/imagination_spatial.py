"""
OPTIMIZED SPATIAL IMAGINATION ROLLOUT FOR POLICY TRAINING

DROP-IN REPLACEMENT for imagination_spatial.py with three speed fixes:

FIX 1: AMP (float16) on all frozen WM forwards
  - RTX 5060 Ti: ~2x throughput for matmul-heavy transformer forwards
  - Safe because WM is frozen (no gradient scaling needed)

FIX 2: Reduced MAX_CONTEXT_T (15 → 10)
  - Attention: 370² vs 555² = 2.25x reduction
  - Pong policy doesn't need 15 timesteps of context for early training
  - Configurable via constructor arg

FIX 3: torch.compile on frozen WM forward (optional, ~10-30% kernel fusion)

COMBINED SPEEDUP: ~3-5x over original imagination_spatial.py

WHAT WOULD GIVE 10-20x MORE (future work):
  - Implement forward_incremental() for _SpatialTransformerBlock
  - Mirror the KV-cache cascade from imagination.py (_do_cached_cascade)
  - Each cascade pass becomes O(37) instead of O(370) = 10x per pass
  - The original (non-spatial) WM already has this; spatial just needs the port

WHY WE CAN'T MERGE _extract_features INTO CASCADE PASS 1:
  Features (from L0/L1/L2 hidden states) don't depend on the action at
  position 36 (causal mask blocks it). But the actor needs features to
  SELECT the action, and the cascade needs the action to PREDICT L0.
  Merging would require KV-cache to re-process only position 36.
"""

import torch
import torch.nn.functional as F
from torch import autocast
from dataclasses import dataclass
from typing import Optional

from world_model_spatial import SpatialHierarchicalWorldModel

MAX_CONTEXT_T = 10  # DOWN FROM 15: 370 vs 555 positions


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
        use_amp:     bool  = True,
        max_context_t: int = MAX_CONTEXT_T,
        compile_wm:  bool  = False,
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
        self.use_amp          = use_amp and (self.device.type == 'cuda')
        self.max_context_t    = max_context_t

        # OPTIONAL: torch.compile for kernel fusion on frozen WM
        if compile_wm and hasattr(torch, 'compile'):
            self._wm_forward = torch.compile(self.world_model, mode='reduce-overhead')
            print("  [ImagineRollout] WM compiled with torch.compile (reduce-overhead)")
        else:
            self._wm_forward = self.world_model

    def _wm_call(self, ctx_l0, ctx_l1, ctx_l2, ctx_a):
        """Unified WM forward with AMP wrapping."""
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            return self._wm_forward(ctx_l0, ctx_l1, ctx_l2, ctx_a)

    @torch.no_grad()
    def _sample_patches(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample independently from each patch position."""
        B, P, V = logits.shape
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        flat  = probs.reshape(B * P, V)
        samp  = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return samp.reshape(B, P)

    def _trim(self, t):
        """Trim to max_context_t timesteps."""
        return t[:, -self.max_context_t:] if t.size(1) > self.max_context_t else t

    def _extract_features(
        self,
        context_l0:      torch.Tensor,
        context_l1:      torch.Tensor,
        context_l2:      torch.Tensor,
        context_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen WM, pool per level, pass through feature extractor."""
        ctx_l0 = self._trim(context_l0)
        ctx_l1 = self._trim(context_l1)
        ctx_l2 = self._trim(context_l2)
        ctx_a  = self._trim(context_actions)

        out = self._wm_call(ctx_l0, ctx_l1, ctx_l2, ctx_a)
        x   = out['hidden']            # (B, T*37, D)
        B   = x.size(0)
        T   = ctx_l0.size(1)
        D   = x.size(-1)

        x_ts    = x.reshape(B, T, 37, D)
        mean_l0 = x_ts[:, -1, 0:4,   :].mean(dim=1)   # (B, D)
        mean_l1 = x_ts[:, -1, 4:20,  :].mean(dim=1)   # (B, D)
        mean_l2 = x_ts[:, -1, 20:36, :].mean(dim=1)   # (B, D)

        hidden_3 = torch.stack([mean_l0, mean_l1, mean_l2], dim=1)  # (B, 3, D)
        return self.feature_extractor(hidden_3)                      # (B, feat_dim)

    @torch.no_grad()
    def _cascade_predict_step(
        self,
        context_l0:      torch.Tensor,
        context_l1:      torch.Tensor,
        context_l2:      torch.Tensor,
        context_actions: torch.Tensor,
    ):
        """3-pass cascade to generate one new (l0, l1, l2) timestep."""
        B = context_l0.size(0)

        ctx_l0 = self._trim(context_l0)
        ctx_l1 = self._trim(context_l1)
        ctx_l2 = self._trim(context_l2)
        ctx_a  = self._trim(context_actions)

        # Pass 1: predict L0
        out1   = self._wm_call(ctx_l0, ctx_l1, ctx_l2, ctx_a)
        new_l0 = self._sample_patches(out1['logits_l0'][:, -1])   # (B, 4)

        # Pass 2: extend context with new L0 (L1/L2 zeroed), predict L1
        ext_l0 = self._trim(torch.cat([ctx_l0, new_l0.unsqueeze(1)], dim=1))
        ext_l1 = self._trim(torch.cat([ctx_l1, torch.zeros(B, 1, 16, dtype=torch.long, device=self.device)], dim=1))
        ext_l2 = self._trim(torch.cat([ctx_l2, torch.zeros(B, 1, 16, dtype=torch.long, device=self.device)], dim=1))
        ext_a  = self._trim(torch.cat([ctx_a,  torch.zeros(B, 1,     dtype=torch.long, device=self.device)], dim=1))

        out2   = self._wm_call(ext_l0, ext_l1, ext_l2, ext_a)
        new_l1 = self._sample_patches(out2['logits_l1'][:, -1])   # (B, 16)

        # Pass 3: fill real L1, predict L2
        ext_l1[:, -1] = new_l1
        out3   = self._wm_call(ext_l0, ext_l1, ext_l2, ext_a)
        new_l2 = self._sample_patches(out3['logits_l2'][:, -1])   # (B, 16)

        return new_l0, new_l1, new_l2

    def rollout(
        self,
        seed_context: dict,
        horizon:      Optional[int] = None,
    ) -> SpatialTrajectory:
        """Run imagination rollout seeded from real context."""
        H = horizon if horizon is not None else self.max_horizon

        ctx_l0 = seed_context['tokens_l0'].to(self.device)
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
            # Feature extraction (1 WM forward)
            feature = self._extract_features(ctx_l0, ctx_l1, ctx_l2, ctx_a)

            # Actor picks action
            distribution = self.actor_network(feature)
            action       = distribution.sample()
            log_probs    = distribution.log_prob(action)
            entropy      = distribution.entropy()

            # Critic, reward, continue predictions
            with torch.no_grad():
                value          = self.critic_network(feature)
                reward         = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob  = torch.sigmoid(continue_logit)

            # Write action, then cascade (3 WM forwards)
            ctx_a = ctx_a.clone()
            ctx_a[:, -1] = action

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

        # Bootstrap value
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
