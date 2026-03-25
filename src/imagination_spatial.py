"""
KV-CACHED SPATIAL IMAGINATION ROLLOUT FOR POLICY TRAINING

Replaces the 4-full-forward-per-step approach with incremental KV-cache:
  BEFORE: 4 full 370-pos forwards per step x 15 steps = 60 x 370^2 attn ops
  AFTER:  1 full forward for seed + 3 incremental 37-pos per step = ~10x speedup

The 3-pass cascade is preserved:
  Pass 1: re-embed previous timestep with real action, predict L0
  Pass 2: extend with real L0 + zeroed L1/L2, predict L1 (then revert)
  Pass 3: extend with real L0 + real L1 + zeroed L2, predict L2
"""

import torch
import torch.nn.functional as F
from torch import autocast
from dataclasses import dataclass
from typing import Optional, Tuple

from world_model_spatial import (
    SpatialHierarchicalWorldModel,
    TOKENS_PER_TIMESTEP, NUM_L0, NUM_L1, NUM_L2,
    L0_START, L0_END, L1_START, L1_END, L2_START, L2_END, ACTION_POS,
)

MAX_CONTEXT_T = 5


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
    """KV-cached imagination inside frozen SpatialHierarchicalWorldModel."""

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

    @torch.no_grad()
    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution. logits: (B, V) -> (B,)"""
        if self.temperature <= 0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def _extract_features_from_hidden(self, x_ts: torch.Tensor) -> torch.Tensor:
        """Pool per-level hidden states and pass through feature extractor.
        x_ts: (B, 37, D) hidden states for one timestep.
        """
        mean_l0 = x_ts[:, L0_START:L0_END, :].mean(dim=1)
        mean_l1 = x_ts[:, L1_START:L1_END, :].mean(dim=1)
        mean_l2 = x_ts[:, L2_START:L2_END, :].mean(dim=1)
        hidden_3 = torch.stack([mean_l0, mean_l1, mean_l2], dim=1)  # (B, 3, D)
        return self.feature_extractor(hidden_3)

    @torch.no_grad()
    def _do_cached_cascade(
        self,
        kv_cache: list,
        cached_seq_len: int,
        action: torch.Tensor,
        prev_l0: torch.Tensor,
        prev_l1: torch.Tensor,
        prev_l2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, int, torch.Tensor]:
        """KV-cached cascade: pop-and-redo 3 passes to predict L0, L1, L2."""
        B = action.size(0)
        P = TOKENS_PER_TIMESTEP

        # REVERT: pop last 37 positions (had placeholder action)
        kv_reverted = [
            (K[:, :, :-P, :], V[:, :, :-P, :]) for (K, V) in kv_cache
        ]
        revert_len = cached_seq_len - P

        # PASS 1: re-embed previous timestep with real action, predict L0
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            x_step0, kv_step0, len_step0 = self.world_model.forward_incremental(
                prev_l0, prev_l1, prev_l2, action.unsqueeze(1),
                kv_reverted, revert_len,
            )

        logits_l0 = self.world_model.head_l0(x_step0[:, ACTION_POS, :])
        new_l0_token = self._sample_token(logits_l0)
        new_l0 = new_l0_token.unsqueeze(1).expand(B, NUM_L0)

        # PASS 2: extend with real L0, zeroed L1/L2, predict L1
        new_l0_ts = new_l0.unsqueeze(1)
        zero_l1 = torch.zeros(B, 1, NUM_L1, dtype=torch.long, device=self.device)
        zero_l2 = torch.zeros(B, 1, NUM_L2, dtype=torch.long, device=self.device)
        zero_a  = torch.zeros(B, 1, dtype=torch.long, device=self.device)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            x_step2, _, _ = self.world_model.forward_incremental(
                new_l0_ts, zero_l1, zero_l2, zero_a,
                kv_step0, len_step0,
            )

        logits_l1 = self.world_model.head_l1(x_step2[:, L0_END - 1, :])
        new_l1_token = self._sample_token(logits_l1)
        new_l1 = new_l1_token.unsqueeze(1).expand(B, NUM_L1)

        # PASS 3: revert to step0, extend with real L0 + real L1, predict L2
        real_l1 = new_l1.unsqueeze(1)

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            x_step3, kv_step3, len_step3 = self.world_model.forward_incremental(
                new_l0_ts, real_l1, zero_l2, zero_a,
                kv_step0, len_step0,
            )

        logits_l2 = self.world_model.head_l2(x_step3[:, L1_END - 1, :])
        new_l2_token = self._sample_token(logits_l2)
        new_l2 = new_l2_token.unsqueeze(1).expand(B, NUM_L2)

        return new_l0, new_l1, new_l2, kv_step3, len_step3, x_step3

    def rollout(
        self,
        seed_context: dict,
        horizon:      Optional[int] = None,
    ) -> SpatialTrajectory:
        """KV-cached imagination rollout."""
        H = horizon if horizon is not None else self.max_horizon

        # TRIM SEED TO MAX_CONTEXT_T
        ctx_l0 = seed_context['tokens_l0'].to(self.device)[:, -self.max_context_t:]
        ctx_l1 = seed_context['tokens_l1'].to(self.device)[:, -self.max_context_t:]
        ctx_l2 = seed_context['tokens_l2'].to(self.device)[:, -self.max_context_t:]
        ctx_a  = seed_context['actions'].to(self.device)[:, -self.max_context_t:]

        B = ctx_l0.size(0)
        T_seed = ctx_l0.size(1)

        # PHASE 1: PRIME KV-CACHE WITH SEED CONTEXT
        with torch.no_grad():
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                hidden_full, kv_cache = self.world_model.get_context_hidden(
                    ctx_l0, ctx_l1, ctx_l2, ctx_a
                )
        cached_seq_len = T_seed * TOKENS_PER_TIMESTEP

        last_ts_hidden = hidden_full[:, -TOKENS_PER_TIMESTEP:, :]

        # ALLOCATE TRAJECTORY STORAGE
        traj_l0        = torch.zeros(B, H, NUM_L0, dtype=torch.long,  device=self.device)
        traj_l1        = torch.zeros(B, H, NUM_L1, dtype=torch.long,  device=self.device)
        traj_l2        = torch.zeros(B, H, NUM_L2, dtype=torch.long,  device=self.device)
        traj_actions   = torch.zeros(B, H,         dtype=torch.long,  device=self.device)
        traj_log_probs = torch.zeros(B, H,         dtype=torch.float, device=self.device)
        traj_entropies = torch.zeros(B, H,         dtype=torch.float, device=self.device)
        traj_features  = []
        traj_values    = torch.zeros(B, H,         dtype=torch.float, device=self.device)
        traj_rewards   = torch.zeros(B, H,         dtype=torch.float, device=self.device)
        traj_continues = torch.zeros(B, H,         dtype=torch.float, device=self.device)

        prev_l0 = ctx_l0[:, -1:]
        prev_l1 = ctx_l1[:, -1:]
        prev_l2 = ctx_l2[:, -1:]

        # PHASE 2: IMAGINATION LOOP
        for h in range(H):
            feature = self._extract_features_from_hidden(last_ts_hidden)

            distribution = self.actor_network(feature)
            action    = distribution.sample()
            log_probs = distribution.log_prob(action)
            entropy   = distribution.entropy()

            with torch.no_grad():
                value          = self.critic_network(feature)
                reward         = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob  = torch.sigmoid(continue_logit)

            new_l0, new_l1, new_l2, kv_cache, cached_seq_len, last_ts_hidden = \
                self._do_cached_cascade(
                    kv_cache, cached_seq_len, action,
                    prev_l0, prev_l1, prev_l2,
                )

            prev_l0 = new_l0.unsqueeze(1)
            prev_l1 = new_l1.unsqueeze(1)
            prev_l2 = new_l2.unsqueeze(1)

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

        # BOOTSTRAP VALUE
        last_feat  = self._extract_features_from_hidden(last_ts_hidden)
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
