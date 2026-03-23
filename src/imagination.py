"""
IMAGINATION ROLLOUT FOR POLICY TRAINING
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from world_model import HierarchicalWorldModel

@dataclass
class Trajectory:
    """ CONTAINER FOR ONE IMAGINATION ROLLOUT BATCH """
    tokens : torch.tensor           # (B, H, 3)         - PREDICTED HRVQ TOKENS
    actions : torch.tensor          # (B, H)            - ACTIONS TAKEN BY POLICY
    log_probs : torch.tensor        # (B, H)            - LOG PROBS OF ACTIONS
    feats : torch.tensor            # (B, H, feat_dim)  - DENSE FEATURES AT EACH STEP
    values : torch.tensor           # (B, H)            - CRITIC PREDICTIONS
    rewards : torch.tensor          # (B, H)            - REWARD PREDICTIONS
    continues : torch.tensor        # (B, H)            - CONTINUE LOGITS
    last_value : torch.tensor       # (B,)              - BOOTSTRAP VALUE FOR LAST STATE
    entropies : torch.tensor        # (B, H)            - POLICY ENTROPY AT EACH STEP
    last_feat : torch.tensor        # (B, feat_dim)     - FEATURE OF LAST STATE


class ImagineRollout:
    """ RUN HORIZON-STEP IMAGINATION INSIDE FROZEN WORLD MODEL """

    def __init__(
        self,
        world_model : HierarchicalWorldModel,
        feature_extractor,
        actor_network,
        critic_network,
        reward_network,
        continue_network,
        max_horizon : int = 30,
        temperature : float = 1.0,
        device : torch.device = None,
        use_kv_cache : bool = True,
    ):
        self.world_model = world_model
        self.feature_extractor = feature_extractor
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.reward_network = reward_network
        self.continue_network = continue_network
        self.max_horizon = max_horizon
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_kv_cache = use_kv_cache


    @torch.no_grad()
    def _sample_token(
        self,
        logits : torch.tensor,
    ) -> torch.tensor:
        """ SAMPLE FROM CATEGORICAL DISTRIBUTION OVER CODEBOOK ENTRIES """

        if self.temperature <= 0:
            return logits.argmax(dim = -1)

        scaled_logits = logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        distribution = torch.multinomial(input = probabilities, num_samples = 1).squeeze(-1)

        return distribution


    # ORIGINAL CASCADE (NO CACHE) - KEPT FOR VERIFICATION
    @torch.no_grad()
    def _cascade_predict_next(
        self,
        tokens_context : torch.tensor,
        actions_context : torch.tensor,
    ) -> torch.tensor:
        """ CASCADE PREDICTION - THREE FULL FORWARD PASSES, NO CACHE """

        batch_items = tokens_context.size(0)

        # 1. PREDICT L0
        logits_l0, _, _ = self.world_model(tokens_context, actions_context)
        token_l0 = self._sample_token(logits = logits_l0[:, -1, :])

        # 2. PREDICT L1 - EXTEND CONTEXT WITH [L0_NEW, 0, 0]
        next_partial = torch.zeros(batch_items, 1, 3, dtype = torch.long, device = self.device)
        next_partial[:, 0, 0] = token_l0
        dummy_action = actions_context[:, -1:]

        tokens_context_extended = torch.cat(tensors = [tokens_context, next_partial], dim = 1)
        actions_context_extended = torch.cat(tensors = [actions_context, dummy_action], dim = 1)

        _, logits_l1, _ = self.world_model(tokens_context_extended, actions_context_extended)
        token_l1 = self._sample_token(logits = logits_l1[:, -1, :])

        # 3. PREDICT L2 - FILL IN L1 IN EXTENDED CONTEXT
        tokens_context_extended[:, -1, 1] = token_l1
        _, _, logits_l2 = self.world_model(tokens_context_extended, actions_context_extended)
        token_l2 = self._sample_token(logits = logits_l2[:, -1, :])

        predicted_tokens = torch.stack(tensors = [token_l0, token_l1, token_l2], dim = -1)
        return predicted_tokens


    # KV-CACHED CASCADE
    @torch.no_grad()
    def _do_cached_cascade(
        self,
        kv_cache : list,
        cached_seq_len : int,
        action : torch.Tensor,
        prev_tokens : torch.Tensor,
        prev_action_placeholder : bool,
    ) -> Tuple[torch.Tensor, list, int]:
        """ CASCADE PREDICTION USING INCREMENTAL FORWARDS WITH KV-CACHE """
        B = action.size(0)

        # STEP 0: POP LAST 4 POSITIONS, RE-PROCESS WITH CORRECT ACTION
        kv_cache_reverted = [
            (K[:, :, :-4, :], V[:, :, :-4, :]) for (K, V) in kv_cache
        ]
        revert_len = cached_seq_len - 4

        x_step0, kv_cache_step0 = self.world_model.forward_incremental(
            prev_tokens,
            action.unsqueeze(1),
            kv_cache_reverted,
            revert_len,
        )
        cached_len_after_step0 = revert_len + 4

        # STEP 1: L0 FROM ACTION POSITION
        logits_l0 = self.world_model.headl0(x_step0[:, 3, :])  # (B, 256)
        token_l0 = self._sample_token(logits_l0)

        # STEP 2: L1 FROM L0 POSITION OF NEW TIMESTEP
        tokens_pass2 = torch.zeros(B, 1, 3, dtype=torch.long, device=self.device)
        tokens_pass2[:, 0, 0] = token_l0
        actions_pass2 = torch.zeros(B, 1, dtype=torch.long, device=self.device)

        x_pass2, _ = self.world_model.forward_incremental(
            tokens_pass2, actions_pass2, kv_cache_step0, cached_len_after_step0
        )
        logits_l1 = self.world_model.headl1(x_pass2[:, 0, :])
        token_l1 = self._sample_token(logits_l1)

        # STEP 3: L2 FROM L1 POSITION OF NEW TIMESTEP
        tokens_pass3 = torch.zeros(B, 1, 3, dtype=torch.long, device=self.device)
        tokens_pass3[:, 0, 0] = token_l0
        tokens_pass3[:, 0, 1] = token_l1
        actions_pass3 = torch.zeros(B, 1, dtype=torch.long, device=self.device)

        x_pass3, kv_cache_final = self.world_model.forward_incremental(
            tokens_pass3, actions_pass3, kv_cache_step0, cached_len_after_step0
        )
        logits_l2 = self.world_model.headl2(x_pass3[:, 1, :])
        token_l2 = self._sample_token(logits_l2)

        # STEP 4: ASSEMBLE
        predicted_tokens = torch.stack([token_l0, token_l1, token_l2], dim=-1)

        # CACHE NOW HAS [L0, L1, 0, 0] FOR THE NEW TIMESTEP
        new_cached_len = cached_len_after_step0 + 4

        next_hidden = x_pass3[:, :3, :]  # (B, 3, d_model)

        return predicted_tokens, kv_cache_final, new_cached_len, next_hidden


    # ROLLOUT ENTRY POINT
    def rollout(
        self,
        seed_tokens : torch.tensor,
        seed_actions : torch.tensor,
        horizon : Optional[int] = None,
    ) -> Trajectory:
        """ RUN IMAGINATION ROLLOUT - DISPATCHES TO CACHED OR ORIGINAL PATH """
        if self.use_kv_cache:
            return self._rollout_cached(seed_tokens, seed_actions, horizon)
        else:
            return self._rollout_original(seed_tokens, seed_actions, horizon)

    def _rollout_original(
        self,
        seed_tokens : torch.tensor,
        seed_actions : torch.tensor,
        horizon : Optional[int] = None,
    ) -> Trajectory:
        """ ORIGINAL ROLLOUT - THREE FULL FORWARD PASSES PER STEP """

        batch_size = seed_tokens.size(0)
        t_seed = seed_tokens.size(1)
        H = horizon if horizon is not None else self.max_horizon

        max_timesteps = self.world_model.config.max_seq_len // 4
        assert t_seed + H <= max_timesteps

        tokens_context = seed_tokens.clone()
        actions_context = seed_actions.clone()

        # ALLOCATE STORAGE FOR ROLLOUT TRAJECTORY
        traj_tokens     = torch.zeros(batch_size, H, 3, dtype=torch.long, device=self.device)
        traj_actions    = torch.zeros(batch_size, H, dtype=torch.long, device=self.device)
        traj_log_probs  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_entropies  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_features   = []
        traj_values     = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_rewards    = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_continues  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)

        for h in range(H):

            # FULL FORWARD TO GET HIDDEN STATES
            x = self.world_model.embedding(
                tokens_context, actions_context
            )

            mask = self.world_model._get_mask(
                x.size(1),
                x.device
            )

            for block in self.world_model.blocks:
                x = block(x, mask = mask)

            x = self.world_model.ln_final(x)

            # EXTRACT FEATURE FOR CURRENT STATE
            is_visual = getattr(self.feature_extractor, 'is_visual_mode', False)
            if is_visual:
                current_tok = tokens_context[:, -1, :]           # (B, 3)
                feature = self.feature_extractor(current_tok)
            else:
                t_current  = tokens_context.size(1)
                last_start = (t_current - 1) * 4
                current_hidden = x[:, last_start:last_start + 3, :]  # (B, 3, d_model)
                feature = self.feature_extractor(current_hidden)

            # ACTOR PICKS ACTION FROM FEATURE
            distribution = self.actor_network(feature)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)
            entropy = distribution.entropy()

            # CRITIC, REWARD, CONTINUE PREDICTIONS
            with torch.no_grad():
                value = self.critic_network(feature)
                reward = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob = torch.sigmoid(continue_logit)

            actions_context[:, -1] = action
            next_tokens = self._cascade_predict_next(tokens_context, actions_context)

            tokens_context = torch.cat([tokens_context, next_tokens.unsqueeze(1)], dim=1)
            actions_context = torch.cat([
                actions_context,
                torch.zeros(batch_size, 1, dtype = torch.long, device = self.device)
            ], dim = 1)

            # STORE TRAJECTORY STEP
            traj_tokens[:, h] = next_tokens
            traj_actions[:, h] = action
            traj_log_probs[:, h] = log_probs
            traj_entropies[:, h] = entropy
            traj_features.append(feature)
            traj_values[:, h] = value
            traj_rewards[:, h] = reward
            traj_continues[:, h] = continue_prob

        # BOOTSTRAP VALUES
        is_visual_orig = getattr(self.feature_extractor, 'is_visual_mode', False)
        if is_visual_orig:
            final_feature = self.feature_extractor(tokens_context[:, -1, :])
        else:
            x = self.world_model.embedding(tokens_context, actions_context)
            mask = self.world_model._get_mask(x.size(1), x.device)
            for block in self.world_model.blocks:
                x = block(x, mask=mask)
            x = self.world_model.ln_final(x)
            t_final    = tokens_context.size(1)
            last_start = (t_final - 1) * 4
            final_hidden  = x[:, last_start:last_start + 3, :]
            final_feature = self.feature_extractor(final_hidden)
        last_value = self.critic_network(final_feature)

        # RETURN TRAJECTORY
        return Trajectory(
            tokens=traj_tokens, actions=traj_actions,
            log_probs=traj_log_probs, feats=torch.stack(traj_features, dim=1),
            values=traj_values, rewards=traj_rewards,
            continues=traj_continues, last_value=last_value,
            entropies=traj_entropies,
            last_feat=final_feature,
        )

    def _rollout_cached(
        self,
        seed_tokens : torch.Tensor,
        seed_actions : torch.Tensor,
        horizon : Optional[int] = None,
    ) -> Trajectory:
        """ KV-CACHED ROLLOUT WITH TRANSFORMER HIDDEN STATE FEATURES """

        batch_size = seed_tokens.size(0)
        t_seed = seed_tokens.size(1)
        H = horizon if horizon is not None else self.max_horizon

        max_timesteps = self.world_model.config.max_seq_len // 4
        assert t_seed + H <= max_timesteps

        # PHASE 1: PRIME KV CACHE WITH SEED CONTEXT
        _, _, _, kv_cache, x_full = self.world_model.forward_with_kv(
            seed_tokens, seed_actions
        )

        cached_seq_len = t_seed * 4

        is_visual = getattr(self.feature_extractor, 'is_visual_mode', False)

        if is_visual:
            current_tokens = seed_tokens[:, -1, :]    # (B, 3)
            current_hidden = None
        else:
            last_t_start   = (t_seed - 1) * 4
            current_hidden = x_full[:, last_t_start:last_t_start + 3, :]  # (B, 3, d_model)
            current_tokens = None

        next_hidden = x_full[:, -3:, :]  # KEPT FOR COMPATIBILITY

        # ALLOCATE STORAGE FOR ROLLOUT TRAJECTORY
        traj_tokens     = torch.zeros(batch_size, H, 3, dtype=torch.long, device=self.device)
        traj_actions    = torch.zeros(batch_size, H, dtype=torch.long, device=self.device)
        traj_log_probs  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_entropies  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_features   = []
        traj_values     = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_rewards    = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)
        traj_continues  = torch.zeros(batch_size, H, dtype=torch.float, device=self.device)

        last_cached_tokens = seed_tokens[:, -1:, :]  # (B, 1, 3)

        # PHASE 2: IMAGINATION ROLLOUT
        for h in range(H):

            # FEATURE EXTRACTION
            if is_visual:
                feature = self.feature_extractor(current_tokens)  # (B, feat_dim)
            else:
                feature = self.feature_extractor(current_hidden)  # (B, feat_dim)

            # ACTOR PICKS ACTION FROM FEATURE
            distribution = self.actor_network(feature)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)
            entropy = distribution.entropy()

            # CRITIC, REWARD, CONTINUE PREDICTIONS
            with torch.no_grad():
                value = self.critic_network(feature)
                reward = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob = torch.sigmoid(continue_logit)

            # CASCADE PREDICTION WITH CACHE
            with torch.no_grad():
                predicted_tokens, kv_cache, cached_seq_len, next_hidden = \
                    self._do_cached_cascade(
                        kv_cache = kv_cache,
                        cached_seq_len = cached_seq_len,
                        action = action,
                        prev_tokens = last_cached_tokens,
                        prev_action_placeholder = (h > 0),
                    )

            # UPDATE STEP FOR NEXT ITERATION
            last_cached_tokens = predicted_tokens.unsqueeze(1)  # (B, 1, 3)
            if is_visual:
                current_tokens = predicted_tokens     # (B, 3)
            else:
                current_hidden = next_hidden          # (B, 3, d_model)

            # STORE TRAJECTORY STEP
            traj_tokens[:, h] = predicted_tokens
            traj_actions[:, h] = action
            traj_log_probs[:, h] = log_probs
            traj_entropies[:, h] = entropy
            traj_features.append(feature)
            traj_values[:, h] = value
            traj_rewards[:, h] = reward
            traj_continues[:, h] = continue_prob

        # BOOTSTRAP VALUES
        if is_visual:
            final_feature = self.feature_extractor(current_tokens)
        else:
            final_feature = self.feature_extractor(current_hidden)
        last_value = self.critic_network(final_feature)

        return Trajectory(
            tokens=traj_tokens, actions=traj_actions,
            log_probs=traj_log_probs, feats=torch.stack(traj_features, dim=1),
            values=traj_values, rewards=traj_rewards,
            continues=traj_continues, last_value=last_value,
            entropies=traj_entropies,
            last_feat=final_feature,
        )
