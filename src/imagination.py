""" 
IMAGINATION ROLLOUT FOR POLICY TRAINING

References:
- IRIS (Micheli et al., ICLR 2023) — discrete token autoregressive rollout
- TWISTER (Burchi & Timofte, ICLR 2025) — imagination in transformer WM
- Dreamer 4 (Hafner et al., Sep 2025) — block-causal imagination

KV-Cache Optimization:
- Standard GPT inference caching adapted for hierarchical causal masking
- 3-5x speedup on imagination rollouts by avoiding redundant attention computation
- Original non-cached path retained for verification and WM training
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from world_model import HierarchicalWorldModel

@dataclass
class Trajectory:
    """ CONTAINER for 1 x imagination rollout batch."""
    tokens : torch.tensor           # (B, H, 3)         - PREDICTED HRVQ TOKENS (at each step)
    actions : torch.tensor          # (B, H)            - ACTIONS TAKEN BY POLICY
    log_probs : torch.tensor        # (B, H)            - LOG PROBS OF ACTIONS UNDER POLICY
    feats : torch.tensor            # (B, H, feat_dim)  - DENSE FEATURES AT EACH STEP
    values : torch.tensor           # (B, H)            - CRITIC PREDICTIONS AT EACH STEP
    rewards : torch.tensor          # (B, H)            - REWARD PREDICTIONS AT EACH STEP
    continues : torch.tensor        # (B, H)            - CONTINUE LOGITS AT EACH STEP
    last_value : torch.tensor       # (B,)              - VALUE PREDICTION FOR LAST STATE TO BOOTSTRAP FROM
    entropies : torch.tensor        # (B, H)            - ENTROPY OF POLICY AT EACH STEP


class ImagineRollout:
    """ Run Horizon-Step imagination inside FROZEN world model. 
    
    Rollout():
    1. CASCADE PREDICTION of L0 -> L1 -> L2 tokens at each step.
    2. Extract FEATURES from predicted tokens, pass through ACTOR to get ACTIONS and LOG PROBS.
    3. Return Trajectory (dataclass) for actor-critic training.
    
    KV-CACHE MODE (default, use_kv_cache=True):
    - Seed context processed once via forward_with_kv()  -> caches K/V for all layers
    - Each horizon step: 3 incremental forwards (4 new positions each) instead of 3 full O(n²) passes
    - Net: 3 x O(n) per step instead of 3 x O(n²) per step (previous)
    """
    
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
        logits : torch.tensor,   # (B, num_tokens)
    ) -> torch.tensor:           # (B,) sampled token indices
        """ Sample from Categorical Distribution over Codebook entries. """
        
        if self.temperature <= 0:
            return logits.argmax(dim = -1)
        
        scaled_logits = logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        distribution = torch.multinomial(input = probabilities, num_samples = 1).squeeze(-1)  
        
        return distribution


    
    # ORIGINAL CASCADE (NO CACHE) — kept for verification
    @torch.no_grad()
    def _cascade_predict_next(
        self,
        tokens_context : torch.tensor,   # (B, timesteps, 3)
        actions_context : torch.tensor,   # (B, timesteps)
    ) -> torch.tensor:                   # (B, 3)
        """ CASCADE PREDICTION (ORIGINAL — 3 full forward passes, no cache).
        
        1. PREDICT L0 from Action Context (full forward)
        2. PREDICT L1 from L0 (full forward over extended context)
        3. PREDICT L2 from L1 (full forward over extended context)
        
        Kept for verification against cached version. Used when use_kv_cache=False.
        """
        
        batch_items = tokens_context.size(0)
        
        """ 1. PREDICT L0 """
        logits_l0, _, _ = self.world_model(tokens_context, actions_context)
        token_l0 = self._sample_token(logits = logits_l0[:, -1, :])  
        
        """ 2. PREDICT L1 — extend context with [L0_new, 0, 0] """
        next_partial = torch.zeros(batch_items, 1, 3, dtype = torch.long, device = self.device) 
        next_partial[:, 0, 0] = token_l0 
        dummy_action = actions_context[:, -1:] 
        
        tokens_context_extended = torch.cat(tensors = [tokens_context, next_partial], dim = 1)
        actions_context_extended = torch.cat(tensors = [actions_context, dummy_action], dim = 1)
        
        _, logits_l1, _ = self.world_model(tokens_context_extended, actions_context_extended)
        token_l1 = self._sample_token(logits = logits_l1[:, -1, :])  
        
        """ 3. PREDICT L2 — fill in L1 in extended context """
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
        action : torch.Tensor,            # (B,) selected action
        prev_tokens : torch.Tensor,        # (B, 1, 3) tokens of timestep whose action we just chose
        prev_action_placeholder : bool,    # True if last cached timestep had action=0
    ) -> Tuple[torch.Tensor, list, int]:
        """Cascade prediction using incremental forwards.
        
        Mirrors _cascade_predict_next but each forward pass processes only 
        4 new positions against cached K/V, giving O(n) instead of O(n²).
        
        WHY 3 passes not 1:
        headl1 reads L0 hidden state → needs correct L0 embedding
        headl2 reads L1 hidden state → needs correct L1 embedding
        So we progressively fill tokens and re-run the 4-position incremental.
        
        WHY pop+redo:
        The original cascade writes the actor's action into context THEN does 
        a full forward. Our cache was built BEFORE the action was known, so 
        the last 4 cached positions have a stale action embedding. We pop them 
        and re-process with the real action.
        
        Returns:
            predicted_tokens: (B, 3)
            updated_cache:    KV cache extended by new timestep
            new_cached_len:   total positions now in cache
        """
        B = action.size(0)
        
        #  STEP 0: Pop last 4 positions, re-process with correct action 
        kv_cache_reverted = [
            (K[:, :, :-4, :], V[:, :, :-4, :]) for (K, V) in kv_cache
        ]
        revert_len = cached_seq_len - 4
        
        x_step0, kv_cache_step0 = self.world_model.forward_incremental(
            prev_tokens,                      # (B, 1, 3) 
            action.unsqueeze(1),              # (B, 1) real action
            kv_cache_reverted, 
            revert_len,
        )
        cached_len_after_step0 = revert_len + 4
        
        #  STEP 1: L0 from action position 
        logits_l0 = self.world_model.headl0(x_step0[:, 3, :])  # (B, 256)
        token_l0 = self._sample_token(logits_l0)
        
        #  STEP 2: L1 from L0 position of new timestep 
        tokens_pass2 = torch.zeros(B, 1, 3, dtype=torch.long, device=self.device)
        tokens_pass2[:, 0, 0] = token_l0
        actions_pass2 = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        
        x_pass2, _ = self.world_model.forward_incremental(
            tokens_pass2, actions_pass2, kv_cache_step0, cached_len_after_step0
        )
        logits_l1 = self.world_model.headl1(x_pass2[:, 0, :])
        token_l1 = self._sample_token(logits_l1)
        
        #  STEP 3: L2 from L1 position of new timestep 
        # Revert to step0 cache (discard pass2's 4 positions)
        tokens_pass3 = torch.zeros(B, 1, 3, dtype=torch.long, device=self.device)
        tokens_pass3[:, 0, 0] = token_l0
        tokens_pass3[:, 0, 1] = token_l1
        actions_pass3 = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        
        x_pass3, kv_cache_final = self.world_model.forward_incremental(
            tokens_pass3, actions_pass3, kv_cache_step0, cached_len_after_step0
        )
        logits_l2 = self.world_model.headl2(x_pass3[:, 1, :])
        token_l2 = self._sample_token(logits_l2)
        
        #  STEP 4: ASSEMBLE
        predicted_tokens = torch.stack([token_l0, token_l1, token_l2], dim=-1)
        
        # Cache now has [L0, L1, 0, 0] for the new timestep.
        # L2 placeholder is invisible (hierarchical mask blocks future from seeing past L2).
        # Action=0 placeholder will be popped and re-done at next horizon step.
        new_cached_len = cached_len_after_step0 + 4
        
        next_hidden = x_pass3[:, :3, :] # (B, 3, d_model) - L0, L1, L2 POSITIONS for new timestep
        
        return predicted_tokens, kv_cache_final, new_cached_len, next_hidden


    # ROLLOUT ENTRY POINT
    def rollout(
        self,
        seed_tokens : torch.tensor,     # (B, T_seed, 3)
        seed_actions : torch.tensor,    # (B, T_seed)
        horizon : Optional[int] = None,
    ) -> Trajectory:
        """ RUN IMAGINATION ROLLOUT. Dispatches to cached or original path. """
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
        """ORIGINAL rollout — 3 full forward passes per step. Kept for verification."""
        
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
            
            # EXTRACT HIDDEN STATE for LAST timestep's L0, L1 and L2 positions
            
            t_current = tokens_context.size(1)
            last_start = (t_current - 1) * 4
            current_hidden = x[:, last_start:last_start+3, :]  # (B, 3, d_model)
            
            feature = self.feature_extractor(current_hidden)
            
            # ACTOR PICKS ACTION  - from feature
            distribution = self.actor_network(feature)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)
            entropy = distribution.entropy()
            
            # CRITIC, REWARD, CONTINUE Predictions (no grad)
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
        x = self.world_model.embedding(tokens_context, actions_context)
        mask = self.world_model._get_mask(x.size(1), x.device)
        
        for block in self.world_model.blocks:
            x = block(x, mask = mask)
        
        x = self.world_model.ln_final(x)
        t_final = tokens_context.size(1)
        last_start = (t_final - 1) * 4
        final_hidden = x[:, last_start:last_start+3, :]
        final_feature = self.feature_extractor(final_hidden)
        last_value = self.critic_network(final_feature)
        
        # RETURN TRAJECTORY
        return Trajectory(
            tokens=traj_tokens, actions=traj_actions,
            log_probs=traj_log_probs, feats=torch.stack(traj_features, dim=1),
            values=traj_values, rewards=traj_rewards,
            continues=traj_continues, last_value=last_value,
            entropies=traj_entropies,
        )
    
    def _rollout_cached(
        self,
        seed_tokens : torch.Tensor,
        seed_actions : torch.Tensor,
        horizon : Optional[int] = None,
    ) -> Trajectory:
        """KV-CACHED rollout - WITH TRANSFORMER HIDDEN STATE FEATURES
        
         FEATURES COME FROM TRANSFORMER HIDDEN STATES, NOT SEPARATE MLP ENCODER.
         
         (positions L0, L1, L2 after ln_final) instead of codebook embeddings = richer
        """
        
        batch_size = seed_tokens.size(0)
        t_seed = seed_tokens.size(1)
        H = horizon if horizon is not None else self.max_horizon
        
        max_timesteps = self.world_model.config.max_seq_len // 4
        assert t_seed + H <= max_timesteps
        
        #  PHASE 1: Prime KV cache with seed context  + extract initial hidden stats 
        
        
        _, _, _, kv_cache, x_full = self.world_model.forward_with_kv(
            seed_tokens, seed_actions
        )
        
        cached_seq_len = t_seed * 4
        
        # EXTRACT HIDDEN STATES FOR LAST SEED TIMESTEP (pos L0, L1, L2)
        # x_full is (B, t_seed*4, d_model) - we want the last 3 positions for the new timestep
        next_hidden = x_full[:, -3:, :]  # (B, 3, d_model)

        last_t_start = (t_seed - 1) * 4
        current_hidden = x_full[:, last_t_start:last_t_start+3, :]  # (B, 3, d_model)
        
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
        
        #  PHASE 2: Imagination rollout 
        for h in range(H):
            
            # FEATURE FROM Transformer Hidden State (not  codebook lookup)
            
            feature = self.feature_extractor(current_hidden)    # B, 1152
            
            # ACTOR PICKS ACTION  - from feature
            distribution = self.actor_network(feature)
            action = distribution.sample()
            log_probs = distribution.log_prob(action)
            entropy = distribution.entropy()
            
            # CRITIC, REWARD, CONTINUE Predictions (no grad)
            
            with torch.no_grad():
                value = self.critic_network(feature)
                reward = self.reward_network(feature)
                continue_logit = self.continue_network(feature)
                continue_prob = torch.sigmoid(continue_logit)
            
            # CASCADE PREDICTION (return hidden states , not just tokens) - with cache. 
            # L0 predicted from action, then L1 from L0, then L2 from L1, each via incremental forward.
            
            with torch.no_grad():
                predicted_tokens, kv_cache, cached_seq_len, next_hidden = \
                    self._do_cached_cascade(
                        kv_cache = kv_cache,
                        cached_seq_len = cached_seq_len,
                        action = action,
                        prev_tokens = last_cached_tokens,
                        prev_action_placeholder = (h > 0),  # pop+redo from step0 if not first step
                    )
            
            # UPDATE STEP for next iteration
            last_cached_tokens = predicted_tokens.unsqueeze(1)  # (B, 1, 3)
            current_hidden = next_hidden  # (B, 3, d_model)
            
            # STORE TRAJECTORY STEP
            
            traj_tokens[:, h] = predicted_tokens
            traj_actions[:, h] = action
            traj_log_probs[:, h] = log_probs
            traj_entropies[:, h] = entropy
            traj_features.append(feature)
            traj_values[:, h] = value
            traj_rewards[:, h] = reward
            traj_continues[:, h] = continue_prob
        
        #  BOOTSTRAP VALUES
        final_feature = self.feature_extractor(current_hidden)
        last_value = self.critic_network(final_feature)
        
        return Trajectory(
            tokens=traj_tokens, actions=traj_actions,
            log_probs=traj_log_probs, feats=torch.stack(traj_features, dim=1),
            values=traj_values, rewards=traj_rewards,
            continues=traj_continues, last_value=last_value,
            entropies=traj_entropies,
        )
