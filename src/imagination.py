""" 
IMAGINATION ROLLOUT FOR PPO POLICY TRAINING

References:
- IRIS (Micheli et al., ICLR 2023) — discrete token autoregressive rollout
- TWISTER (Burchi & Timofte, ICLR 2025) — imagination in transformer WM
- Dreamer 4 (Hafner et al., Sep 2025) — block-causal imagination
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
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
    2. Extract FEATURES from predicted tokens, pass through PPO ACTOR to get ACTIONS and LOG PROBS.
    3. Return Trajectory (dataclass) for PPO training.
    """
    
    def __init__(
        self,
        world_model : HierarchicalWorldModel,
        feature_extractor,
        
        actor_network,  
        critic_network,
        reward_network,
        continue_network,
        max_horizon : int = 30, # UPPER BOUND (Cosine Schedule)
        temperature : float = 1.0,
        device : torch.device = None,
    ):
        
        # 1. NEURAL NETWORKS
        self.world_model = world_model
        self.feature_extractor = feature_extractor
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.reward_network = reward_network
        self.continue_network = continue_network
        
        # 2. HYPERPARAMETERS
        self.max_horizon = max_horizon
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    @torch.no_grad()
    def _sample_token(
        self,
        logits : torch.tensor,   # (B, num_tokens)
    ) -> torch.tensor:           # (B,) sampled token indices
        """ STEP A HELPER: SAMPLE OF L0 -> L1 -> L2 TOKENS.
        
        Receives raw scores from world model, samples a
        Categorical Distribution over Codebook entries
        """
        
        # GREEDY SAMPLING (eval / debug)
        if self.temperature <= 0:
            return logits.argmax(dim = -1)
        
        # STOCHASTIC SAMPLING
        scaled_logits = logits / self.temperature                               # SCALE by temp
        probabilities = F.softmax(scaled_logits, dim=-1)                        # CONVERT TO PROBABILITIES
        distribution = torch.multinomial(input = probabilities, num_samples = 1).squeeze(-1)  
        
        return distribution


    @torch.no_grad()
    def _cascade_predict_next(
        self,
        tokens_context : torch.tensor,   # (B, timesteps, 3)    - input token context (HRVQ tokens)
        actions_context : torch.tensor,   # (B, timesteps)       - input action context
    ) -> torch.tensor:                   # (B, 3)               - predicted next [L0, L1, L2] tokens
        """ STEP A: CASCADE PREDICTION OF L0 -> L1 -> L2 TOKENS.
        
        Predict next frame tokens via 3 forward passes through the world model
        (With Hierarchical Flow)
        
        1. PREDICT L0 from Action Context
        2. PREDICT L1 from L0
        3. PREDICT L2 from L1 
        """
        
        # INITIALISE CASCADE 
        batch_items = tokens_context.size(0)
        timesteps = tokens_context.size(1)
        
        """ 1. PREDICT L0 from ACTION CONTEXT """
        # L0_{t + 1} = f(A pos't) (L0_{t + 1} is predicted from Action context at time t)
        
        logits_l0, _, _ = self.world_model(tokens_context, actions_context)
        
        # (B,) SAMPLED L0 tokens for next step
        token_l0 = self._sample_token(logits = logits_l0[:, -1, :])  
        
        """ 2. PREDICT L1 from L0 """
        # L1_{t + 1} = f(L0_{t + 1})
        
        # PLACE SAMPLED L0 TOKEN IN CONTEXT (Replace last L0 token with sampled)
        # (B, 1, 3) - [0, 0, 0] placeholder for next step
        next_partial = torch.zeros(batch_items, 1, 3, dtype = torch.long, device = self.device) 
        
        # INSERT SAMPLED L0 TOKEN INTO CONTEXT
        next_partial[:, 0, 0] = token_l0 
        
        # DUMMY ACTION CONTEXT (zeros) FOR L1 PREDICTION
        # (B, 1) - dummy action for next step
        dummy_action = actions_context[:, -1:] 
        
        # EXTEND BOTH CONTEXT BUFFERS by 1 STEP
        tokens_context_extended = torch.cat(tensors = [tokens_context, next_partial], dim = 1)
        actions_context_extended = torch.cat(tensors = [actions_context, dummy_action], dim = 1)
        
        # FORWARD PASS THROUGH WORLD MODEL TO PREDICT L1 LOGITS
        _, logits_l1, _ = self.world_model(tokens_context_extended, actions_context_extended)
        
        # (B,) SAMPLED L1 TOKENS FOR NEXT STEP
        token_l1 = self._sample_token(logits = logits_l1[:, -1, :])  
        
        """ 3. PREDICT L2 from L1 """
        
        # FILL IN L1 TOKENS IN CONTEXT
        tokens_context_extended[:, -1, 1] = token_l1
        
        # FORWARD PASS THROUGH WORLD MODEL TO PREDICT L2 LOGITS
        _, _, logits_l2 = self.world_model(tokens_context_extended, actions_context_extended)
        
        # (B,) SAMPLED L2 TOKENS FOR NEXT STEP
        token_l2 = self._sample_token(logits = logits_l2[:, -1, :])
        
        """ CONCATENATE PREDICTED TOKENS INTO (B, 3) OUTPUT """
        predicted_tokens = torch.stack(tensors = [token_l0, token_l1, token_l2], dim = -1)
        
        return predicted_tokens
        
    
    def rollout(
        self,
        seed_tokens : torch.tensor,     # (B, T_seed, 3)    - input token context (HRVQ tokens)
        seed_actions : torch.tensor,    # (B, T_seed)       - input action context
        horizon : Optional[int] = None, # override horizon (scheduler)
    ) -> Trajectory:
        """ RUN IMAGINATION ROLLOUT INSIDE WORLD MODEL. 
        
        SEED CONTEXT = Replay Buffer 
        """     
        
        # INITIALISE CONTEXT BUFFERS WITH SEED CONTEXT
        batch_size = seed_tokens.size(0)                          # batch size from seed context
        t_seed = seed_tokens.size(1)                              # real frames in seed context
        H = horizon if horizon is not None else self.max_horizon  # rollout horizon
        
        # VALIDATE sequence length
        max_timesteps = self.world_model.config.max_seq_len // 4
        assert t_seed + H <= max_timesteps, (
            f"Seed ({t_seed}) + Horizon ({H}) = {t_seed + H} exceeds "
            f"max timesteps ({max_timesteps})"
        )
        
        """ Growing Context Buffers for Imagination Rollout """
        tokens_context = seed_tokens.clone()    # (B, t_current, 3)
        actions_context = seed_actions.clone()  # (B, t_current)
        
        """ Trajectory Storage (only imagination rollout outputs)"""
        trajectory_tokens       = torch.zeros(batch_size, H, 3, dtype = torch.long, device = self.device)
        trajectory_actions      = torch.zeros(batch_size, H, dtype = torch.long, device = self.device)
        trajectory_log_probs    = torch.zeros(batch_size, H, dtype = torch.float, device = self.device)
        trajectory_entropies    = torch.zeros(batch_size, H, dtype = torch.float, device = self.device)
        trajectory_feats        = []
        trajectory_values       = torch.zeros(batch_size, H, dtype = torch.float, device = self.device)
        trajectory_rewards      = torch.zeros(batch_size, H, dtype = torch.float, device = self.device)
        trajectory_continues    = torch.zeros(batch_size, H, dtype = torch.float, device = self.device)
        
        """ IMAGINATION ROLLOUT LOOP """
        for h in range(H):
            
            """ 1. Get CURRENT FRAME (FEATURE EXTRACTOR) """
            current_tokens = tokens_context[:, -1, :]            # (B, 3) - last step tokens in context
            feature = self.feature_extractor(current_tokens)     # (B, feat_dim) - lookups for features
            
            """ 2. Get ACTION from ACTOR NETWORK (Distribution) """
            distribution = self.actor_network(feature)           # (B, num_actions) - action logits ("odds for every action")
            action = distribution.sample()                       # (B,) - sampled action indices    ("sample concrete decision from odds")
            log_probabilities = distribution.log_prob(action)    # (B,) - log probs of sampled actions ("how likely was x decision?")
            entropy = distribution.entropy()                     # (B,) - entropy of action distribution ("how uncertain was I?")
            
            """ 3. Auxilary Predictions from CRITIC, REWARD, CONTINUE NETWORKS """
            with torch.no_grad():
                value = self.critic_network(feature)             # (B,) - value prediction for current state
                reward = self.reward_network(feature)            # (B,) - reward prediction for current state
                continue_logit = self.continue_network(feature)  # (B,) - continue logit for current state
                continue_prob = torch.sigmoid(continue_logit)    # (B,) - continue probability for current state
                
            """ 4. Write POLICY ACTION into context - WM must see action before cascade """
            actions_context[:, -1] = action
            
            """ 5. MAIN CASCADE PREDICT NEXT TOKENS (L0 -> L1 -> L2) - WM sees POLICY ACTIONS"""
            next_tokens = self._cascade_predict_next(tokens_context, actions_context)  # (B, 3) - predicted next tokens
            
            """ 6. Extend CONTEXT BUFFERS with NEW TOKENS + Placeholder ACTION ZERO (To be filled)"""
            tokens_context = torch.cat(tensors = [tokens_context, next_tokens.unsqueeze(1)], dim = 1)  # (B, t_current + 1, 3)
            actions_context = torch.cat(tensors = [actions_context, torch.zeros(batch_size, 1, dtype = torch.long, device = self.device)], dim = 1)  # (B, t_current + 1)
        
            """ 7. Record TRAJECTORY DATA """
            trajectory_tokens[:, h] = next_tokens
            trajectory_actions[:, h] = action
            trajectory_log_probs[:, h] = log_probabilities
            trajectory_entropies[:, h] = entropy
            trajectory_feats.append(feature)
            trajectory_values[:, h] = value
            trajectory_rewards[:, h] = reward
            trajectory_continues[:, h] = continue_prob
            
            pass
        
        with torch.no_grad():
            pass
        
        return Trajectory(...)
    

