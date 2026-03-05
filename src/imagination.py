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
    
    1. Receive Seed Context (tokens, actions from replay buffer)
    
    2. For H steps:
        a. Extract features from current tokens,
        b. Sample action from policy
        c. Run 3-sub-step cascade to predict [L0, L1, L2] 
        d. Record (feat, action, log_prob, reward, continue, value)
    
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
        
        logits_l0 = self.world_model(tokens_context, actions_context)
        
        # (B,) SAMPLED L0 tokens for next step
        token_l0 = self._sample_token(logits = logits_l0[:, -1, :])  
        
        """ 2. PREDICT L1 from L0 """
        # L1_{t + 1} = f(L0_{t + 1})
        
        # PLACE SAMPLED L0 TOKEN IN CONTEXT (Replace last L0 token with sampled)
        # (B, 1, 3) - [0, 0, 0] placeholder for next step
        next_partial = torch.zeros(batch_items, 1, 3, dtype = torch.long, device = self.device) 
        
        # INSERT SAMPLED L0 TOKEN INTO CONTEXT
        next_partial[:, 0, :] = token_l0 
        
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
        
    
    def rollout():
        pass
        

