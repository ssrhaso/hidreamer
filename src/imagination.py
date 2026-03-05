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
    def _sample_token():
        pass


    @torch.no_grad()
    def _cascade_predict_next():
        pass
    
    def rollout():
        pass
        

