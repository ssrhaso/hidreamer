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

from world_model import HierarchicalWorldModel, hierarchical_causal_mask

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
    """ Run Horizon-Step imagination inside frozen world model. """
    pass

