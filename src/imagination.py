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
    pass


class ImagineRollout:
    """ Run Horizon-Step imagination inside frozen world model. """
    pass

