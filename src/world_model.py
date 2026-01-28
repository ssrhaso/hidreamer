""" SKELETON CODE FOR WORLD MODEL MODULE 

- WIP 1 : BASELINE IMPLEMENTATION - STORM(2023) INSPIRED
- WIP 2 : TWISTER(2025) / DREAMERv4(2025) INSPIRED IMPROVEMENTS

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import yaml
from pathlib import Path

@dataclass 
class WorldModelConfig:
    """ CONFIG MATCHING configs/worldmodel.yaml """

class TokenEmbedding(nn.Module):
    """ EMBED HIERARCHICAL (HRVQ) TOKENS + ACTIONS into TRANSFORMER SEQUENCE"""
    

def hierarchical_causal_mask():
    """ CAUSAL MASK  """
    
class TransformerBlock(nn.Module):
    """ STANDARD TRANSFORMER BLOCK  """
    

class HierarchicalWorldModel(nn.Module):
    """ MAIN HIERARCHICAL WORLD MODEL FOR ATARI100K PREDICTION  """
    

def hierarchical_loss():
    """ HIERARCHICAL LOSS FUNCTION  """
    
if __name__ == "__main__":
    pass

    
   