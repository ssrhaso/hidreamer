""" POLICY NETWORKS FOR ACTOR CRITIC

ALL TRAINABLE COMPONENTS FOR IMAGINATION BASED RL

World Model is FROZEN during PPO training - only these networks are updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Optional
import math

# SYMLOG / SYMEXP TRANSFORMS FOR REWARDS AND VALUES TO HANDLE ATARI'S WIDE REWARD SCALE 
def symlog(
    x : torch.Tensor
    ) -> torch.Tensor:
    """Compress large magnitudes: sign(x) * ln(|x| + 1). 
    
    Keeps small values ~unchanged, squashes 999 -> 6.9.
    Applied to reward targets and critic values so Breakout's large rewards don't dominate Pong's small ones."""
    
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(
    x : torch.Tensor
    ) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1). 
    
    Decompress predictions back to real scale."""
    return torch.sign(x) * (torch.expm1(torch.abs(x)) - 1)


# POLICY NETWORKS
class HierarchicalFeatureExtractor(nn.Module):
    """Converts HRVQ token indices -> dense feature vector by looking up frozen codebook embeddings.
    
    OPTION 1 - Concat mode: [codebook_0[L0] | codebook_1[L1] | codebook_2[L2]] → 1152D.
    OPTION 2 - Attention mode: 3-token self-attention over the three layer embeddings -> pooled 384D."""
    
    def __init__(
        self,
        hrvq_tokenizer, # FROZEN - used only for lookups
        mode : str = 'concat', # 'concat' or 'attention'
        d_model : int = 384,   # only used for attention mode
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.hrvq = hrvq_tokenizer # FROZEN 
        
        if mode == "concat":
            self.feat_dim = d_model * 3  # 1152 Dimension (3 layers * 384 each)
        
        elif mode == "attention":
            self.feat_dim = d_model      # 384 Dimension (pooled output)
            
            # 3 TOKEN SELF ATTENTION LAYER AGGREGRATION
            self.cross_attn = nn.MultiheadAttention(
                embed_dim = d_model, # 384 DIM
                num_heads = 4,       # 4 HEADS
                dropout = 0.0,       # 0.0 DROPOUT SINCE FEATURE EXTRACTION
                batch_first = True  
            )
            
        else:
            raise ValueError(f"Unknown Mode : {mode}. Use 'concat' or 'attention'.")
            
            
            
class PolicyNetwork(nn.Module):
    """ The Actor. 
    
    Maps 1152D feature -> categorical distribution over Atari actions.
    LayerNorm -> 2-layer MLP (ELU) -> softmax with 1% uniform mix to prevent collapse.
    Zero-init final layer -> uniform initial policy -> unbiased exploration at start."""


class ValueNetwork(nn.Module):
    """ The Critic. 
    
    Maps 1152D feature -> scalar 'how good is this state?'
    Same architecture as actor but outputs 1 value instead of num_actions logits.
    Separate from actor to avoid gradient interference (MSE vs REINFORCE scales differ)."""

class RewardPredictor(nn.Module):
    """Predicts immediate reward from state features in symlog space.
    
    Trained supervised on real transitions (where true rewards exist).
    Used during imagination to provide reward signal when no real env is available."""

class ContinueNetwork(nn.Module):
    """Predicts p(episode continues) as a logit -> sigmoid for probability.
    
    Bias-initialized positive (sigmoid(2)≈0.88) because 99% of Atari steps aren't terminal.
    During imagination: effective_discount = gamma * p(continue), soft-truncating near game-over states."""

class SlowValueTarget:
    """EMA copy of the critic updated at τ=0.02 per step for stable λ-return targets.
    
    Solves the moving-target problem: critic can't train on its own rapidly-changing predictions.
    Same technique as DDPG, SAC, DreamerV3."""

def compute_lambda_returns():
    """Backwards-recursive λ-return: blends 1-step TD (low variance) with Monte Carlo (low bias).
    G_t = r_t + Y * c_t * [(1-λ)*V(s_{t+1}) + λ*G_{t+1}]. 
    
    λ=0.95 uses 95% of long-horizon info.
    Returns (B, H) targets that the critic is trained to predict."""

class ReturnNormalizer:
    """Normalizes advantages by the 5th-95th percentile range of recent returns.
    
    Robust under sparse rewards where std≈0 would cause division-by-zero explosion.
    EMA-tracked percentiles (decay=0.99) adapt smoothly as training progresses."""

def count_policy_params():
    """Counts trainable parameters across all four networks. Sanity check: should be ~1.5-3M total"""