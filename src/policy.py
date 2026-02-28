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
            self.attn_norm = nn.LayerNorm(d_model)  # LAYER NORM FOR ATTENTION OUTPUT
            self.feat_dim = d_model                 # FINAL FEATURE DIMENSION AFTER ATTENTION (384)
            
        else:
            raise ValueError(f"Unknown Mode : {mode}. Use 'concat' or 'attention'.")
    
    @torch.no_grad()
    def _lookup_codebooks(
        self,
        tokens : torch.Tensor,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up codebook embeddings for each HRVQ layer """
        
        emb_l0 = self.hrvq.vq_layers[0].codebook(tokens[:, 0]) # (B, 384)
        emb_l1 = self.hrvq.vq_layers[1].codebook(tokens[:, 1]) # (B, 384)
        emb_l2 = self.hrvq.vq_layers[2].codebook(tokens[:, 2]) # (B, 384)
        
        return emb_l0, emb_l1, emb_l2
    
    def forward(
        self,
        tokens : torch.Tensor, 
    )-> torch.Tensor:
        """ Forward Pass """
        
        # LOOKUP CODEBOOK EMBEDDINGS FOR EACH LAYER 
        emb_l0, emb_l1, emb_l2 = self._lookup_codebooks(tokens) # (B, 384) each
        
        # OPTION A - CONCATENATE LAYER EMBEDDINGS 
        if self.mode == "concat":
            feat = torch.cat([emb_l0, emb_l1, emb_l2], dim = -1) # (B, 1152)
        
        # OPTION B - ATTENTION OVER LAYER EMBEDDINGS 
        
        elif self.mode == "attention":
            seq = torch.stack(tensors = [emb_l0, emb_l1, emb_l2], dim = 1)  # (B, 3, 384)
            attended, _ = self.cross_attn(seq, seq, seq)        # (B, 3, 384)
            attended = self.attn_norm(attended + seq)           # RESIDUAL + NORM 
            feat = attended.mean(dim = 1)                         # MEAN POOL (B, 384) 
        
        return feat
    
        
            
class PolicyNetwork(nn.Module):
    """ The Actor. 
    
    Maps 1152D feature -> categorical distribution over Atari actions.
    LayerNorm -> 2-layer MLP (ELU) -> softmax with 1% uniform mix to prevent collapse.
    Zero-init final layer -> uniform initial policy -> unbiased exploration at start."""

    def __init__(
        self,
        feat_dim : int,                 # CONCAT = 1152, ATTENTION = 384
        num_actions : int,              # GAME SPECIFIC (PONG = 6, BREAKOUT = 4, MsPACMAN = 9)
        hidden_dim : int = 512,         # SIZE OF HIDDEN LAYER IN MLP
        unimix : float = 0.01,          # DreamerV3 = 1%
    ):
        # SETUP 
        super().__init__()
        self.num_actions = num_actions
        self.unimix = unimix
        
        # NETWORK ARCHITECTURE, DreamerV3
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = num_actions) ,
        )
        
        # ZERO INITIALISATION - UNBIASED STARTING POLICY CONTRARY TO KAIMING INIT (PYTORCH DEFAULT)
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        
    def forward(
        self,
        feat : torch.Tensor,
    ) -> D.Categorical: 
        """ Forward Pass, returns Categorical Distribution over actions in given state """
        
        # PASS MLP TO GET ACTION LOGITS
        logits = self.net(feat)
        
        # UNIMIX BLENDING TO PREVENT COLLAPSE EARLY IN TRAINING
        
        if self.unimix > 0:
            
            probs = F.softmax(input = logits, dim = -1)                        # SOFTMAX - PROBABILITY DISTRIBUTION
            uniform_probs = torch.ones_like(input = probs) / self.num_actions  # UNIFORM DISTRIBUTION OVER ACTIONS
            probs = (1 - self.unimix) * probs + self.unimix * uniform_probs    # BLEND WITH UNIFORM
            dist = D.Categorical(probs = probs)                                # CATEGORICAL DISTRIBUTION
        
        else:
            dist = D.Categorical(logits = logits)                              # NO UNIMIX, STANDARD CATEGORICAL
        
        return dist
    

class ValueNetwork(nn.Module):
    """ The Critic. 
    
    Maps 1152D feature -> scalar 'how good is this state?'
    Same architecture as actor but outputs 1 value instead of num_actions logits.
    Separate from actor to avoid gradient interference (MSE vs REINFORCE scales differ)."""
    
    def __init__(
        self,
        feat_dim : int,         # CONCAT = 1152, ATTENTION = 384
        hidden_dim : int = 512, # SIZE OF HIDDEN LAYER IN MLP
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = 1) ,
        )
        
        # ZERO INITIALISATION FOR STABLE STARTING VALUE PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        
    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns scalar value prediction for given state """
        
        value = self.net(feat).squeeze(-1) # (B,) SCALAR VALUE PREDICTION
        return value

class RewardPredictor(nn.Module):
    """Predicts immediate reward from state features in symlog space.
    
    Trained supervised on real transitions (where true rewards exist).
    Used during imagination to provide reward signal when no real env is available."""
    
    def __init__(
        self,
        feat_dim : int,         # CONCAT = 1152, ATTENTION = 384
        hidden_dim : int = 512, # SIZE OF HIDDEN LAYER IN MLP
    ):
        super().__init__()
        
        # SAME ARCHITECTURE AS CRITIC BUT OUTPUTS 1 VALUE (REWARD PREDICTION) INSTEAD OF VALUE PREDICTION
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = 1) ,
        )
        
        # ZERO INITIALISATION FOR STABLE STARTING REWARD PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        

    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns scalar reward prediction for given state features """

        return self.net(feat).squeeze(-1) # (B,) SCALAR REWARD PREDICTION
    

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