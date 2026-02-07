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
    
    d_model : int  = 384            # EMBEDDING DIMENSION (WIDTH OF NN)             - HRVQ EMBEDDING DIMENSION
    n_layers : int = 6              # NUMBER OF TRANSFORMER BLOCKS (DEPTH OF NN)    - (Kaplan et al. 2020)
    n_heads : int = 6               # NUMBER OF ATTENTION HEADS                     - (Vaswani et al. 2017) - standard rule of = d_model / 64
    d_ff : int    = 1536            # DIMENSION OF FEEDFORWARD NETWORK              - (Vaswani et al. 2017) - standard rule of = 4 * d_model
    dropout : float = 0.1           # DROPOUT RATE                                  - (Devlin et al. 2017)  - BERT, GPT, STORM use 0.1       
    max_seq_len : int = 256         # MAXIMUM SEQUENCE LENGTH                       - ~ 65k positions of memory, Fits T4 GPU safely
    num_codes : int = 256           # NUMBER OF CODEBOOK ENTRIES                    - HRVQ Codebook Size
    num_actions : int = 9           # NUMBER OF POSSIBLE ACTIONS                    - ATARI100K has 9 discrete actions
    
    # HIERARCHICAL LOSS (NOVELTY) 
    
    # e.g.: layer_weights = [1.0, 0.5, 0.1] for 3-layer HRVQ (L0, L1, L2)
    # HIGHER WEIGHTING FOR COARSE LAYER (L0) if desired; default set in __post_init__
    layer_weights: list = None
    
    
    def __post_init__(self):
        """ INITIALISE DERIVED PARAMETERS """
        if self.layer_weights is None:
            # DEFAULT FOR 3-LAYER HRVQ (L0, L1, L2)
            self.layer_weights = [1.0, 0.5, 0.1]    
        
        # VALIDATE HEAD DIMENSIONS
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        
        # VALIDATE SEQUENCE LENGTH
        assert self.max_seq_len % 4 == 0 , f"max_seq_len {self.max_seq_len} must be divisible by 4 (tokens per time step)"


    @classmethod
    def from_yaml(
        cls,
        path : str = "configs/worldmodel.yaml"
    ):
        """ LOAD CONFIG FROM YAML FILE  """
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        return cls(
            # ARCHITECTURE
            d_model = cfg['model']['d_model'],
            n_layers = cfg['model']['n_layers'],
            n_heads = cfg['model']['n_heads'],
            d_ff = cfg['model']['d_ff'],
            dropout = cfg['model']['dropout'],
            max_seq_len = cfg['model']['max_seq_len'],
            num_codes = cfg['model']['num_codes'],
            num_actions = cfg['model']['num_actions'],
            
            # HIERARCHICAL LOSS
            layer_weights = cfg['model']['layer_weights'],
        )
        
    
    def __repr__(self):
        return(
            f"WORLD MODEL CONFIG:\n"
            
            f"  d_model: {self.d_model}\n"
            f"  n_layers: {self.n_layers}\n"
            f"  n_heads: {self.n_heads}\n"
            f"  d_ff: {self.d_ff}\n"
            f"  dropout: {self.dropout}\n"
            f"  max_seq_len: {self.max_seq_len}\n"
            f"  num_codes: {self.num_codes}\n"
            f"  num_actions: {self.num_actions}\n"
            f"  layer_weights: {self.layer_weights}\n"
        )
    

    
    
    
class TokenEmbedding(nn.Module):
    """ EMBED HIERARCHICAL (HRVQ) TOKENS + ACTIONS into INTERLEAVED TRANSFORMER SEQUENCE
    
    INPUT EXAMPLE:
    obs_sequence = [L0_t0, L1_t0, L2_t0, L0_t1, L1_t1, L2_t1, ...]
    action_sequence = [A_t0, A_t1, ...]
    
    OUTPUT EXAMPLE:
    sequence = [L0_t0, L1_t0, L2_t0, A_t0, L0_t1, L1_t1, L2_t1, A_t1, ...]
           └──── timestep 0 ────┘  └──── timestep 1 ────┘
    """


    def __init__(
        self,
        config : WorldModelConfig,
    ):
        super().__init__()
        self.config = config
    
    
        """ LOOKUPS  """
        
        # 1. TOKEN Lookups - (3 tables for L0, L1, L2)
        self.token_embeds = nn.ModuleList([
            nn.Embedding(
                num_embeddings = config.num_codes,  # 256
                embedding_dim = config.d_model      # 384
            ) 
            for _ in range(3) # L0, L1, L2
        ])
        
        # 2. ACTION Lookup - (1 table for all actions)
        self.action_embed = nn.Embedding(
            num_embeddings = config.num_actions,  # 9
            embedding_dim = config.d_model        # 384
        )
        
        # 3. LEVEL Lookup - (1 table for L0, L1, L2 + ACTION)
        self.level_embed = nn.Embedding(
            num_embeddings = 4,   # LEVELS: L0, L1, L2 + ACTION
            embedding_dim = config.d_model
        )
        
        # 4. POSITION Lookup - (1 table for all positions)
        self.pos_embed = nn.Embedding(
            num_embeddings = config.max_seq_len,
            embedding_dim = config.d_model
        )
        
    def forward(
        self,
        tokens : torch.tensor,      # SHAPE: (B, T, 3) - 3 LAYERS
        actions : torch.tensor,     # SHAPE: (B, T) - ACTIONS
    ) -> torch.tensor:              # SHAPE: (B, T*4, d_model) - 4 TOKENS PER TIME STEP
        """ FORWARD PASS  """
        
        B, T, _ = tokens.shape
        device = tokens.device
        
        
        # 1. EMBED Each component
        emb_level0 = self.token_embeds[0](tokens[..., 0]) 
        emb_level1 = self.token_embeds[1](tokens[..., 1])
        emb_level2 = self.token_embeds[2](tokens[..., 2])
        emb_action = self.action_embed(actions)
        
        # 2. ADD LEVEL EMBEDDING
        level_ids = torch.arange(4, device = device)  
        level_embeds = self.level_embed(level_ids)
        
        emb_level0 = emb_level0 + level_embeds[0]
        emb_level1 = emb_level1 + level_embeds[1]
        emb_level2 = emb_level2 + level_embeds[2]
        emb_action = emb_action + level_embeds[3]
        
        # 3. INTERLEAVE TOKENS PER TIME STEP
        seq = torch.stack(
            [emb_level0, emb_level1, emb_level2, emb_action], 
            dim = 2
        )
        
        seq = seq.reshape(B, T * 4, self.config.d_model)  # (B, T*4, d_model)
        
        # 4. ADD POSITIONAL EMBEDDING
        positions = torch.arange(T * 4, device = device)
        seq = seq + self.pos_embed(positions)
        
        return seq
        
     
        
        
def hierarchical_causal_mask(
    seq_len : int,
    device : torch.device
) -> torch.tensor:
    """ CAUSAL MASK TO ENSURE MODEL CANNOT 'SEE' FUTURE TOKENS (TRIANGULAR MASK SINCE TOKENS ARE INTERLEAVED SEQUENTIALLY) """
    assert seq_len % 4 == 0, f"Sequence Length {seq_len} must be 4 tokens per timestep (L0, L1, L2, ACTION)"
    
    # 1. STANDARD CAUSAL MASK (NO HIERARCHY), example:
    """ 
    F T T T 
    F F T T
    F F F T
    F F F F
    """
    # TRUE = BLOCK ATTENTION, FALSE = ALLOW ATTENTION
    mask = torch.triu(
        torch.ones(size = (seq_len, seq_len), device = device), 
        diagonal = 1
    ).bool()
    
    # 2. REFINE within each timestep
    num_timesteps = seq_len // 4
    
    for t in range(num_timesteps):
        base = t * 4
        
        # POS 0 (L0) - SEES : SELF
        
        # POS 1 (L1) - SEES : SELF, L0(SAME TIMESTEP)
        mask[base + 1, base + 0] = False
        
        # POS 2 (L2) - SEES : SELF, L0(SAME TIMESTEP), L1(SAME TIMESTEP)
        mask[base + 2, base] = False
        mask[base + 2, base + 1] = False
        
        # POS 3 (ACTION) - SEES : SELF, L0(SAME TIMESTEP), L1(SAME TIMESTEP), L2(SAME TIMESTEP)
        mask[base + 3, base] = False
        mask[base + 3, base + 1] = False
        mask[base + 3, base + 2] = False
        
        

    # 3. CROSS TIMESTEP HIERARCHY
    for t_query in range(num_timesteps):
        for t_key in range(t_query):
            base_query = t_query * 4
            base_key = t_key * 4
            
            # BLOCK SEEING PAST
            for position in [1, 2, 3]:
                mask[base_query + position, base_key + 1] = True   # BLOCK L1, L2, ACTION seeing L1 (past)
                mask[base_query + position, base_key + 2] = True   # BLOCK L1, L2, ACTION seeing L2 (past)
                
                
    # 4. CONVERT TO ATTENTION FORMAT (FROM BOOL)
    
    float_mask = torch.zeros(size = (seq_len, seq_len), device = device)
    float_mask = float_mask.masked_fill(mask = mask, value = float('-inf'))
    
    """"
    Standard: sees ALL past    Hierarchical: blocks fine details from past
    L1₁: F F F F               L1₁: F X X F    (X = blocks L1₀, L2₀)

    """
    return float_mask
    
    
class TransformerBlock(nn.Module):
    """ STANDARD TRANSFORMER BLOCK 
    
    - ATTENTION LAYERS (Controlled by causal mask)
    - FFN LAYERS (Processes information within position)
    """
    
    def __init__(
        self,
        config : WorldModelConfig
    ):
        super().__init__()
        
        """ 6 HEAD MULTI-HEAD SELF-ATTENTION LAYER
        'Group Reflection' layer """
        self.attn = nn.MultiheadAttention(
            embed_dim = config.d_model, # 384 DIM
            num_heads = config.n_heads, # 6 HEADS
            dropout = config.dropout,   # 0.1 DROPOUT (PREVENT OVERFITTING)
            batch_first = True
        )
        
        
        """ FEEDFORWARD NETWORK (POSITION-WISE) 
        'Self Reflection' layer """
        self.ffn = nn.Sequential(
            nn.Linear(in_features = config.d_model, out_features = config.d_ff),  
            # 384 -> 1536 (EXAPND FOR COMPLEX FEATURES)
            
            nn.GELU(),                               # NON LINEARITY
            nn.Dropout(config.dropout),              # 0.1 DROPOUT (PREVENT OVERFITTING)
            
            nn.Linear(in_features = config.d_ff, out_features = config.d_model),  
            # 1536 -> 384 (PROJECT BACK TO MODEL DIMENSION)                  
            
            nn.Dropout(config.dropout)               # 0.1 DROPOUT (PREVENT OVERFITTING)
        )
        
        
        """ PRE-NORM LAYER NORMALIZATION (STABILISE TRAINING) """
        self.ln1 = nn.LayerNorm(normalized_shape = config.d_model) # BEFORE ATTENTION
        self.ln2 = nn.LayerNorm(normalized_shape = config.d_model) # BEFORE FFN
        

    def forward(
        self,
        x : torch.tensor,         # SHAPE: (B, seq_len, 384) 
        mask : torch.tensor       # SHAPE: (seq_len, seq_len) - HCAUSAL MASK
    ) -> torch.tensor:            # SHAPE: (B, seq_len, 384) 
        """ FORWARD PASS THROUGH 1 TRANSFORMER BLOCK
        X > NORM > ATTENTION > RESIDUAL > NORM > FFN > RESIDUAL > Y
        """
        
        # 1. PRE-NORM (BEFORE ATTENTION)
        x_norm = self.ln1(x)
        
        # 2. ATTENTION 
        attn_out, _ = self.attn(
            query = x_norm,         # WHAT WE ARE LOOKING FOR (ALL AFTER MASKING)
            key = x_norm,           # WHAT WE ARE COMPARING TO (ALL AFTER MASKING)
            value = x_norm,         # WHAT WE ARE USING TO UPDATE (ALL AFTER MASKING)
            attn_mask = mask,       # MASK FOR CAUSALITY + HIERARCHY
            need_weights = False    # NO WEIGHTS (MEMORY SAVER)
        )
        
        # 3. RESIDUAL (ATTENTION OUT)
        x = x + attn_out 
        
        # 4. PRE-NORM (BEFORE FFN)
        x_norm = self.ln2(x)
        
        # 5. FEEDFORWARD NETWORK (MLP)
        ffn_out = self.ffn(x_norm)
        
        # 6. RESIDUAL (FFN OUT)
        x = x + ffn_out
        
        return x
        
    


class HierarchicalWorldModel(nn.Module):
    """ MAIN HIERARCHICAL WORLD MODEL FOR ATARI100K PREDICTION  """
    pass

def hierarchical_loss():
    """ HIERARCHICAL LOSS FUNCTION  """
    pass



if __name__ == "__main__":
    
    
    
    """ HIERARCHICAL MASK TEST"""
    # mask = hierarchical_causal_mask(8, torch.device('cpu'))
    # print("Hierarchical Causal Mask (8 positions = 2 timesteps):")
    # print(mask)
    
    # # Verify specific properties
    # # Test the hierarchical blocking
    # assert mask[5, 1] == float('-inf'), "L1_t1 should NOT see L1_t0 (hierarchical block)"
    # assert mask[5, 2] == float('-inf'), "L1_t1 should NOT see L2_t0 (hierarchical block)"
    # assert mask[6, 1] == float('-inf'), "L2_t1 should NOT see L1_t0 (hierarchical block)"
    # assert mask[6, 2] == float('-inf'), "L2_t1 should NOT see L2_t0 (hierarchical block)"

    # # Test that L0 from past is still visible
    # assert mask[5, 0] == 0, "L1_t1 SHOULD see L0_t0 (physics from past)"
    # assert mask[6, 0] == 0, "L2_t1 SHOULD see L0_t0 (physics from past)"

    # # Test within-timestep hierarchy
    # assert mask[5, 4] == 0, "L1_t1 SHOULD see L0_t1 (current physics)"
    # assert mask[6, 4] == 0, "L2_t1 SHOULD see L0_t1 (current physics)"
    # assert mask[6, 5] == 0, "L2_t1 SHOULD see L1_t1 (current mechanics)"

    # print("ALL hierarchical assertions passed")


    """ TRANSFORMER BLOCK TEST """    
    device = torch.device('cpu')
    config = WorldModelConfig()  # defaults
    
    # Create fake input: batch=2, timesteps=4
    B, T = 2, 4
    tokens = torch.randint(0, config.num_codes, (B, T, 3))   # Random HRVQ tokens
    actions = torch.randint(0, config.num_actions, (B, T))    # Random actions
    
    # Step 1: Embed (using existing TokenEmbedding)
    embed = TokenEmbedding(config)
    seq = embed(tokens, actions)  # (2, 16, 384)
    print(f"TokenEmbedding output: {seq.shape}")
    
    # Step 2: Create mask
    mask = hierarchical_causal_mask(T * 4, device)  # (16, 16)
    print(f"Mask shape: {mask.shape}")
    
    # Step 3: Pass through ONE TransformerBlock
    block = TransformerBlock(config)
    out = block(seq, mask)
    
    # TEST A: Shape preserved (must be identical in/out)
    assert out.shape == seq.shape, f"FAIL: shape {out.shape} != {seq.shape}"
    print(f" Shape preserved: {seq.shape} → {out.shape}")
    
    # TEST B: Output is different from input (block actually transformed something)
    assert not torch.allclose(out, seq, atol=1e-6), "FAIL: output identical to input"
    print(f" Output differs from input (block did work)")
    
    # TEST C: No NaN or Inf in output (mask didn't break anything)
    assert not torch.isnan(out).any(), "FAIL: NaN in output"
    assert not torch.isinf(out).any(), "FAIL: Inf in output"
    print(f" No NaN/Inf in output")
    
    # TEST D: Count parameters
    num_params = sum(p.numel() for p in block.parameters())
    print(f" Parameters per block: {num_params:,}")
    print(f"  ( X {config.n_layers} blocks = {num_params * config.n_layers:,} total for transformer)")
    
    # TEST E: Gradients flow through (can we train this?)
    loss = out.sum()
    loss.backward()
    has_grads = all(p.grad is not None for p in block.parameters())
    assert has_grads, "FAIL: some parameters have no gradients"
    print(f" Gradients flow to all parameters")
    print("\nALL TRANSFORMER BLOCK TESTS PASSED")
    
   