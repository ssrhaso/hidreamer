""" SKELETON CODE FOR WORLD MODEL MODULE 

- WIP 1 : BASELINE IMPLEMENTATION - STORM(2023) INSPIRED
- WIP 2 : TWISTER(2025) / DREAMERv4(2025) INSPIRED IMPROVEMENTS
- WIP 3 : KV-CACHE FOR IMAGINATION ROLLOUT ACCELERATION

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import yaml
from pathlib import Path

import wandb

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

    def embed_partial(
        self,
        tokens : torch.Tensor,      # (B, 1, 3) — single timestep
        actions : torch.Tensor,     # (B, 1)    — single action
        start_pos : int,             # position offset in the full sequence
    ) -> torch.Tensor:              # (B, 4, d_model) — 4 new positions
        """Embed a SINGLE TIMESTEP with correct positional encoding.
        
        Used by incremental KV-cache forward to embed only the NEW positions
        without re-embedding the entire context.
        """
        B = tokens.size(0)
        device = tokens.device
        
        # 1. EMBED Each component (single timestep)
        emb_l0 = self.token_embeds[0](tokens[:, 0, 0])    # (B, d_model)
        emb_l1 = self.token_embeds[1](tokens[:, 0, 1])    # (B, d_model)
        emb_l2 = self.token_embeds[2](tokens[:, 0, 2])    # (B, d_model)
        emb_act = self.action_embed(actions[:, 0])          # (B, d_model)
        
        # 2. ADD LEVEL EMBEDDING
        level_ids = torch.arange(4, device=device)
        level_embeds = self.level_embed(level_ids)
        
        emb_l0 = emb_l0 + level_embeds[0]
        emb_l1 = emb_l1 + level_embeds[1]
        emb_l2 = emb_l2 + level_embeds[2]
        emb_act = emb_act + level_embeds[3]
        
        # 3. STACK into (B, 4, d_model)
        seq = torch.stack([emb_l0, emb_l1, emb_l2, emb_act], dim=1)
        
        # 4. ADD POSITIONAL EMBEDDING (offset by start_pos)
        positions = torch.arange(start_pos, start_pos + 4, device=device)
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
        self.config = config
        self.d_head = config.d_model // config.n_heads
        
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
        """ FORWARD PASS THROUGH 1 TRANSFORMER BLOCK (ORIGINAL - UNCHANGED)
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

    def forward_with_kv(
        self,
        x : torch.Tensor,         # (B, seq_len, 384)
        mask : torch.Tensor,      # (seq_len, seq_len)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that ALSO RETURNS the KV cache for this layer.
        
        Identical computation to forward(), but extracts K and V tensors
        after the in_proj inside nn.MultiheadAttention so we can reuse them.
        
        Returns:
            x:      (B, seq_len, 384) — same as forward()
            kv:     (cached_k, cached_v) each (B, n_heads, seq_len, d_head)
        """
        
        # 1. PRE-NORM
        x_norm = self.ln1(x)
        
        # 2. ATTENTION — use forward() but also extract K, V for caching
        # nn.MHA stores in_proj_weight: (3*d_model, d_model) and in_proj_bias: (3*d_model,)
        # We project Q, K, V manually to extract K, V for caching
        attn_out, cached_k, cached_v = self._attention_with_kv_extract(
            x_norm, x_norm, x_norm, mask
        )
        
        # 3. RESIDUAL
        x = x + attn_out
        
        # 4. FFN
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x, (cached_k, cached_v)
    
    def forward_incremental(
        self,
        x_new : torch.Tensor,                               # (B, new_len, 384) — NEW positions only
        past_kv : Tuple[torch.Tensor, torch.Tensor],       # (cached_k, cached_v) from previous pass
        mask_rows : torch.Tensor,                            # (new_len, total_len) — mask rows for new positions
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward: process ONLY new positions against cached K/V.
        
        This is the core of the KV-cache speedup. Instead of recomputing
        attention over the entire context, we:
        1. Project new positions to get Q_new, K_new, V_new
        2. Concatenate K_new, V_new with cached K, V
        3. Compute attention: Q_new @ [K_cached | K_new]^T
        4. Apply hierarchical mask rows for the new positions
        
        Returns:
            x_new:      (B, new_len, 384) — output for new positions only
            updated_kv: (updated_k, updated_v) — cache extended with new K/V
        """
        
        # 1. PRE-NORM on new positions
        x_norm = self.ln1(x_new)
        
        # 2. INCREMENTAL ATTENTION with cached K/V
        attn_out, updated_k, updated_v = self._incremental_attention(
            x_norm, past_kv, mask_rows
        )
        
        # 3. RESIDUAL
        x_new = x_new + attn_out
        
        # 4. FFN on new positions only
        x_norm = self.ln2(x_new)
        ffn_out = self.ffn(x_norm)
        x_new = x_new + ffn_out
        
        return x_new, (updated_k, updated_v)
    
    def _attention_with_kv_extract(
        self,
        query : torch.Tensor,    # (B, S, d_model)
        key : torch.Tensor,      # (B, S, d_model)
        value : torch.Tensor,    # (B, S, d_model)
        mask : torch.Tensor,     # (S, S)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MHA and extract K, V tensors for caching.
        
        Uses the SAME weights as self.attn (nn.MultiheadAttention).
        We manually do the in_proj to get separate Q, K, V, then use
        F.scaled_dot_product_attention for the actual computation.
        """
        B, S, D = query.shape
        n_heads = self.config.n_heads
        d_head = self.d_head
        
        # Extract weights from nn.MHA (stored as single concatenated matrix)
        W = self.attn.in_proj_weight    # (3*D, D)
        b = self.attn.in_proj_bias      # (3*D,)
        
        # Project Q, K, V using the SAME weights as nn.MHA
        Q = F.linear(query, W[:D], b[:D])          # (B, S, D)
        K = F.linear(key, W[D:2*D], b[D:2*D])      # (B, S, D)
        V = F.linear(value, W[2*D:], b[2*D:])       # (B, S, D)
        
        # Reshape to multi-head: (B, n_heads, S, d_head)
        Q = Q.view(B, S, n_heads, d_head).transpose(1, 2)
        K = K.view(B, S, n_heads, d_head).transpose(1, 2)
        V = V.view(B, S, n_heads, d_head).transpose(1, 2)
        
        # Scaled dot-product attention with mask
        # PyTorch SDPA expects mask: (S, S) or (B, n_heads, S, S)
        # Our mask is (S, S) float with -inf for blocked positions
        # SDPA's attn_mask is ADDITIVE (added to QK^T), matching our format
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = mask,
            dropout_p = self.attn.dropout if self.training else 0.0,
        )  # (B, n_heads, S, d_head)
        
        # Reshape back: (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output projection (same weights as nn.MHA)
        attn_out = self.attn.out_proj(attn_out)
        
        return attn_out, K, V
    
    def _incremental_attention(
        self,
        x_new_norm : torch.Tensor,                           # (B, new_len, D) — pre-normed new positions
        past_kv : Tuple[torch.Tensor, torch.Tensor],        # (K_cached, V_cached) each (B, n_heads, cached_len, d_head)
        mask_rows : torch.Tensor,                             # (new_len, total_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental attention: new Q attends to [cached_K | new_K].
        
        This is where the O(n²) → O(n) savings happen.
        """
        B, new_len, D = x_new_norm.shape
        n_heads = self.config.n_heads
        d_head = self.d_head
        
        K_cached, V_cached = past_kv  # (B, n_heads, cached_len, d_head)
        
        # Extract weights
        W = self.attn.in_proj_weight    # (3*D, D)
        b = self.attn.in_proj_bias      # (3*D,)
        
        # Project ONLY the new positions
        Q_new = F.linear(x_new_norm, W[:D], b[:D])              # (B, new_len, D)
        K_new = F.linear(x_new_norm, W[D:2*D], b[D:2*D])        # (B, new_len, D)
        V_new = F.linear(x_new_norm, W[2*D:], b[2*D:])           # (B, new_len, D)
        
        # Reshape to multi-head
        Q_new = Q_new.view(B, new_len, n_heads, d_head).transpose(1, 2)  # (B, n_heads, new_len, d_head)
        K_new = K_new.view(B, new_len, n_heads, d_head).transpose(1, 2)
        V_new = V_new.view(B, new_len, n_heads, d_head).transpose(1, 2)
        
        # CONCATENATE with cached K/V → full context for attention
        K_full = torch.cat([K_cached, K_new], dim=2)  # (B, n_heads, total_len, d_head)
        V_full = torch.cat([V_cached, V_new], dim=2)  # (B, n_heads, total_len, d_head)
        
        # Attention: Q_new @ K_full^T → only compute new rows
        # mask_rows: (new_len, total_len) — hierarchical mask for new positions
        attn_out = F.scaled_dot_product_attention(
            Q_new, K_full, V_full,
            attn_mask = mask_rows,
            dropout_p = self.attn.dropout if self.training else 0.0,
        )  # (B, n_heads, new_len, d_head)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, new_len, D)
        
        # Output projection
        attn_out = self.attn.out_proj(attn_out)
        
        return attn_out, K_full, V_full


class HierarchicalWorldModel(nn.Module):
    """ MAIN HIERARCHICAL WORLD MODEL FOR ATARI100K PREDICTION  """
    def __init__(
        self,
        config : WorldModelConfig,
    ):
        """ SETUP """
        super().__init__()
        self.config = config
        
        # 1. EMBEDDING LAYER (HRVQ TOKENS + ACTIONS -> TRANSFORMER SEQUENCE)
        self.embedding = TokenEmbedding(config = config)
        
        # 2. TRANSFORMER BLOCKS (6 BLOCKS - HMASKING, ATTENTION, FFN)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 3. NORM 
        self.ln_final = nn.LayerNorm(normalized_shape = config.d_model)
        
        # 4. OUTPUT PREDICTION HEADS (384dim -> 256 codebook logits for each layer)
        self.headl0 = nn.Linear(in_features = config.d_model, out_features = config.num_codes)  # L0 (PHYSICS, COARSE)
        self.headl1 = nn.Linear(in_features = config.d_model, out_features = config.num_codes)  # L1 (MECHANICS, MEDIUM)
        self.headl2 = nn.Linear(in_features = config.d_model, out_features = config.num_codes)  # L2 (OBJECTS, FINE)

        # 5. MASK CACHING (HIERARCHICAL CAUSAL MASK) memory optimization
        self._cached_mask = None
        self._cached_mask_len = 0
        
    def _get_mask(
        self,
        seq_len : int,
        device : torch.device,
    )-> torch.tensor:
        """ GET OR CREATE CACHED HIERARCHICAL CAUSAL MASK TO SAVE COMPUTATION  """
        if self._cached_mask is None or self._cached_mask_len != seq_len or self._cached_mask.device != device:
            
            self._cached_mask = hierarchical_causal_mask(seq_len = seq_len, device = device)
            self._cached_mask_len = seq_len
        
        return self._cached_mask
    
    
    def forward(
        self,
        tokens : torch.tensor,      # SHAPE: (B, T, 3) - HRVQ TOKENS , 3 LAYERS [L0, L1, L2]
        actions : torch.tensor,     # SHAPE: (B, T) - ACTIONS
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:  
        """ FORWARD PASS THROUGH WORLD MODEL (ORIGINAL — UNCHANGED for WM training) """
        
        
        """ PRE PROCESSING"""
        # 1. UNPACK TUPLE, B = BATCH SIZE, T = TIMESTEPS
        B, T, _ = tokens.shape
        
        # 2. EMBEDDING LAYER
        x = self.embedding(tokens = tokens, actions = actions)  # (B, T*4, d_model)
        
        # 3. GET MASK 
        mask = self._get_mask(seq_len = x.size(1), device = x.device)  # (T*4, T*4)
        
        """ MAIN PASS THROUGH TRANSFORMER BLOCKS """
        for block in self.blocks:
            x = block(x, mask = mask)  # (B, T*4, d_model)
        
        # FINAL NORM
        x = self.ln_final(x)  # (B, T*4, d_model)
        
        """ EXTRACT PREDICTIONS FOR EACH LAYER [L0, L1, L2] FROM THE TRANSFORMER OUTPUT SEQUENCE (INTERLEAVED) """
        
        # ALL BATCHES, TOKENS BY POSITION, ALL FEATURES
        
        action_positions = x[:, 3::4, :]  # GRAB ACTION TOKENS (EVERY 4TH POSITION STARTING FROM 3)
        l0_positions = x[:, 0::4, :]      # GRAB L0 TOKENS (EVERY 4TH POSITION STARTING FROM 0)
        l1_positions = x[:, 1::4, :]      # GRAB L1 TOKENS (EVERY 4TH POSITION STARTING FROM 1)
        
        """ APPLY PREDICTION HEADS """
        logits_l0 = self.headl0(action_positions)  # (B, T, 256 codes) - PREDICT L0 (PHYSICS, COARSE)
        logits_l1 = self.headl1(l0_positions)      # (B, T, 256 codes) - PREDICT L1 (MECHANICS, MEDIUM)
        logits_l2 = self.headl2(l1_positions)      # (B, T, 256 codes) - PREDICT L2 (OBJECTS, FINE)
        
        return logits_l0, logits_l1, logits_l2

    def forward_with_kv(
        self,
        tokens : torch.Tensor,      # (B, T, 3)
        actions : torch.Tensor,     # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Full forward pass that ALSO RETURNS KV cache from each layer.
        
        Used for the INITIAL seed context processing during imagination.
        After this call, subsequent timesteps use forward_incremental().
        
        Returns:
            logits_l0:  (B, T, 256)
            logits_l1:  (B, T, 256)
            logits_l2:  (B, T, 256)
            kv_cache:   list of (K, V) tuples, one per transformer layer
                        each K, V is (B, n_heads, T*4, d_head)
        """
        B, T, _ = tokens.shape
        
        # Embedding
        x = self.embedding(tokens=tokens, actions=actions)  # (B, T*4, d_model)
        
        # Get mask for full sequence
        mask = self._get_mask(seq_len=x.size(1), device=x.device)
        
        # Forward through blocks, collecting KV cache
        kv_cache = []
        for block in self.blocks:
            x, kv = block.forward_with_kv(x, mask)
            kv_cache.append(kv)
        
        # Final norm
        x = self.ln_final(x)
        
        # Extract predictions (same logic as forward())
        action_positions = x[:, 3::4, :]
        l0_positions = x[:, 0::4, :]
        l1_positions = x[:, 1::4, :]
        
        logits_l0 = self.headl0(action_positions)
        logits_l1 = self.headl1(l0_positions)
        logits_l2 = self.headl2(l1_positions)
        
        return logits_l0, logits_l1, logits_l2, kv_cache, x # RETURN FULL HIDDEN STATES 
    
    def forward_incremental(
        self,
        tokens_new : torch.Tensor,   # (B, 1, 3) — single new timestep tokens
        actions_new : torch.Tensor,  # (B, 1)    — single new action
        kv_cache : list,              # list of (K, V) per layer
        cached_seq_len : int,         # how many positions are already cached
    ) -> Tuple[torch.Tensor, list]:
        """Incremental forward: process ONLY new positions using cached K/V.
        
        Embeds the new timestep (4 positions), runs through all transformer
        blocks using cached K/V, returns hidden states at new positions
        and the updated KV cache.
        
        Returns:
            x_new:          (B, 4, d_model) — hidden states for 4 new positions
            updated_cache:  list of (K, V) per layer (extended by 4 positions)
        """
        # 1. Embed ONLY the new timestep (4 positions with correct pos encoding)
        x_new = self.embedding.embed_partial(
            tokens_new, actions_new, start_pos=cached_seq_len
        )  # (B, 4, d_model)
        
        total_len = cached_seq_len + 4
        
        # 2. Get mask rows for the 4 new positions
        # We need the full mask at total_len, then extract the last 4 rows
        full_mask = self._get_mask(seq_len=total_len, device=x_new.device)
        mask_rows = full_mask[cached_seq_len:total_len, :total_len]  # (4, total_len)
        
        # 3. Forward through blocks incrementally
        updated_cache = []
        for layer_idx, block in enumerate(self.blocks):
            x_new, updated_kv = block.forward_incremental(
                x_new, kv_cache[layer_idx], mask_rows
            )
            updated_cache.append(updated_kv)
        
        # 4. Final norm (only on new positions)
        x_new = self.ln_final(x_new)  # (B, 4, d_model)
        
        return x_new, updated_cache
    
    def extract_logits_from_positions(
        self,
        x : torch.Tensor,   # (B, 4, d_model) — one timestep's hidden states
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract L0, L1, L2 logits from a single timestep's hidden states.
        
        Position mapping within one timestep:
            pos 0 = L0,  pos 1 = L1,  pos 2 = L2,  pos 3 = Action
            
        Prediction heads:
            headl0 reads ACTION position (pos 3) → predicts NEXT L0
            headl1 reads L0 position (pos 0) → predicts SAME-STEP L1
            headl2 reads L1 position (pos 1) → predicts SAME-STEP L2
        """
        logits_l0 = self.headl0(x[:, 3, :])  # (B, 256) from Action position
        logits_l1 = self.headl1(x[:, 0, :])  # (B, 256) from L0 position
        logits_l2 = self.headl2(x[:, 1, :])  # (B, 256) from L1 position
        
        return logits_l0, logits_l1, logits_l2
        
        
        
    
def hierarchical_loss(
    logits_l0 : torch.tensor,                           # PREDICTIONS FROM WORLD MODEL
    logits_l1 : torch.tensor,                           # PREDICTIONS FROM WORLD MODEL
    logits_l2 : torch.tensor,                           # PREDICTIONS FROM WORLD MODEL
    tokens : torch.tensor,                              # GROUND TRUTH HRVQ TOKENS
    layer_weights : list[float] = [1.0, 0.5, 0.1],      
) -> Tuple[torch.tensor, dict]:
    """ HIERARCHICAL LOSS FUNCTION  """
    num_codes = logits_l0.size(-1)  # 256
    
    """ 1. GROUND TRUTH FOR EACH LAYER (TARGET / MARK SCHEME) """
    groundtruth_l0 = tokens[:, 1:, 0]   # L0 TARGETS (PHYSICS, COARSE) - (B, T-1, LAYER 0 CODES)
    groundtruth_l1 = tokens[:, :, 1]    # L1 TARGETS (MECHANICS, MEDIUM) - (B, T, LAYER 1 CODES)
    groundtruth_l2 = tokens[:, :, 2]    # L2 TARGETS (OBJECTS, FINE) - (B, T, LAYER 2 CODES)
    
    logits_l0 = logits_l0[:, :-1, :]  # DROP OFF LAST TIME STEP (NO FUTURE)
    
    
    """ 2. CROSS ENTROPY LOSS FOR EACH LAYER (COMPUTE SEPARATELY) """
    crossentropy_l0 = F.cross_entropy(
        input = logits_l0.reshape(-1, num_codes),  # SQUASH (B + T, 256)
        target = groundtruth_l0.reshape(-1),      # FLATTEN 
    )    
    
    crossentropy_l1 = F.cross_entropy(
        input = logits_l1.reshape(-1, num_codes), # SQUASH (B + T, 256)
        target = groundtruth_l1.reshape(-1),
    )
    
    crossentropy_l2 = F.cross_entropy(
        input = logits_l2.reshape(-1, num_codes), # SQUASH (B + T, 256)
        target = groundtruth_l2.reshape(-1),      
    )
    
    
    """ 3. WEIGHTED HIERARCHICAL SUM OF LOSSES (HYPOTHESIS)"""
    total_loss = (
        layer_weights[0] * crossentropy_l0 +
        layer_weights[1] * crossentropy_l1 +
        layer_weights[2] * crossentropy_l2
    )
    
    """ 4. METRICS LOGGING (FOR WAND.B) """
    with torch.no_grad():
        accuracy_10 = (logits_l0.argmax(dim = -1) == groundtruth_l0).float().mean().item()  # TOP-1 ACCURACY FOR LAYER 0
        accuracy_11 = (logits_l1.argmax(dim = -1) == groundtruth_l1).float().mean().item()  # TOP-1 ACCURACY FOR LAYER 1
        accuracy_12 = (logits_l2.argmax(dim = -1) == groundtruth_l2).float().mean().item()  # TOP-1 ACCURACY FOR LAYER 2
        
    metrics = {
        'loss_total': total_loss.item(),
        'loss_l0': crossentropy_l0.item(),
        'loss_l1': crossentropy_l1.item(),
        'loss_l2': crossentropy_l2.item(),
        'accuracy_l0': accuracy_10,
        'accuracy_l1': accuracy_11,
        'accuracy_l2': accuracy_12,
    }
    
    return total_loss, metrics




if __name__ == "__main__":
    pass
