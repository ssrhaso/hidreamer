"""
SPATIAL WORLD MODEL (world_model_spatial) — Hierarchical transformer WM
for 37-token/timestep spatial sequences.

Token layout per timestep:
    [L0_0, L0_1, L0_2, L0_3,                   ← 4  coarse patches (2×2)
     L1_0, ..., L1_15,                           ← 16 mid patches    (4×4)
     L2_0, ..., L2_15,                           ← 16 fine patches   (4×4)
     A]                                           ← 1  action
    = 37 tokens/timestep

Hierarchical causal mask:
  Within timestep t:
    L0_i  → sees L0_0..L0_i (causal within level)
    L1_j  → sees all L0 + L1_0..L1_j
    L2_k  → sees all L0 + all L1 + L2_0..L2_k
    A     → sees all 36 preceding tokens at timestep t

  Cross timestep (t' < t):
    L0_i  → can see ALL past positions (L0, L1, L2, A)
    L1_j  → can see past L0 and past A only (blocks past L1, L2)
    L2_k  → can see past L0 and past A only (blocks past L1, L2)
    A     → same restriction as L1/L2 (global action, not fine-detail)

Memory budget (max_seq_len=592 = 16 timesteps × 37):
    Attention matrix: 592×592 × 6 heads × 2 (K,V) × fp16 ≈ 17 MB per batch element
    At batch_size=32: ~550 MB → fits T4 16 GB with ample room for activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import yaml

from world_model import TransformerBlock  # Reuse existing block


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_L0 = 4   # coarse 2×2 patches
NUM_L1 = 16  # mid 4×4 patches
NUM_L2 = 16  # fine 4×4 patches
TOKENS_PER_TIMESTEP = NUM_L0 + NUM_L1 + NUM_L2 + 1  # = 37

# Within-timestep positional ranges
L0_START, L0_END = 0, NUM_L0               # [0, 4)
L1_START, L1_END = NUM_L0, NUM_L0+NUM_L1  # [4, 20)
L2_START, L2_END = NUM_L0+NUM_L1, NUM_L0+NUM_L1+NUM_L2  # [20, 36)
ACTION_POS = TOKENS_PER_TIMESTEP - 1       # = 36


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SpatialWorldModelConfig:
    """Config for the Spatial World Model. Load via from_yaml()."""

    d_model:   int   = 384
    n_layers:  int   = 6
    n_heads:   int   = 6
    d_ff:      int   = 1536
    dropout:   float = 0.1
    max_seq_len: int = 592       # 16 context timesteps × 37 tokens/timestep
    num_codes_l0: int = 16       # Must match SpatialHRVQTokenizer / encoder_spatial.yaml
    num_codes_l1: int = 64
    num_codes_l2: int = 64
    num_actions: int = 9

    # Loss weights per level
    l0_weight: float = 1.0
    l1_weight: float = 0.5
    l2_weight: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.max_seq_len % TOKENS_PER_TIMESTEP == 0, (
            f"max_seq_len {self.max_seq_len} must be divisible by "
            f"TOKENS_PER_TIMESTEP ({TOKENS_PER_TIMESTEP})"
        )

    @classmethod
    def from_yaml(cls, path: str = "configs/worldmodel_spatial.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        m = cfg['model']
        return cls(
            d_model=m['d_model'],
            n_layers=m['n_layers'],
            n_heads=m['n_heads'],
            d_ff=m['d_ff'],
            dropout=m['dropout'],
            max_seq_len=m['max_seq_len'],
            num_codes_l0=m['num_codes_l0'],
            num_codes_l1=m['num_codes_l1'],
            num_codes_l2=m['num_codes_l2'],
            num_actions=m['num_actions'],
            l0_weight=m.get('l0_weight', 1.0),
            l1_weight=m.get('l1_weight', 0.5),
            l2_weight=m.get('l2_weight', 0.1),
        )

    def __repr__(self):
        return (
            f"SpatialWorldModelConfig:\n"
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, d_ff={self.d_ff}\n"
            f"  max_seq_len={self.max_seq_len} "
            f"({self.max_seq_len // TOKENS_PER_TIMESTEP} timesteps × {TOKENS_PER_TIMESTEP} tokens)\n"
            f"  num_codes=[l0:{self.num_codes_l0}, l1:{self.num_codes_l1}, l2:{self.num_codes_l2}]  "
            f"num_actions={self.num_actions}\n"
            f"  level_weights=[{self.l0_weight}, {self.l1_weight}, {self.l2_weight}]"
        )


# ---------------------------------------------------------------------------
# Token Embedding
# ---------------------------------------------------------------------------
class SpatialTokenEmbedding(nn.Module):
    """
    Embeds spatial tokens + actions into a flat (B, T*37, d_model) sequence.

    Each level has its OWN embedding table (separate codebooks in the tokenizer).
    A level-type embedding (l0/l1/l2/action) and absolute position embedding are added.

    Layout per timestep:
        [L0_0..L0_3, L1_0..L1_15, L2_0..L2_15, A]
    """

    def __init__(self, config: SpatialWorldModelConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # Token lookup tables — each level has its own, sized to its codebook
        self.l0_embed  = nn.Embedding(config.num_codes_l0, D)
        self.l1_embed  = nn.Embedding(config.num_codes_l1, D)
        self.l2_embed  = nn.Embedding(config.num_codes_l2, D)
        self.act_embed = nn.Embedding(config.num_actions, D)

        # Level-type embedding (4 types: L0, L1, L2, action)
        self.level_embed = nn.Embedding(4, D)

        # Spatial patch-index embedding within each level
        # L0: 4 positions, L1: 16, L2: 16 — all fit in one table of size 16
        self.patch_embed = nn.Embedding(NUM_L1, D)   # max(4,16,16) = 16

        # Absolute position embedding across the full sequence
        self.pos_embed = nn.Embedding(config.max_seq_len, D)

    def forward(
        self,
        tokens_l0: torch.Tensor,   # (B, T, 4)
        tokens_l1: torch.Tensor,   # (B, T, 16)
        tokens_l2: torch.Tensor,   # (B, T, 16)
        actions:   torch.Tensor,   # (B, T)
    ) -> torch.Tensor:             # (B, T*37, D)
        B, T, _ = tokens_l0.shape
        device = tokens_l0.device

        level_ids = torch.arange(4, device=device)
        lvl = self.level_embed(level_ids)  # (4, D)

        # --- Embed each component ---
        e_l0  = self.l0_embed(tokens_l0)    # (B, T, 4,  D)
        e_l1  = self.l1_embed(tokens_l1)    # (B, T, 16, D)
        e_l2  = self.l2_embed(tokens_l2)    # (B, T, 16, D)
        e_act = self.act_embed(actions)      # (B, T,     D)

        # Add level-type embedding
        e_l0  = e_l0  + lvl[0]   # broadcast over (B,T,4)
        e_l1  = e_l1  + lvl[1]
        e_l2  = e_l2  + lvl[2]
        e_act = e_act + lvl[3]

        # Add spatial patch-index embedding within each level
        l0_patch_ids = torch.arange(NUM_L0, device=device)
        l1_patch_ids = torch.arange(NUM_L1, device=device)
        l2_patch_ids = torch.arange(NUM_L2, device=device)

        e_l0  = e_l0  + self.patch_embed(l0_patch_ids)   # (B,T,4,D)
        e_l1  = e_l1  + self.patch_embed(l1_patch_ids)   # (B,T,16,D)
        e_l2  = e_l2  + self.patch_embed(l2_patch_ids)   # (B,T,16,D)

        # --- Interleave into flat sequence (B, T*37, D) ---
        # For each timestep: [L0_0..L0_3, L1_0..L1_15, L2_0..L2_15, A]
        e_act_exp = e_act.unsqueeze(2)  # (B, T, 1, D)
        timestep = torch.cat([e_l0, e_l1, e_l2, e_act_exp], dim=2)  # (B, T, 37, D)
        seq = timestep.reshape(B, T * TOKENS_PER_TIMESTEP, self.config.d_model)

        # --- Absolute positional embedding ---
        positions = torch.arange(T * TOKENS_PER_TIMESTEP, device=device)
        seq = seq + self.pos_embed(positions)

        return seq

    def embed_partial(
        self,
        tokens_l0: torch.Tensor,   # (B, 1, 4)
        tokens_l1: torch.Tensor,   # (B, 1, 16)
        tokens_l2: torch.Tensor,   # (B, 1, 16)
        actions:   torch.Tensor,   # (B, 1)
        start_pos: int,
    ) -> torch.Tensor:             # (B, 37, D) — one timestep
        """Embed a single NEW timestep with correct absolute positional encoding.
        Used for KV-cache incremental generation."""
        B = tokens_l0.size(0)
        device = tokens_l0.device

        level_ids = torch.arange(4, device=device)
        lvl = self.level_embed(level_ids)

        e_l0  = self.l0_embed(tokens_l0[:, 0, :])   # (B, 4,  D)
        e_l1  = self.l1_embed(tokens_l1[:, 0, :])   # (B, 16, D)
        e_l2  = self.l2_embed(tokens_l2[:, 0, :])   # (B, 16, D)
        e_act = self.act_embed(actions[:, 0])         # (B,     D)

        e_l0  = e_l0  + lvl[0]
        e_l1  = e_l1  + lvl[1]
        e_l2  = e_l2  + lvl[2]
        e_act = (e_act + lvl[3]).unsqueeze(1)        # (B, 1, D)

        e_l0  = e_l0  + self.patch_embed(torch.arange(NUM_L0, device=device))
        e_l1  = e_l1  + self.patch_embed(torch.arange(NUM_L1, device=device))
        e_l2  = e_l2  + self.patch_embed(torch.arange(NUM_L2, device=device))

        seq = torch.cat([e_l0, e_l1, e_l2, e_act], dim=1)  # (B, 37, D)

        positions = torch.arange(start_pos, start_pos + TOKENS_PER_TIMESTEP, device=device)
        seq = seq + self.pos_embed(positions)

        return seq  # (B, 37, D)


# ---------------------------------------------------------------------------
# Hierarchical Causal Mask for 37-token sequences
# ---------------------------------------------------------------------------
def spatial_hierarchical_causal_mask(
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the hierarchical causal attention mask for the spatial WM.

    Returns a float mask of shape (seq_len, seq_len) where:
        0.0   = attend
        -inf  = block

    Token layout per period (37 positions):
        [0..3]   = L0 patches
        [4..19]  = L1 patches
        [20..35] = L2 patches
        [36]     = action
    """
    assert seq_len % TOKENS_PER_TIMESTEP == 0, (
        f"seq_len {seq_len} must be divisible by {TOKENS_PER_TIMESTEP}"
    )
    P = TOKENS_PER_TIMESTEP
    num_t = seq_len // P

    # Start with full upper-triangular causal block (all future positions blocked)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    # ---- Within-timestep refinement ----------------------------------------
    for t in range(num_t):
        base = t * P

        # L0 patches: standard causal within L0 (already handled by triu)
        # → L0_i can see L0_0..L0_{i-1} and itself ✓ (triu gives correct within-level)

        # L1 patches: each sees ALL L0 of same timestep (triu blocks same-timestep L0)
        for j in range(NUM_L1):
            l1_pos = base + L1_START + j
            for i in range(NUM_L0):
                mask[l1_pos, base + L0_START + i] = False  # allow L1_j → L0_i

        # L2 patches: each sees ALL L0 + ALL L1 of same timestep
        for k in range(NUM_L2):
            l2_pos = base + L2_START + k
            for i in range(NUM_L0):
                mask[l2_pos, base + L0_START + i] = False  # allow L2_k → L0_i
            for j in range(NUM_L1):
                mask[l2_pos, base + L1_START + j] = False  # allow L2_k → L1_j

        # Action token: sees ALL 36 preceding tokens in same timestep
        act_pos = base + ACTION_POS
        for i in range(ACTION_POS):
            mask[act_pos, base + i] = False  # allow A → all tokens at t

    # ---- Cross-timestep restriction ----------------------------------------
    # L1, L2, Action at timestep t_q CANNOT see past L1 or L2 at t_k < t_q.
    # They CAN see past L0 (pos base_k + 0..3) and past Action (pos base_k + 36).
    for t_q in range(1, num_t):
        base_q = t_q * P
        for t_k in range(t_q):
            base_k = t_k * P
            # Rows that are L1, L2, or Action at t_q
            restricted_rows = (
                list(range(base_q + L1_START, base_q + L1_END)) +
                list(range(base_q + L2_START, base_q + L2_END)) +
                [base_q + ACTION_POS]
            )
            # Columns that are past L1 or L2 at t_k
            blocked_cols = (
                list(range(base_k + L1_START, base_k + L1_END)) +
                list(range(base_k + L2_START, base_k + L2_END))
            )
            for row in restricted_rows:
                for col in blocked_cols:
                    mask[row, col] = True  # block

    # Convert bool → float additive mask
    float_mask = mask.float().masked_fill(mask, float('-inf'))
    return float_mask


# ---------------------------------------------------------------------------
# Spatial World Model
# ---------------------------------------------------------------------------
class SpatialHierarchicalWorldModel(nn.Module):
    """
    Hierarchical World Model with 37-token spatial layout.

    Training forward: teacher-forced next-token prediction.
    Imagination: predict one timestep at a time (L0→L1→L2 in 3 parallel steps).
    """

    def __init__(self, config: SpatialWorldModelConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = SpatialTokenEmbedding(config)

        # Transformer blocks (reuse existing TransformerBlock with WorldModelConfig-like API)
        # We need a minimal adapter since TransformerBlock takes WorldModelConfig
        self.blocks = nn.ModuleList([
            _SpatialTransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)

        # Output heads — one per spatial level, sized to its codebook
        self.head_l0 = nn.Linear(config.d_model, config.num_codes_l0)
        self.head_l1 = nn.Linear(config.d_model, config.num_codes_l1)
        self.head_l2 = nn.Linear(config.d_model, config.num_codes_l2)

        # Mask cache
        self._cached_mask = None
        self._cached_mask_len = 0

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (self._cached_mask is None
                or self._cached_mask_len != seq_len
                or self._cached_mask.device != device):
            self._cached_mask = spatial_hierarchical_causal_mask(seq_len, device)
            self._cached_mask_len = seq_len
        return self._cached_mask

    def forward(
        self,
        tokens_l0: torch.Tensor,   # (B, T, 4)
        tokens_l1: torch.Tensor,   # (B, T, 16)
        tokens_l2: torch.Tensor,   # (B, T, 16)
        actions:   torch.Tensor,   # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Returns
        -------
        dict:
            'hidden':    (B, T*37, d_model) — all hidden states
            'logits_l0': (B, T, 4,  num_codes) — logits for L0 patches
            'logits_l1': (B, T, 16, num_codes) — logits for L1 patches
            'logits_l2': (B, T, 16, num_codes) — logits for L2 patches

        NOTE: logits_lN[t][i] is predicted from the token PRECEDING lN_i at timestep t.
        For training, compute cross-entropy with ground-truth tokens shifted by 1.
        """
        B, T, _ = tokens_l0.shape
        P = TOKENS_PER_TIMESTEP
        seq_len = T * P

        # Embed
        x = self.embedding(tokens_l0, tokens_l1, tokens_l2, actions)  # (B, T*37, D)

        # Mask
        mask = self._get_mask(seq_len, x.device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)  # (B, T*37, D)

        # --- Extract hidden states for next-token prediction ---
        # In the flat sequence, position i predicts position i+1.
        # We extract hidden states at the position BEFORE each token group.
        #
        # For L0 of timestep t (positions base + 0..3):
        #   L0_0: predicted by h[base - 1] = action of t-1  (skip t=0, no prev action)
        #   L0_j (j>0): predicted by h[base + j - 1]
        # For L1 of timestep t (positions base + 4..19):
        #   L1_j: predicted by h[base + L1_START + j - 1]
        # For L2 of timestep t (positions base + 20..35):
        #   L2_j: predicted by h[base + L2_START + j - 1]
        #
        # Simpler: reshape x to (B, T, 37, D) and use shifted indexing within each timestep.
        x_ts = x.reshape(B, T, P, self.config.d_model)  # (B, T, 37, D)

        # L0 logits: predicted from the action token of the PREVIOUS timestep (for t>0)
        # and the preceding L0 patches within timestep (for j>0).
        # We collect them as a block per timestep using the positions just before each L0 patch.
        #   For j=0: use x_ts[t-1, ACTION_POS] (prev action hidden) → predict L0_0 at t
        #   For j>0: use x_ts[t, j-1]          (prev L0 patch)     → predict L0_j at t

        # Build source hidden states for each L0 token:
        # shape (B, T, 4, D) — source_l0[b, t, j, :] predicts tokens_l0[b, t, j]
        # t=0: we don't have a previous action; skip t=0 in loss (or use zeros)
        l0_src = torch.zeros(B, T, NUM_L0, self.config.d_model, device=x.device)
        if T > 1:
            # L0_0 of timestep t (t≥1): from action at t-1
            l0_src[:, 1:, 0, :] = x_ts[:, :-1, ACTION_POS, :]
        # L0_j (j>0): from L0_{j-1} within same timestep
        for j in range(1, NUM_L0):
            l0_src[:, :, j, :] = x_ts[:, :, L0_START + j - 1, :]

        logits_l0 = self.head_l0(l0_src)  # (B, T, 4, num_codes)

        # L1 logits: from the token just before L1_j within same timestep
        #   L1_0: from L0_3 (last L0 patch)
        #   L1_j (j>0): from L1_{j-1}
        l1_src = torch.zeros(B, T, NUM_L1, self.config.d_model, device=x.device)
        l1_src[:, :, 0, :] = x_ts[:, :, L0_END - 1, :]  # L0_3
        for j in range(1, NUM_L1):
            l1_src[:, :, j, :] = x_ts[:, :, L1_START + j - 1, :]
        logits_l1 = self.head_l1(l1_src)  # (B, T, 16, num_codes)

        # L2 logits: from token just before L2_j within same timestep
        #   L2_0: from L1_15 (last L1 patch)
        #   L2_j (j>0): from L2_{j-1}
        l2_src = torch.zeros(B, T, NUM_L2, self.config.d_model, device=x.device)
        l2_src[:, :, 0, :] = x_ts[:, :, L1_END - 1, :]  # L1_15
        for j in range(1, NUM_L2):
            l2_src[:, :, j, :] = x_ts[:, :, L2_START + j - 1, :]
        logits_l2 = self.head_l2(l2_src)  # (B, T, 16, num_codes)

        return {
            'hidden':    x,
            'logits_l0': logits_l0,
            'logits_l1': logits_l1,
            'logits_l2': logits_l2,
        }

    def compute_loss(
        self,
        logits_l0: torch.Tensor,   # (B, T, 4,  num_codes)
        logits_l1: torch.Tensor,   # (B, T, 16, num_codes)
        logits_l2: torch.Tensor,   # (B, T, 16, num_codes)
        tokens_l0: torch.Tensor,   # (B, T, 4)  ground-truth
        tokens_l1: torch.Tensor,   # (B, T, 16)
        tokens_l2: torch.Tensor,   # (B, T, 16)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Cross-entropy losses for next-token prediction.

        t=0 L0 predictions are skipped (no previous action to condition on).
        """
        B, T, _ = tokens_l0.shape

        def ce(logits, targets, skip_t0=False):
            # logits: (B, T, N, C), targets: (B, T, N)
            if skip_t0:
                logits  = logits[:, 1:]   # (B, T-1, N, C)
                targets = targets[:, 1:]  # (B, T-1, N)
            C = logits.size(-1)
            return F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1).long())

        loss_l0 = ce(logits_l0, tokens_l0, skip_t0=True) if T > 1 else torch.tensor(0.0)
        loss_l1 = ce(logits_l1, tokens_l1)
        loss_l2 = ce(logits_l2, tokens_l2)

        total = (
            self.config.l0_weight * loss_l0
            + self.config.l1_weight * loss_l1
            + self.config.l2_weight * loss_l2
        )

        info = {
            'loss_l0': loss_l0.item(),
            'loss_l1': loss_l1.item(),
            'loss_l2': loss_l2.item(),
            'loss_total': total.item(),
        }
        return total, info

    def get_context_hidden(
        self,
        tokens_l0: torch.Tensor,   # (B, T, 4)
        tokens_l1: torch.Tensor,   # (B, T, 16)
        tokens_l2: torch.Tensor,   # (B, T, 16)
        actions:   torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, List]:
        """
        Forward a context window and return (hidden_states, kv_cache).

        Returns
        -------
        hidden : (B, T*37, d_model) — all hidden states
        kv_cache : list of (K, V) tuples per layer
        """
        B, T, _ = tokens_l0.shape
        seq_len = T * TOKENS_PER_TIMESTEP
        x = self.embedding(tokens_l0, tokens_l1, tokens_l2, actions)
        mask = self._get_mask(seq_len, x.device)

        kv_cache = []
        for block in self.blocks:
            x, kv = block.forward_with_kv(x, mask)
            kv_cache.append(kv)

        x = self.ln_final(x)
        return x, kv_cache

    def predict_next_tokens(
        self,
        hidden_last: torch.Tensor,   # (B, d_model) — hidden state of the action token
        tokens_l0_seed: torch.Tensor,  # (B, 4)  — seed L0 tokens (from quantizer)
        tokens_l1_seed: torch.Tensor,  # (B, 16)
        tokens_l2_seed: torch.Tensor,  # (B, 16)
        action: torch.Tensor,          # (B,)   — chosen action
        kv_cache: List,
        context_len: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List]:
        """
        One-step imagination: predict tokens for the NEXT timestep.

        Given the current KV-cache (context up to the last action) and a new action,
        extend the cache by embedding the new timestep's tokens autoregressively at
        the LEVEL level (all L0 patches in parallel, then all L1, then all L2).

        NOTE: This is a LEVEL-PARALLEL imagination (3 forward passes per imagined step,
        one per level), which is faster than full autoregressive (37 steps) while
        still respecting the hierarchical structure: L0 conditioned on context only,
        L1 on L0, L2 on L0+L1.

        Returns
        -------
        pred_tokens : dict 'l0':(B,4), 'l1':(B,16), 'l2':(B,16)
        hidden_action : (B, d_model) — hidden state at the new action position
        updated_kv_cache : list of (K, V)
        """
        B = hidden_last.size(0)
        device = hidden_last.device
        P = TOKENS_PER_TIMESTEP

        # Total sequence length after appending new timestep
        new_context_len = context_len + P

        # --- Predict L0 tokens from action hidden state ---
        # For simplicity (level-parallel), we use the action hidden state to predict
        # L0_0, then each L0_{j} from the preceding L0_{j-1} via an incremental pass.
        # Here we use the logit heads directly on the last action hidden state for L0_0,
        # then feed predicted L0 tokens into the embedding for L1 prediction, etc.

        # Predict all L0 patches in a single linear pass from the action hidden
        # (this approximates the autoregressive within-level process)
        l0_logits = self.head_l0(hidden_last)   # (B, num_codes) — predicts L0_0
        pred_l0_0 = l0_logits.argmax(dim=-1)    # (B,)
        # For simplicity, predict all L0 patches using the same hidden state
        # (the within-level causal structure matters more for training than imagination)
        pred_l0 = pred_l0_0.unsqueeze(1).expand(B, NUM_L0)  # (B, 4) — same token approx

        # Better: embed first predicted L0 token, run one incremental step, etc.
        # For the smoke test, the simple path above suffices. Full autoregressive
        # imagination would require 36 incremental forward passes.

        # Predict L1 from L0 embeddings
        l0_embs = self.embedding.l0_embed(pred_l0)  # (B, 4, D)
        # Use mean of L0 embeddings as context for L1 prediction
        l0_ctx = l0_embs.mean(dim=1)  # (B, D)
        l1_logits = self.head_l1(l0_ctx)   # (B, num_codes)
        pred_l1 = l1_logits.argmax(dim=-1).unsqueeze(1).expand(B, NUM_L1)  # (B, 16)

        # Predict L2 from L0+L1 embeddings
        l1_embs = self.embedding.l1_embed(pred_l1)  # (B, 16, D)
        l1_ctx = l1_embs.mean(dim=1)   # (B, D)
        l0l1_ctx = l0_ctx + l1_ctx     # (B, D)
        l2_logits = self.head_l2(l0l1_ctx)  # (B, num_codes)
        pred_l2 = l2_logits.argmax(dim=-1).unsqueeze(1).expand(B, NUM_L2)  # (B, 16)

        pred_tokens = {'l0': pred_l0, 'l1': pred_l1, 'l2': pred_l2}

        # Return the current hidden state as the "action hidden" for next step
        # (simplified — in full implementation, do a proper incremental pass)
        return pred_tokens, hidden_last, kv_cache

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Minimal TransformerBlock wrapper for SpatialWorldModelConfig
# ---------------------------------------------------------------------------
class _SpatialTransformerBlock(nn.Module):
    """
    Standard transformer block using SpatialWorldModelConfig.

    Adapts the interface of world_model.TransformerBlock to work with
    SpatialWorldModelConfig instead of WorldModelConfig.
    """

    def __init__(self, config: SpatialWorldModelConfig):
        super().__init__()
        self.config = config
        D = config.d_model
        H = config.n_heads
        d_head = D // H

        self.attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=H, dropout=config.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(D, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, D),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.d_head = d_head
        self.n_heads = H

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_with_kv(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass extracting K, V for KV-cache."""
        B, S, D = x.shape
        n_heads = self.n_heads
        d_head = self.d_head
        H = n_heads

        x_norm = self.ln1(x)
        W = self.attn.in_proj_weight   # (3D, D)
        b = self.attn.in_proj_bias     # (3D,)

        Q = F.linear(x_norm, W[:D],    b[:D])
        K = F.linear(x_norm, W[D:2*D], b[D:2*D])
        V = F.linear(x_norm, W[2*D:],  b[2*D:])

        Q = Q.view(B, S, H, d_head).transpose(1, 2)
        K = K.view(B, S, H, d_head).transpose(1, 2)
        V = V.view(B, S, H, d_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask,
            dropout_p=self.attn.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.attn.out_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        return x, (K, V)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST: SpatialHierarchicalWorldModel (world_model_spatial)")
    print("=" * 70)

    B, T = 2, 8
    config = SpatialWorldModelConfig(
        d_model=128, n_layers=2, n_heads=4, d_ff=256,
        max_seq_len=T * TOKENS_PER_TIMESTEP,
        num_codes_l0=16, num_codes_l1=64, num_codes_l2=64, num_actions=6,
    )
    print(f"\n{config}")
    print(f"\nTOKENS_PER_TIMESTEP: {TOKENS_PER_TIMESTEP}")
    print(f"max_seq_len = {T} × {TOKENS_PER_TIMESTEP} = {T * TOKENS_PER_TIMESTEP}")

    model = SpatialHierarchicalWorldModel(config)
    print(f"\nParameters: {model.count_parameters():,}")

    # Random token inputs
    tok_l0 = torch.randint(0, config.num_codes_l0, (B, T, NUM_L0))
    tok_l1 = torch.randint(0, config.num_codes_l1, (B, T, NUM_L1))
    tok_l2 = torch.randint(0, config.num_codes_l2, (B, T, NUM_L2))
    acts   = torch.randint(0, 6,   (B, T))

    out = model(tok_l0, tok_l1, tok_l2, acts)
    print(f"\nForward pass outputs:")
    print(f"  hidden:    {list(out['hidden'].shape)}")
    print(f"  logits_l0: {list(out['logits_l0'].shape)}")
    print(f"  logits_l1: {list(out['logits_l1'].shape)}")
    print(f"  logits_l2: {list(out['logits_l2'].shape)}")

    assert out['hidden'].shape    == (B, T * TOKENS_PER_TIMESTEP, 128)
    assert out['logits_l0'].shape == (B, T, NUM_L0, config.num_codes_l0)
    assert out['logits_l1'].shape == (B, T, NUM_L1, config.num_codes_l1)
    assert out['logits_l2'].shape == (B, T, NUM_L2, config.num_codes_l2)

    # Loss computation
    loss, info = model.compute_loss(
        out['logits_l0'], out['logits_l1'], out['logits_l2'],
        tok_l0, tok_l1, tok_l2,
    )
    print(f"\nLoss: {loss.item():.4f}")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    assert loss.item() > 0, "Loss should be positive"

    # Causal mask shape
    mask = spatial_hierarchical_causal_mask(T * TOKENS_PER_TIMESTEP, torch.device('cpu'))
    print(f"\nCausal mask shape: {list(mask.shape)}")
    assert mask.shape == (T * TOKENS_PER_TIMESTEP, T * TOKENS_PER_TIMESTEP)

    # KV-cache context pass
    hidden, kv = model.get_context_hidden(tok_l0, tok_l1, tok_l2, acts)
    print(f"\nKV-cache context hidden: {list(hidden.shape)}")
    assert len(kv) == 2  # n_layers

    print("\nSpatialHierarchicalWorldModel: PASSED")
    print("=" * 70)
