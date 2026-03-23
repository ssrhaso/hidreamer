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


# Constants
NUM_L0 = 4   # coarse 2×2 patches
NUM_L1 = 16  # mid 4×4 patches
NUM_L2 = 16  # fine 4×4 patches
TOKENS_PER_TIMESTEP = NUM_L0 + NUM_L1 + NUM_L2 + 1  # = 37

# Within-timestep positional ranges
L0_START, L0_END = 0, NUM_L0               # [0, 4)
L1_START, L1_END = NUM_L0, NUM_L0+NUM_L1  # [4, 20)
L2_START, L2_END = NUM_L0+NUM_L1, NUM_L0+NUM_L1+NUM_L2  # [20, 36)
ACTION_POS = TOKENS_PER_TIMESTEP - 1       # = 36


# Config
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


# Token Embedding
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

        # Separate embedding tables per level (independent codebooks)
        self.l0_embed  = nn.Embedding(config.num_codes_l0, D)
        self.l1_embed  = nn.Embedding(config.num_codes_l1, D)
        self.l2_embed  = nn.Embedding(config.num_codes_l2, D)
        self.act_embed = nn.Embedding(config.num_actions, D)
        self.level_embed = nn.Embedding(4, D)  # 4 types: L0, L1, L2, action
        self.patch_embed = nn.Embedding(NUM_L1, D)  # shared patch-index table, size 16 covers all levels
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

        e_l0  = self.l0_embed(tokens_l0)    # (B, T, 4,  D)
        e_l1  = self.l1_embed(tokens_l1)    # (B, T, 16, D)
        e_l2  = self.l2_embed(tokens_l2)    # (B, T, 16, D)
        e_act = self.act_embed(actions)      # (B, T,     D)

        e_l0  = e_l0  + lvl[0]
        e_l1  = e_l1  + lvl[1]
        e_l2  = e_l2  + lvl[2]
        e_act = e_act + lvl[3]

        e_l0  = e_l0  + self.patch_embed(torch.arange(NUM_L0, device=device))
        e_l1  = e_l1  + self.patch_embed(torch.arange(NUM_L1, device=device))
        e_l2  = e_l2  + self.patch_embed(torch.arange(NUM_L2, device=device))

        # Interleave into flat sequence (B, T*37, D) — per timestep: [L0_0..L0_3, L1_0..L1_15, L2_0..L2_15, A]
        e_act_exp = e_act.unsqueeze(2)  # (B, T, 1, D)
        timestep = torch.cat([e_l0, e_l1, e_l2, e_act_exp], dim=2)  # (B, T, 37, D)
        seq = timestep.reshape(B, T * TOKENS_PER_TIMESTEP, self.config.d_model)

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


# Hierarchical Causal Mask for 37-token sequences
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

    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    for t in range(num_t):
        base = t * P

        # L1: each patch sees all L0 of same timestep (triu incorrectly blocks them)
        for j in range(NUM_L1):
            l1_pos = base + L1_START + j
            for i in range(NUM_L0):
                mask[l1_pos, base + L0_START + i] = False

        # L2: each patch sees all L0 + all L1 of same timestep
        for k in range(NUM_L2):
            l2_pos = base + L2_START + k
            for i in range(NUM_L0):
                mask[l2_pos, base + L0_START + i] = False
            for j in range(NUM_L1):
                mask[l2_pos, base + L1_START + j] = False

        # Action: sees all 36 preceding tokens in same timestep
        act_pos = base + ACTION_POS
        for i in range(ACTION_POS):
            mask[act_pos, base + i] = False

    # L1, L2, Action cannot attend to past L1/L2 (only past L0 and past Action)
    for t_q in range(1, num_t):
        base_q = t_q * P
        for t_k in range(t_q):
            base_k = t_k * P
            restricted_rows = (
                list(range(base_q + L1_START, base_q + L1_END)) +
                list(range(base_q + L2_START, base_q + L2_END)) +
                [base_q + ACTION_POS]
            )
            blocked_cols = (
                list(range(base_k + L1_START, base_k + L1_END)) +
                list(range(base_k + L2_START, base_k + L2_END))
            )
            for row in restricted_rows:
                for col in blocked_cols:
                    mask[row, col] = True

    return mask.float().masked_fill(mask, float('-inf'))


# Spatial World Model
class SpatialHierarchicalWorldModel(nn.Module):
    """
    Hierarchical World Model with 37-token spatial layout.

    Training forward: teacher-forced next-token prediction.
    Imagination: predict one timestep at a time (L0→L1→L2 in 3 parallel steps).
    """

    def __init__(self, config: SpatialWorldModelConfig):
        super().__init__()
        self.config = config

        self.embedding = SpatialTokenEmbedding(config)
        self.blocks = nn.ModuleList([
            _SpatialTransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head_l0 = nn.Linear(config.d_model, config.num_codes_l0)
        self.head_l1 = nn.Linear(config.d_model, config.num_codes_l1)
        self.head_l2 = nn.Linear(config.d_model, config.num_codes_l2)
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
        """Teacher-forced next-token prediction. Returns hidden, logits_l0/l1/l2."""
        B, T, _ = tokens_l0.shape
        P = TOKENS_PER_TIMESTEP
        seq_len = T * P

        x = self.embedding(tokens_l0, tokens_l1, tokens_l2, actions)
        mask = self._get_mask(seq_len, x.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        x_ts = x.reshape(B, T, P, self.config.d_model)  # (B, T, 37, D)

        # Each patch is predicted by the hidden state of the token immediately before it.
        # L0_0 at t>0 uses the action hidden from t-1; t=0 rows stay zero (skipped in loss).
        l0_src = torch.zeros(B, T, NUM_L0, self.config.d_model, device=x.device)
        if T > 1:
            l0_src[:, 1:, 0, :] = x_ts[:, :-1, ACTION_POS, :]
        for j in range(1, NUM_L0):
            l0_src[:, :, j, :] = x_ts[:, :, L0_START + j - 1, :]
        logits_l0 = self.head_l0(l0_src)

        # L1_0 conditioned on L0_3; L1_j on L1_{j-1}.
        l1_src = torch.zeros(B, T, NUM_L1, self.config.d_model, device=x.device)
        l1_src[:, :, 0, :] = x_ts[:, :, L0_END - 1, :]
        for j in range(1, NUM_L1):
            l1_src[:, :, j, :] = x_ts[:, :, L1_START + j - 1, :]
        logits_l1 = self.head_l1(l1_src)

        # L2_0 conditioned on L1_15; L2_j on L2_{j-1}.
        l2_src = torch.zeros(B, T, NUM_L2, self.config.d_model, device=x.device)
        l2_src[:, :, 0, :] = x_ts[:, :, L1_END - 1, :]
        for j in range(1, NUM_L2):
            l2_src[:, :, j, :] = x_ts[:, :, L2_START + j - 1, :]
        logits_l2 = self.head_l2(l2_src)

        return {
            'hidden':    x,
            'logits_l0': logits_l0,
            'logits_l1': logits_l1,
            'logits_l2': logits_l2,
        }

    def compute_loss(
        self,
        logits_l0: torch.Tensor,
        logits_l1: torch.Tensor,
        logits_l2: torch.Tensor,
        tokens_l0: torch.Tensor,
        tokens_l1: torch.Tensor,
        tokens_l2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted cross-entropy losses. t=0 L0 predictions skipped (no prior action)."""
        B, T, _ = tokens_l0.shape

        def ce(logits, targets, skip_t0=False):
            if skip_t0:
                logits  = logits[:, 1:]
                targets = targets[:, 1:]
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
        """Forward a context window and return (hidden, kv_cache)."""
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
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List]:
        """Level-parallel one-step imagination: L0 from action hidden, L1 from L0, L2 from L0+L1."""
        B = hidden_last.size(0)
        device = hidden_last.device

        # Predict L0 patches from the last action hidden state.
        # All L0 patches share the same conditioning hidden (level-parallel approximation).
        l0_logits = self.head_l0(hidden_last)                           # (B, num_codes_l0)
        pred_l0_0 = l0_logits.argmax(dim=-1)                           # (B,)
        pred_l0 = pred_l0_0.unsqueeze(1).expand(B, NUM_L0)             # (B, 4)

        # Predict L1 patches conditioned on mean L0 embedding.
        l0_ctx = self.embedding.l0_embed(pred_l0).mean(dim=1)          # (B, D)
        l1_logits = self.head_l1(l0_ctx)                               # (B, num_codes_l1)
        pred_l1 = l1_logits.argmax(dim=-1).unsqueeze(1).expand(B, NUM_L1)  # (B, 16)

        # Predict L2 patches conditioned on mean L0+L1 embedding.
        l1_ctx = self.embedding.l1_embed(pred_l1).mean(dim=1)          # (B, D)
        l2_logits = self.head_l2(l0_ctx + l1_ctx)                      # (B, num_codes_l2)
        pred_l2 = l2_logits.argmax(dim=-1).unsqueeze(1).expand(B, NUM_L2)  # (B, 16)

        pred_tokens = {'l0': pred_l0, 'l1': pred_l1, 'l2': pred_l2}
        return pred_tokens, hidden_last, kv_cache

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Transformer block for SpatialWorldModelConfig
class _SpatialTransformerBlock(nn.Module):

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
        """Forward pass that also returns K, V tensors for the KV-cache.
        Accesses nn.MultiheadAttention's in_proj_weight directly (packed Q,K,V projection)."""
        B, S, D = x.shape
        n_heads = self.n_heads
        d_head = self.d_head
        H = n_heads

        x_norm = self.ln1(x)
        W = self.attn.in_proj_weight   # (3D, D) — packed Q, K, V projections
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
