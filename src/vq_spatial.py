"""
SPATIAL HRVQ TOKENIZER (vq_spatial) — Per-patch VQ for spatial token grids.

Unlike HRVQTokenizer (vq.py) which applies 3-layer residual VQ to a single
global embedding, SpatialHRVQTokenizer applies INDEPENDENT single-layer VQ to
each spatial level produced by SpatialAtariEncoder.

Each spatial patch is quantized independently (batch dimension is B*N_patches).

    Input (dict from SpatialAtariEncoder.forward):
        'l0': (B,  4, 384)  — coarse 2×2 patches
        'l1': (B, 16, 384)  — mid 4×4 patches
        'l2': (B, 16, 384)  — fine 4×4 patches

    Output:
        token_dict:
            'l0': (B,  4) — integer token indices  [0, num_codes)
            'l1': (B, 16)
            'l2': (B, 16)
        total_vq_loss: scalar (commitment losses summed)
        quant_dict:
            'l0': (B,  4, 384) — quantised embeddings (STE)
            'l1': (B, 16, 384)
            'l2': (B, 16, 384)

The 3 VQ layers have SEPARATE codebooks (not residual).
Commitment cost schedule mirrors HRVQTokenizer: [0.05, 0.25, 0.60].

DEAD CODE REVIVAL (DALL-E / SoundStream / VQGAN practice):
Every `revival_interval` training steps, codebook entries that received zero
assignments in that window are replaced with randomly sampled encoder outputs
from the current batch (plus small noise).  Their EMA cluster size is reset to
the mean of the active entries so they compete fairly on the next step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vq import VQVAE  # Reuse existing EMA-based VQ layer


NUM_L0_PATCHES = 4
NUM_L1_PATCHES = 16
NUM_L2_PATCHES = 16


# ---------------------------------------------------------------------------
# Gradient-based VQ (replaces EMA when use_gradient_vq=True)
# ---------------------------------------------------------------------------
class GradientVQ(nn.Module):
    """
    Gradient-based vector quantisation with straight-through estimator.

    Unlike EMA-based VQVAE, the codebook is a standard nn.Embedding whose
    weights receive gradients directly (codebook_loss pulls it toward encoder
    outputs).  This is more responsive when the encoder outputs are clustered,
    because the codebook can actively pull away from the attractor rather than
    passively following EMA.

    Interface is identical to VQVAE:
        forward(x) → (quantized_st, vq_loss, indices)
    where x has shape (..., dim) — any leading batch dims work.
    """

    def __init__(self, num_codes: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codebook_entries = num_codes   # alias kept for compatibility
        self.num_codes = num_codes
        self.codebook_dim = dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, dim)
        # Uniform init scaled by 1/num_codes — gives initial norm ≈ 1/sqrt(dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, x: torch.Tensor):
        """
        x : (..., dim) — arbitrary batch shape, last dim is the embedding.
        Returns
        -------
        quantized_st : (..., dim) — STE quantised (same shape as x)
        vq_loss      : scalar
        indices      : (...,)     — integer indices, same leading shape as x
        """
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])           # (N, D)

        # Pairwise squared distance to all codebook entries
        dists = torch.cdist(flat, self.codebook.weight)  # (N, num_codes)
        indices_flat = dists.argmin(dim=-1)              # (N,)
        quantized_flat = self.codebook(indices_flat)     # (N, D)

        # Losses: codebook (pull codebook → encoder) + commitment (pull encoder → codebook)
        codebook_loss   = F.mse_loss(quantized_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: gradients pass through to encoder unchanged
        quantized_st = flat + (quantized_flat - flat).detach()

        return (
            quantized_st.reshape(orig_shape),
            vq_loss,
            indices_flat.reshape(orig_shape[:-1]),
        )


class SpatialHRVQTokenizer(nn.Module):
    """
    Per-level spatial VQ tokenizer with dead-code revival.

    Parameters
    ----------
    d_model : int
        Patch embedding dimension (must match SpatialAtariEncoder.d_model).
    num_codes_l0 / num_codes_l1 / num_codes_l2 : int
        Per-level codebook sizes.  Smaller codebooks are appropriate because
        each level sees far fewer distinct patches than 256:
          L0 has only 4 patches/frame  → 16 codes is already 16^4 = 65k states
          L1/L2 have 16 patches/frame  → 64 codes gives 64^16 ≈ 10^29 states
    commitment_costs : list[float]
        Per-level commitment costs [l0, l1, l2].
    revival_interval : int
        Check for dead codes every N training steps (default 100).
    revival_noise : float
        Std of Gaussian noise added to revived codes (default 1e-3).
        Applied before re-normalising to the encoder's L2-norm scale.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_codes_l0: int = 16,
        num_codes_l1: int = 64,
        num_codes_l2: int = 64,
        commitment_costs: list = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        revival_interval: int = 100,
        revival_noise: float = 1e-3,
        use_gradient_vq: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes_l0 = num_codes_l0
        self.num_codes_l1 = num_codes_l1
        self.num_codes_l2 = num_codes_l2
        self.num_codes = max(num_codes_l0, num_codes_l1, num_codes_l2)
        self.revival_interval = revival_interval
        self.revival_noise = revival_noise
        self.use_gradient_vq = use_gradient_vq

        if commitment_costs is None:
            commitment_costs = [0.05, 0.25, 0.60]

        # Separate VQ layer per spatial level.
        # use_gradient_vq=True  → GradientVQ  (codebook trained by gradients)
        # use_gradient_vq=False → VQVAE       (EMA codebook update, original)
        if use_gradient_vq:
            self.vq_l0 = GradientVQ(num_codes_l0, d_model, commitment_costs[0])
            self.vq_l1 = GradientVQ(num_codes_l1, d_model, commitment_costs[1])
            self.vq_l2 = GradientVQ(num_codes_l2, d_model, commitment_costs[2])
        else:
            self.vq_l0 = VQVAE(
                num_codebook_entries=num_codes_l0,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[0],
                decay=decay,
                epsilon=epsilon,
                layer_idx=0,
            )
            self.vq_l1 = VQVAE(
                num_codebook_entries=num_codes_l1,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[1],
                decay=decay,
                epsilon=epsilon,
                layer_idx=1,
            )
            self.vq_l2 = VQVAE(
                num_codebook_entries=num_codes_l2,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[2],
                decay=decay,
                epsilon=epsilon,
                layer_idx=2,
            )

        # --- Dead-code tracking ---
        # Separate buffer per level because sizes differ.
        self.register_buffer('_usage_l0', torch.zeros(num_codes_l0, dtype=torch.long))
        self.register_buffer('_usage_l1', torch.zeros(num_codes_l1, dtype=torch.long))
        self.register_buffer('_usage_l2', torch.zeros(num_codes_l2, dtype=torch.long))

        # Step counter (plain int — resets harmlessly on checkpoint resume)
        self._train_step = 0

    # ------------------------------------------------------------------
    # Dead-code revival
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _revive_dead_codes(
        self,
        level_name: str,
        vq_layer: VQVAE,
        usage_buf: torch.Tensor,    # (num_codes,) long — assignments this window
        encoder_flat: torch.Tensor, # (M, d_model) — current-batch encoder outputs
    ) -> int:
        """
        Replace zero-usage codebook entries with randomly sampled encoder outputs.

        Returns the number of codes revived (0 if none needed).
        """
        dead_mask = (usage_buf == 0)   # (num_codes,) bool
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0

        dead_idx = dead_mask.nonzero(as_tuple=True)[0]   # (n_dead,)
        M = encoder_flat.size(0)

        # Sample n_dead encoder outputs with replacement
        sample_idx = torch.randint(0, M, (n_dead,), device=encoder_flat.device)
        new_codes = encoder_flat[sample_idx].clone()     # (n_dead, d_model)

        # Add small noise then re-normalise so codes land on the unit sphere
        # (encoder outputs are L2-normalised in encoder_v2.py)
        new_codes = new_codes + self.revival_noise * torch.randn_like(new_codes)
        new_codes = F.normalize(new_codes, p=2, dim=-1)

        # --- Update codebook weights (shared by both EMA and gradient VQ) ---
        vq_layer.codebook.weight.data[dead_idx] = new_codes

        # EMA-specific state — only present on VQVAE, not on GradientVQ.
        # GradientVQ has no EMA buffers; the codebook is updated by gradients.
        active_mask = ~dead_mask
        if hasattr(vq_layer, 'ema_weight'):
            # Set ema_weight so the first EMA update moves in the right direction.
            vq_layer.ema_weight[dead_idx] = new_codes
            # Reset cluster size to the mean of active entries so revived codes
            # compete on equal footing rather than starting at ~0.
            if active_mask.any():
                mean_active = vq_layer.ema_cluster_size[active_mask].mean()
            else:
                mean_active = torch.tensor(1.0, device=encoder_flat.device)
            vq_layer.ema_cluster_size[dead_idx] = mean_active

        n_active = int(active_mask.sum()) if active_mask.any() else 0
        print(f"  [Revival] Revived {n_dead:3d} dead codes at level {level_name} "
              f"(active before: {n_active}/{vq_layer.num_codebook_entries})")
        return n_dead

    def _accumulate_usage(
        self,
        idx0: torch.Tensor,   # (B, 4)
        idx1: torch.Tensor,   # (B, 16)
        idx2: torch.Tensor,   # (B, 16)
    ):
        """Add current-batch assignment counts to the rolling usage buffers."""
        for idx, buf in [(idx0, self._usage_l0),
                         (idx1, self._usage_l1),
                         (idx2, self._usage_l2)]:
            flat = idx.reshape(-1).long()
            ones = torch.ones(flat.size(0), dtype=torch.long, device=flat.device)
            buf.scatter_add_(0, flat, ones)

    def maybe_revive(
        self,
        l0_feats: torch.Tensor,  # (B, 4,  d_model) — raw encoder outputs this step
        l1_feats: torch.Tensor,  # (B, 16, d_model)
        l2_feats: torch.Tensor,  # (B, 16, d_model)
    ) -> dict:
        """
        Check for dead codes and revive them if the revival interval has elapsed.

        Called from forward() when self.training is True.

        Returns a dict with revival counts per level (all zeros if not a revival step).
        """
        self._train_step += 1
        counts = {'l0': 0, 'l1': 0, 'l2': 0}

        if self._train_step % self.revival_interval != 0:
            return counts

        # Flat encoder outputs for sampling candidates
        flat_l0 = l0_feats.detach().reshape(-1, self.d_model)
        flat_l1 = l1_feats.detach().reshape(-1, self.d_model)
        flat_l2 = l2_feats.detach().reshape(-1, self.d_model)

        counts['l0'] = self._revive_dead_codes('l0', self.vq_l0, self._usage_l0, flat_l0)
        counts['l1'] = self._revive_dead_codes('l1', self.vq_l1, self._usage_l1, flat_l1)
        counts['l2'] = self._revive_dead_codes('l2', self.vq_l2, self._usage_l2, flat_l2)

        # Reset usage counters for the next window
        self._usage_l0.zero_()
        self._usage_l1.zero_()
        self._usage_l2.zero_()

        return counts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, spatial_feats: dict):
        """
        Parameters
        ----------
        spatial_feats : dict
            'l0': (B,  4, 384)
            'l1': (B, 16, 384)
            'l2': (B, 16, 384)

        Returns
        -------
        token_dict : dict — 'l0': (B,4), 'l1': (B,16), 'l2': (B,16)
        total_vq_loss : scalar
        quant_dict : dict — 'l0': (B,4,384), 'l1': (B,16,384), 'l2': (B,16,384)
        """
        l0_feats = spatial_feats['l0']  # (B, 4, 384)
        l1_feats = spatial_feats['l1']  # (B, 16, 384)
        l2_feats = spatial_feats['l2']  # (B, 16, 384)

        # Quantize each level independently.
        # VQVAE expects (..., codebook_dim) and flattens internally.
        q0, loss0, idx0 = self.vq_l0(l0_feats)  # q0:(B,4,384) idx0:(B,4)
        q1, loss1, idx1 = self.vq_l1(l1_feats)  # q1:(B,16,384) idx1:(B,16)
        q2, loss2, idx2 = self.vq_l2(l2_feats)  # q2:(B,16,384) idx2:(B,16)

        total_vq_loss = loss0 + loss1 + loss2

        # --- Dead-code tracking + revival (training mode only) ---
        if self.training:
            self._accumulate_usage(idx0, idx1, idx2)
            self.maybe_revive(l0_feats, l1_feats, l2_feats)

        token_dict = {'l0': idx0, 'l1': idx1, 'l2': idx2}
        quant_dict  = {'l0': q0,   'l1': q1,   'l2': q2}

        return token_dict, total_vq_loss, quant_dict

    @torch.no_grad()
    def encode(self, spatial_feats: dict) -> dict:
        """Encode spatial features to token indices (no gradient)."""
        token_dict, _, _ = self.forward(spatial_feats)
        return token_dict

    @torch.no_grad()
    def decode(self, token_dict: dict) -> dict:
        """
        Decode token indices back to patch embeddings.

        Parameters
        ----------
        token_dict : dict
            'l0': (B, 4) | (B, T, 4) — integer indices
            'l1': (B, 16) | (B, T, 16)
            'l2': (B, 16) | (B, T, 16)

        Returns
        -------
        dict of quantised embeddings matching input shape + (d_model,)
        """
        out = {}
        for key, vq in [('l0', self.vq_l0), ('l1', self.vq_l1), ('l2', self.vq_l2)]:
            out[key] = vq.codebook(token_dict[key])
        return out

    def get_codebook_usage(self, token_dict: dict) -> dict:
        """Return per-level codebook utilisation stats."""
        totals = {'l0': self.num_codes_l0, 'l1': self.num_codes_l1, 'l2': self.num_codes_l2}
        stats = {}
        for key in ['l0', 'l1', 'l2']:
            tokens = token_dict[key]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().numpy()
            unique = len(np.unique(tokens.reshape(-1)))
            total = totals[key]
            stats[key] = {
                'unique_codes': unique,
                'total_codes': total,
                'usage_pct': 100.0 * unique / total,
            }
        return stats


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: SpatialHRVQTokenizer (vq_spatial)")
    print("=" * 60)

    from encoder_v2 import SpatialAtariEncoder

    B = 4
    encoder = SpatialAtariEncoder(input_channels=4, d_model=384)
    tokenizer = SpatialHRVQTokenizer(d_model=384, num_codes_l0=16, num_codes_l1=64, num_codes_l2=64)

    params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"Tokenizer parameters: {params:,}")

    x = torch.randint(0, 256, (B, 4, 84, 84)).float()
    spatial_feats = encoder(x)

    # Encode in eval mode first (codebook frozen) so we can assert determinism
    tokenizer.eval()
    token_dict_eval = tokenizer.encode(spatial_feats)

    # Training forward (EMA updates codebook — results will differ from eval above)
    tokenizer.train()
    token_dict, vq_loss, quant_dict = tokenizer(spatial_feats)

    print(f"\nForward pass:")
    print(f"  vq_loss: {vq_loss.item():.4f}")
    for key in ['l0', 'l1', 'l2']:
        t = token_dict[key]
        q = quant_dict[key]
        print(f"  {key} tokens: {list(t.shape)}  quant: {list(q.shape)}")

    assert token_dict['l0'].shape == (B, 4),   f"l0 token shape wrong"
    assert token_dict['l1'].shape == (B, 16),  f"l1 token shape wrong"
    assert token_dict['l2'].shape == (B, 16),  f"l2 token shape wrong"
    assert quant_dict['l0'].shape == (B, 4,  384)
    assert quant_dict['l1'].shape == (B, 16, 384)
    assert quant_dict['l2'].shape == (B, 16, 384)

    # Verify eval encode is deterministic (two consecutive eval passes agree)
    tokenizer.eval()
    token_dict2 = tokenizer.encode(spatial_feats)
    tokenizer.train()
    assert (token_dict2['l0'] == token_dict_eval['l0']).all(), "Eval encode not deterministic"

    # Decode
    recon = tokenizer.decode(token_dict)
    for key in ['l0', 'l1', 'l2']:
        assert recon[key].shape == quant_dict[key].shape

    # Codebook usage
    usage = tokenizer.get_codebook_usage(token_dict)
    print(f"\nCodebook usage (before revival):")
    for key, stat in usage.items():
        print(f"  {key}: {stat['unique_codes']}/{stat['total_codes']} codes ({stat['usage_pct']:.1f}%)")

    # --- Revival smoke test ---
    # Force a revival check by simulating `revival_interval` steps.
    print(f"\nRevival smoke test (interval={tokenizer.revival_interval} steps):")
    tokenizer.train()
    tokenizer._train_step = tokenizer.revival_interval - 1   # one step before trigger

    # Accumulate fake zero usage (nothing assigned) to maximise dead codes
    tokenizer._usage_l0.zero_()
    tokenizer._usage_l1.zero_()
    tokenizer._usage_l2.zero_()

    # One more forward — this hits step % interval == 0 → revival fires
    token_dict2, _, _ = tokenizer(spatial_feats)

    usage2 = tokenizer.get_codebook_usage(token_dict2)
    print(f"\nCodebook usage (after revival):")
    for key, stat in usage2.items():
        print(f"  {key}: {stat['unique_codes']}/{stat['total_codes']} codes ({stat['usage_pct']:.1f}%)")

    # After revival, all codes should exist in the codebook.
    # EMA-based VQ: verify ema_cluster_size > 0.
    # GradientVQ: no ema_cluster_size — skip that check.
    for vq_layer, name in [(tokenizer.vq_l0, 'l0'),
                            (tokenizer.vq_l1, 'l1'),
                            (tokenizer.vq_l2, 'l2')]:
        if hasattr(vq_layer, 'ema_cluster_size'):
            assert (vq_layer.ema_cluster_size > 0).all(), \
                f"Some {name} cluster sizes are still 0 after revival"
            print(f"  {name}: all cluster sizes > 0  OK")
        else:
            print(f"  {name}: GradientVQ — no EMA cluster sizes, skipping that check  OK")

    print("\nSpatialHRVQTokenizer (EMA mode): PASSED")

    # ------------------------------------------------------------------
    # GradientVQ mode smoke test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SMOKE TEST: SpatialHRVQTokenizer with GradientVQ")
    print("=" * 60)

    tokenizer_gvq = SpatialHRVQTokenizer(
        d_model=384,
        num_codes_l0=16,
        num_codes_l1=64,
        num_codes_l2=64,
        use_gradient_vq=True,
    )

    # Verify each VQ layer is a GradientVQ instance
    assert isinstance(tokenizer_gvq.vq_l0, GradientVQ), "vq_l0 should be GradientVQ"
    assert isinstance(tokenizer_gvq.vq_l1, GradientVQ), "vq_l1 should be GradientVQ"
    assert isinstance(tokenizer_gvq.vq_l2, GradientVQ), "vq_l2 should be GradientVQ"
    print("  VQ layer types: GradientVQ  OK")

    tokenizer_gvq.train()
    token_dict_gvq, vq_loss_gvq, quant_dict_gvq = tokenizer_gvq(spatial_feats)

    # Shape checks
    assert token_dict_gvq['l0'].shape == (B, 4),   "GradientVQ l0 token shape wrong"
    assert token_dict_gvq['l1'].shape == (B, 16),  "GradientVQ l1 token shape wrong"
    assert token_dict_gvq['l2'].shape == (B, 16),  "GradientVQ l2 token shape wrong"
    assert quant_dict_gvq['l0'].shape == (B, 4,  384)
    assert quant_dict_gvq['l1'].shape == (B, 16, 384)
    assert quant_dict_gvq['l2'].shape == (B, 16, 384)
    print("  Token and quant shapes: OK")

    # vq_loss should be positive (codebook_loss + commitment_loss)
    assert vq_loss_gvq.item() > 0.0, f"GradientVQ vq_loss should be > 0, got {vq_loss_gvq.item()}"
    print(f"  vq_loss = {vq_loss_gvq.item():.4f}  (> 0)  OK")

    # vq_loss should be differentiable (grad_fn present)
    assert vq_loss_gvq.grad_fn is not None, "GradientVQ vq_loss should have grad_fn"
    print("  vq_loss has grad_fn (differentiable)  OK")

    print("\nSpatialHRVQTokenizer (GradientVQ mode): PASSED")
    print("=" * 60)
