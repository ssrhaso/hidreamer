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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vq import VQVAE  # Reuse existing EMA-based VQ layer


NUM_L0_PATCHES = 4
NUM_L1_PATCHES = 16
NUM_L2_PATCHES = 16


class SpatialHRVQTokenizer(nn.Module):
    """
    Per-level spatial VQ tokenizer.

    Parameters
    ----------
    d_model : int
        Patch embedding dimension (must match SpatialAtariEncoder.d_model).
    num_codes : int
        Codebook size per level (default 256).
    commitment_costs : list[float]
        Per-level commitment costs [l0, l1, l2].
    """

    def __init__(
        self,
        d_model: int = 384,
        num_codes: int = 256,
        commitment_costs: list = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        if commitment_costs is None:
            commitment_costs = [0.05, 0.25, 0.60]

        # Separate VQ layer per spatial level
        self.vq_l0 = VQVAE(
            num_codebook_entries=num_codes,
            codebook_dim=d_model,
            commitment_cost=commitment_costs[0],
            decay=decay,
            epsilon=epsilon,
            layer_idx=0,
        )
        self.vq_l1 = VQVAE(
            num_codebook_entries=num_codes,
            codebook_dim=d_model,
            commitment_cost=commitment_costs[1],
            decay=decay,
            epsilon=epsilon,
            layer_idx=1,
        )
        self.vq_l2 = VQVAE(
            num_codebook_entries=num_codes,
            codebook_dim=d_model,
            commitment_cost=commitment_costs[2],
            decay=decay,
            epsilon=epsilon,
            layer_idx=2,
        )

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

        B = l0_feats.size(0)

        # Quantize each level independently.
        # VQVAE expects (..., codebook_dim) and flattens internally.
        q0, loss0, idx0 = self.vq_l0(l0_feats)  # q0:(B,4,384) idx0:(B,4)
        q1, loss1, idx1 = self.vq_l1(l1_feats)  # q1:(B,16,384) idx1:(B,16)
        q2, loss2, idx2 = self.vq_l2(l2_feats)  # q2:(B,16,384) idx2:(B,16)

        total_vq_loss = loss0 + loss1 + loss2

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
        stats = {}
        for key in ['l0', 'l1', 'l2']:
            tokens = token_dict[key]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().numpy()
            unique = len(np.unique(tokens.reshape(-1)))
            stats[key] = {
                'unique_codes': unique,
                'total_codes': self.num_codes,
                'usage_pct': 100.0 * unique / self.num_codes,
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
    tokenizer = SpatialHRVQTokenizer(d_model=384, num_codes=256)

    params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"Tokenizer parameters: {params:,}")

    x = torch.randint(0, 256, (B, 4, 84, 84)).float()
    spatial_feats = encoder(x)

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

    # Encode (no grad) — switch to eval to freeze codebook for comparison
    tokenizer.eval()
    token_dict2 = tokenizer.encode(spatial_feats)
    tokenizer.train()
    assert (token_dict2['l0'] == token_dict['l0']).all()

    # Decode
    recon = tokenizer.decode(token_dict)
    for key in ['l0', 'l1', 'l2']:
        assert recon[key].shape == quant_dict[key].shape

    # Codebook usage
    usage = tokenizer.get_codebook_usage(token_dict)
    print(f"\nCodebook usage:")
    for key, stat in usage.items():
        print(f"  {key}: {stat['unique_codes']}/{stat['total_codes']} codes ({stat['usage_pct']:.1f}%)")

    print("\nSpatialHRVQTokenizer: PASSED")
    print("=" * 60)
