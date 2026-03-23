"""
SPATIAL ATARI ENCODER (encoder_v2) — Multi-scale patch embeddings.

Root cause fix: encoder_v1 flattens (64,7,7) → 3136 → 384, destroying all spatial
structure. The HRVQ then produces 3 global tokens with zero spatial information
(linear probe confirmed: max paddle-y correlation = 0.276).

This encoder maintains spatial structure by pooling to DIFFERENT resolutions from
DIFFERENT CNN depths, producing per-patch embeddings:

    Input:  (B, 4, 84, 84) — 4 stacked grayscale frames

    Output (dict):
      'l0': (B,  4, 384) — 2×2 coarse patches from conv1 features (global physics)
      'l1': (B, 16, 384) — 4×4 mid patches from conv2 features (objects/mechanics)
      'l2': (B, 16, 384) — 4×4 fine patches from conv3 features (fine detail)

CNN backbone (same strides as encoder_v1):
    conv1: (B,4,84,84)  → (B,32,20,20)   kernel=8, stride=4
    conv2: (B,32,20,20) → (B,64,9,9)    kernel=4, stride=2
    conv3: (B,64,9,9)   → (B,64,7,7)    kernel=3, stride=1

Per-level heads (1×1 conv then pool):
    L0 head: pool(2,2) → (B,32,2,2)  → conv1x1 → (B,384,2,2) → (B,4,384)
    L1 head: pool(4,4) → (B,64,4,4)  → conv1x1 → (B,384,4,4) → (B,16,384)
    L2 head: pool(4,4) → (B,64,4,4)  → conv1x1 → (B,384,4,4) → (B,16,384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_L0_PATCHES = 4   # 2×2
NUM_L1_PATCHES = 16  # 4×4
NUM_L2_PATCHES = 16  # 4×4


class SpatialAtariEncoder(nn.Module):
    """
    Multi-scale spatial CNN encoder for Atari frames.

    Parameters
    ----------
    input_channels : int
        Number of stacked frames (default 4).
    d_model : int
        Output embedding dimension per patch (default 384, matches HRVQ codebook dim).
    """

    def __init__(
        self,
        input_channels: int = 4,
        d_model: int = 384,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model

        # --- Shared CNN backbone ---
        # Identical strides to encoder_v1 so weights can be transferred if desired.
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
        )  # (B,32,20,20)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )  # (B,64,9,9)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )  # (B,64,7,7)

        # --- Per-level heads ---
        # Each head: adaptive pool → 1×1 conv (channel projection) → LayerNorm
        # Using 1×1 conv rather than a flat Linear keeps computation local to each patch.

        # L0 head: coarse physics from shallow features
        self.l0_pool = nn.AdaptiveAvgPool2d((2, 2))   # (B,32,2,2)
        self.l0_proj = nn.Conv2d(32, d_model, kernel_size=1)  # (B,384,2,2)
        self.l0_norm = nn.GroupNorm(1, d_model)  # LayerNorm over spatial patches

        # L1 head: object-level features from mid-level
        self.l1_pool = nn.AdaptiveAvgPool2d((4, 4))   # (B,64,4,4)
        self.l1_proj = nn.Conv2d(64, d_model, kernel_size=1)  # (B,384,4,4)
        self.l1_norm = nn.GroupNorm(1, d_model)

        # L2 head: fine detail from deep features
        self.l2_pool = nn.AdaptiveAvgPool2d((4, 4))   # (B,64,4,4)
        self.l2_proj = nn.Conv2d(64, d_model, kernel_size=1)  # (B,384,4,4)
        self.l2_norm = nn.GroupNorm(1, d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : (B, 4, 84, 84) — stacked grayscale frames in [0, 255] or [0, 1]

        Returns
        -------
        dict with keys:
            'l0': (B,  4, 384)
            'l1': (B, 16, 384)
            'l2': (B, 16, 384)
        """
        # Normalise to [0, 1] if uint8-style input
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        f1 = self.conv1(x)   # (B, 32, 20, 20)
        f2 = self.conv2(f1)  # (B, 64,  9,  9)
        f3 = self.conv3(f2)  # (B, 64,  7,  7)

        # L0 — coarse 2×2 patches from shallow features
        l0 = self.l0_pool(f1)          # (B, 32, 2, 2)
        l0 = self.l0_proj(l0)          # (B, 384, 2, 2)
        l0 = self.l0_norm(l0)
        l0 = l0.flatten(2).transpose(1, 2)  # (B, 4, 384)

        # L1 — mid 4×4 patches from mid-level features
        l1 = self.l1_pool(f2)          # (B, 64, 4, 4)
        l1 = self.l1_proj(l1)          # (B, 384, 4, 4)
        l1 = self.l1_norm(l1)
        l1 = l1.flatten(2).transpose(1, 2)  # (B, 16, 384)

        # L2 — fine 4×4 patches from deep features
        l2 = self.l2_pool(f3)          # (B, 64, 4, 4)
        l2 = self.l2_proj(l2)          # (B, 384, 4, 4)
        l2 = self.l2_norm(l2)
        l2 = l2.flatten(2).transpose(1, 2)  # (B, 16, 384)

        # L2-normalise each patch embedding (matches downstream VQVAE expectations)
        l0 = F.normalize(l0, p=2, dim=-1)
        l1 = F.normalize(l1, p=2, dim=-1)
        l2 = F.normalize(l2, p=2, dim=-1)

        return {'l0': l0, 'l1': l1, 'l2': l2}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: SpatialAtariEncoder (encoder_v2)")
    print("=" * 60)

    B = 4
    model = SpatialAtariEncoder(input_channels=4, d_model=384)
    print(f"Parameters: {model.count_parameters():,}")

    x = torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8).float()
    out = model(x)

    for key, val in out.items():
        print(f"  {key}: {list(val.shape)}  norm_mean={val.norm(dim=-1).mean():.4f}")

    assert out['l0'].shape == (B, 4,  384), f"L0 shape wrong: {out['l0'].shape}"
    assert out['l1'].shape == (B, 16, 384), f"L1 shape wrong: {out['l1'].shape}"
    assert out['l2'].shape == (B, 16, 384), f"L2 shape wrong: {out['l2'].shape}"

    # Norms should be ~1.0 after L2 normalisation
    for key in ['l0', 'l1', 'l2']:
        norms = out[key].norm(dim=-1)
        assert (norms - 1.0).abs().max() < 1e-4, f"Norm check failed for {key}"

    print("\nSpatialAtariEncoder: PASSED")
    print("=" * 60)
