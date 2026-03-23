"""
SPATIAL ATARI ENCODER - MULTI-SCALE PATCH EMBEDDINGS FROM CNN BACKBONE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_L0_PATCHES = 4   # 2x2
NUM_L1_PATCHES = 16  # 4x4
NUM_L2_PATCHES = 16  # 4x4


class SpatialAtariEncoder(nn.Module):
    """ MULTI-SCALE SPATIAL CNN ENCODER FOR ATARI FRAMES """

    def __init__(
        self,
        input_channels: int = 4,
        d_model: int = 384,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model

        # SHARED CNN BACKBONE
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

        # PER-LEVEL HEADS
        # L0 HEAD - COARSE PHYSICS FROM SHALLOW FEATURES
        self.l0_pool = nn.AdaptiveAvgPool2d((2, 2))   # (B,32,2,2)
        self.l0_proj = nn.Conv2d(32, d_model, kernel_size=1)  # (B,384,2,2)
        self.l0_norm = nn.GroupNorm(1, d_model)

        # L1 HEAD - OBJECT-LEVEL FEATURES FROM MID-LEVEL
        self.l1_pool = nn.AdaptiveAvgPool2d((4, 4))   # (B,64,4,4)
        self.l1_proj = nn.Conv2d(64, d_model, kernel_size=1)  # (B,384,4,4)
        self.l1_norm = nn.GroupNorm(1, d_model)

        # L2 HEAD - FINE DETAIL FROM DEEP FEATURES
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
        # NORMALIZE TO [0, 1] IF UINT8 INPUT
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        f1 = self.conv1(x)   # (B, 32, 20, 20)
        f2 = self.conv2(f1)  # (B, 64,  9,  9)
        f3 = self.conv3(f2)  # (B, 64,  7,  7)

        # L0 - COARSE 2x2 PATCHES FROM SHALLOW FEATURES
        l0 = self.l0_pool(f1)          # (B, 32, 2, 2)
        l0 = self.l0_proj(l0)          # (B, 384, 2, 2)
        l0 = self.l0_norm(l0)
        l0 = l0.flatten(2).transpose(1, 2)  # (B, 4, 384)

        # L1 - MID 4x4 PATCHES FROM MID-LEVEL FEATURES
        l1 = self.l1_pool(f2)          # (B, 64, 4, 4)
        l1 = self.l1_proj(l1)          # (B, 384, 4, 4)
        l1 = self.l1_norm(l1)
        l1 = l1.flatten(2).transpose(1, 2)  # (B, 16, 384)

        # L2 - FINE 4x4 PATCHES FROM DEEP FEATURES
        l2 = self.l2_pool(f3)          # (B, 64, 4, 4)
        l2 = self.l2_proj(l2)          # (B, 384, 4, 4)
        l2 = self.l2_norm(l2)
        l2 = l2.flatten(2).transpose(1, 2)  # (B, 16, 384)

        # L2-NORMALIZE EACH PATCH EMBEDDING
        l0 = F.normalize(l0, p=2, dim=-1)
        l1 = F.normalize(l1, p=2, dim=-1)
        l2 = F.normalize(l2, p=2, dim=-1)

        return {'l0': l0, 'l1': l1, 'l2': l2}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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

    # NORMS SHOULD BE ~1.0 AFTER L2 NORMALIZATION
    for key in ['l0', 'l1', 'l2']:
        norms = out[key].norm(dim=-1)
        assert (norms - 1.0).abs().max() < 1e-4, f"Norm check failed for {key}"

    print("\nSpatialAtariEncoder: PASSED")
    print("=" * 60)
