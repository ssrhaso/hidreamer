"""
FRAME DECODER FOR HI-DREAMER PIXEL DECODING

Reconstructs Atari frames from HRVQ codebook embeddings.  Used as the grounding
mechanism for the visual policy mode:

    tokens → codebook embeddings → decoded frames (3 levels) → CNN → actor/critic

Why this works:
    IRIS (Micheli et al., ICLR 2023) proves that imagination-based learning with a
    discrete token WM transfers to reality when the policy operates on DECODED FRAMES
    rather than latent hidden states.  Both imagination (predicted tokens) and real play
    (observed tokens) pass through the SAME decode→CNN path, so there is no
    train/eval distribution shift.

Three reconstruction levels via a SINGLE shared decoder:
    coarse:  decode(emb_l0)              — global layout, rough positions
    mid:     decode(emb_l0 + emb_l1)     — mechanics, clearer shapes
    full:    decode(emb_l0+emb_l1+emb_l2)— rendering, finest detail

The sum is valid because HRVQ is a residual VQ: each layer corrects the residual
of the previous, so L0+L1+...+Lk is the partial reconstruction up to level k.

Training:  Supervised MSE on all 3 levels simultaneously (see train_decoder.py).
Usage:     Pre-train, freeze, then use as input to VisualFeatureExtractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDecoder(nn.Module):
    """Decode 384-D codebook embedding → (1, 84, 84) grayscale frame.

    Input:  (B, 384)        — summed HRVQ codebook embeddings (any level)
    Output: (B, 1, 84, 84)  — normalised [0, 1] grayscale reconstruction

    Architecture
    ------------
    Linear(384, base_ch*6*6) → reshape(B, base_ch, 6, 6)
    ConvTranspose × 4: 6→12→24→48→96
    Bilinear interpolation: 96→84
    Sigmoid: [0, 1]

    Total params ≈ 2M (base_ch=128).
    """

    def __init__(self, emb_dim: int = 384, base_ch: int = 128):
        super().__init__()
        self.emb_dim  = emb_dim
        self.base_ch  = base_ch
        self._spatial = 6   # spatial size after project

        # Project embedding → spatial feature map
        self.project = nn.Sequential(
            nn.Linear(emb_dim, base_ch * self._spatial * self._spatial),
            nn.SiLU(),
        )

        # Transposed-CNN: doubles spatial dims at each step
        # 6→12→24→48→96, then bilinear→84
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 64, kernel_size=4, stride=2, padding=1),  # 12×12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,      32, kernel_size=4, stride=2, padding=1),  # 24×24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,      16, kernel_size=4, stride=2, padding=1),  # 48×48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,       1, kernel_size=4, stride=2, padding=1),  # 96×96
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (B, 384) — summed HRVQ codebook embeddings
        Returns:
            frame: (B, 1, 84, 84) in [0, 1]
        """
        B  = emb.size(0)
        s  = self._spatial
        x  = self.project(emb)                  # (B, base_ch * s * s)
        x  = x.view(B, self.base_ch, s, s)      # (B, base_ch, 6, 6)
        x  = self.deconv(x)                     # (B, 1, 96, 96)
        x  = F.interpolate(x, size=(84, 84),
                           mode='bilinear', align_corners=False)  # (B, 1, 84, 84)
        x  = torch.sigmoid(x)                   # [0, 1]
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    decoder = FrameDecoder()
    print(f"FrameDecoder params: {decoder.count_parameters():,}")
    emb = torch.randn(8, 384)
    out = decoder(emb)
    print(f"Input: {list(emb.shape)}  Output: {list(out.shape)}")
    assert out.shape == (8, 1, 84, 84), f"Wrong output shape: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    print("FrameDecoder: OK")
