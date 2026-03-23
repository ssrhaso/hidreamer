"""
FRAME DECODER - RECONSTRUCT ATARI FRAMES FROM HRVQ CODEBOOK EMBEDDINGS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDecoder(nn.Module):
    """ DECODE 384-D CODEBOOK EMBEDDING TO (1, 84, 84) GRAYSCALE FRAME """

    def __init__(self, emb_dim: int = 384, base_ch: int = 128):
        super().__init__()
        self.emb_dim  = emb_dim
        self.base_ch  = base_ch
        self._spatial = 6   # SPATIAL SIZE AFTER PROJECT

        # PROJECT EMBEDDING TO SPATIAL FEATURE MAP
        self.project = nn.Sequential(
            nn.Linear(emb_dim, base_ch * self._spatial * self._spatial),
            nn.SiLU(),
        )

        # TRANSPOSED-CNN: DOUBLES SPATIAL DIMS AT EACH STEP
        # 6->12->24->48->96, THEN BILINEAR->84
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 64, kernel_size=4, stride=2, padding=1),  # 12x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,      32, kernel_size=4, stride=2, padding=1),  # 24x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,      16, kernel_size=4, stride=2, padding=1),  # 48x48
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,       1, kernel_size=4, stride=2, padding=1),  # 96x96
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """ DECODE EMBEDDING (B, 384) TO FRAME (B, 1, 84, 84) IN [0, 1] """
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


if __name__ == "__main__":
    decoder = FrameDecoder()
    print(f"FrameDecoder params: {decoder.count_parameters():,}")
    emb = torch.randn(8, 384)
    out = decoder(emb)
    print(f"Input: {list(emb.shape)}  Output: {list(out.shape)}")
    assert out.shape == (8, 1, 84, 84), f"Wrong output shape: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    print("FrameDecoder: OK")
