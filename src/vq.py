""" VECTOR QUANTIZATION MODULES FOR DISCRETE TOKENIZATION OF EMBEDDINGS """
""" NOTE : BASELINE CODE, TO BE MODIFIED """

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    """ PROJECT 384-DIM DINOv2 EMBEDDINGS TO 128-DIM LATENT DIMENSION FOR VQ-VAE """

    def __init__(
            self,
            input_dim : int = 384,
            output_dim : int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )