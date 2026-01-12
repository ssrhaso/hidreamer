""" VECTOR QUANTIZATION MODULES FOR DISCRETE TOKENIZATION OF EMBEDDINGS """
""" NOTE : BASELINE CODE, TO BE MODIFIED """

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

def load_config(
    config_path : str = "configs/vq.yaml",
):
    """ LOAD VQ-VAE CONFIG FROM YAML FILE """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['model']


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

    def forward(
        self,
        x : torch.Tensor,
    ):
        """ FORWARD PASS THROUGH PROJECTION MLP (384->128) """
        return self.net(x)








class VQVAE(nn.Module):
    """ VQ-VAE MODULE FOR VECTOR QUANTIZATION AND DISCRETE TOKENIZATION OF LATENT EMBEDDINGS """
    def __init__(
        self,
        num_embeddings : int = 256,
        embedding_dim : int = 128,
        commitment_cost : float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(
        self,
        z: torch.Tensor,
    ):
        """ QUANTIZE CONTINIOUS LATENTS TO NEAREST CODEBOOK ENTRIES (DISCRETE TOKENIZATION) """

        # FLATTEN INPUT FOR DISTANCE CALCULATION
        z_flat = z.reshape(-1, self.embedding_dim)

        # COMPUTE DISTANCES BETWEEN LATENTS AND CODEBOOK ENTRIES
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )   

        # FIND NEAREST CODEBOOK ENTRIES
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)

        # COMPUTE LOSSES
        # L1 : CODEBOOK LOSS
        codebook_loss = F.mse_loss(z_q, z_flat.detach())

        # L2 : COMMITMENT LOSS
        commitment_loss = F.mse_loss(z_q.detach(), z_flat)

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # STRAIGHT-THROUGH ESTIMATOR
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.reshape(z.shape)

        return z_q, loss, indices.reshape(z.shape[:-1])  # RETURN QUANTIZED LATENTS, LOSS, AND INDICES