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
        num_codebook_entries : int = 256,
        codebook_dim : int = 128,
        commitment_cost : float = 0.25,
    ):
        super().__init__()
        self.num_codebook_entries = num_codebook_entries
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        
        # CREATE CODEBOOK AND INIT WEIGHTS 
        self.codebook = nn.Embedding(num_codebook_entries, codebook_dim) 
        self.codebook.weight.data.uniform_(-1 / num_codebook_entries, 1 / num_codebook_entries) 

    def forward(
        self,
        z: torch.Tensor,
    ):
        """ QUANTIZE CONTINIOUS LATENTS TO NEAREST CODEBOOK ENTRIES (DISCRETE TOKENIZATION) """

        # FLATTEN INPUT FOR DISTANCE CALCULATION
        z_flat = z.reshape(-1, self.codebook_dim)

        # COMPUTE DISTANCES BETWEEN LATENTS AND CODEBOOK ENTRIES
        
        # a^2 + b^2 - 2ab (EXPANDED FORMULA FOR ||a-b||^2)
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) # |a|^2 SQUARED NORM OF LATENTS
            + torch.sum(self.codebook.weight ** 2, dim=1)  # |b|^2 SQUARED NORM OF CODEBOOK ENTRIES
            - 2 * torch.matmul(z_flat, self.codebook.weight.t()) # -2ab
        )   

        # FIND NEAREST CODEBOOK ENTRIES
        indices = torch.argmin(distances, dim=1)
        z_quantized = self.codebook(indices)

        # COMPUTE LOSSES
        # L1 : CODEBOOK LOSS
        codebook_loss = F.mse_loss(z_quantized, z_flat.detach())

        # L2 : COMMITMENT LOSS
        commitment_loss = F.mse_loss(z_quantized.detach(), z_flat)

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # STRAIGHT-THROUGH ESTIMATOR
        z_quantized = z_flat + (z_quantized - z_flat).detach()
        z_quantized = z_quantized.reshape(z.shape)

        return z_quantized, loss, indices.reshape(z.shape[:-1])  # RETURN QUANTIZED LATENTS, LOSS, AND INDICES
    
    
class VQTokenizer(nn.Module):
    """ COMPLETE VQ TOKENIZER MODULE: PROJECTION MLP + QUANTIZER """
    
    def __init__(
        self,
        input_dim : int = 384,
        latent_dim : int = 128,
        num_codes : int = 256,
        commitment_cost : float = 0.25,
    ):
        super().__init__()
        self.projection = ProjectionMLP(input_dim=input_dim, output_dim=latent_dim)
        self.vq = VQVAE(num_codebook_entries=num_codes, codebook_dim=latent_dim, commitment_cost=commitment_cost)
        self.num_codes = num_codes
        
    def forward(
        self,
        embeddings : torch.Tensor,
    ):
        """ FULL VQ PIPELINE: Project -> Quantize -> Return Tokens """
        
        # 1. 384D TO 128D PROJECTION
        z = self.projection(embeddings) 
        
        # 2. VQ-VAE QUANTIZATION
        z_quantized, vq_loss, indices = self.vq(z) 
        
        return z_quantized, vq_loss, indices
    
    """ ENCODE // DECODE HELPER FUNCTIONS """
    def encode(
        self,
        embeddings : torch.Tensor,
    ):
        """ ENCODE INPUT EMBEDDINGS TO DISCRETE TOKEN INDICES (NO GRADIENTS) """
        with torch.no_grad():
            _, _, tokens = self.forward(embeddings)
            
        return tokens
    
    
    def decode(
        self,
        tokens: torch.Tensor,
    ):
        """ DECODE DISCRETE TOKEN INDICES BACK TO CONTINUOUS LATENT EMBEDDINGS (QUANTIZED VECTORS)"""
        with torch.no_grad():
            z_quantized = self.vq.codebook(tokens)
        return z_quantized
    
          