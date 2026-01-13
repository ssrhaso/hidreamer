""" TRAIN VQ TOKENIZER ON DINOv2 EMBEDDINGS """
""" NOTE: BASELINE CODE, TO BE MODIFIED """

import os 
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from vq import VQVAE, VQTokenizer, load_config

class EmbeddingDataset(Dataset):
    """ DATASET WRAPPER FOR DINOv2 EMBEDDINGS
    INPUT: .npy FILE CONTAINING PRECOMPUTED EMBEDDINGS
    """
    
    # CONSTRUCTOR
    def __init__(self, embeddings_path : str,):
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        print(f"LOADED EMBEDDINGS FROM {embeddings_path}, LENGTH: {len(self.embeddings)}")
    
    # SAMPLING METHODS
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx])
    
def compute_codebook_stats(
    tokens : torch.Tensor,
    num_codes : int = 256,
):
    """ COMPUTE CODEBOOK USAGE STATISTICS (TOKEN FREQUENCY) """
    
    # UNIQUE CODES
    unique_codes = torch.unique(tokens)
    num_used = len(unique_codes)
    
    # FREQUENCY COUNTS
    token_counts = torch.bincount(tokens, minlength = num_codes).float()
    token_probs = token_counts / token_counts.sum()
    
    # PERPLEXITY (EFFECTIVE NUMBER OF CODES USED)
    perplexity = torch.exp(-torch.sum(token_probs * torch.log(token_probs + 1e-10))).item()
    
    # HISTOGRAM 
    usage_histogram = token_counts.cpu().numpy().tolist()
    
    return {
        'num_used_codes': int(num_used),
        'total_codes': num_codes,
        'usage_ratio': float(num_used / num_codes),
        'perplexity': float(perplexity),
        'usage_histogram': usage_histogram,
    }