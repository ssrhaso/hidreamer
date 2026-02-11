""" DATASET CLASS FOR WORLD MODEL TRAINING

1. RESHAPES
2. RESPECT EPISODE BOUNDARIES
3. MULTI-GAME MIXING
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import yaml

class WorldModelDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    

def create_dataloders():
    pass

if __name__ == "__main__":
    pass