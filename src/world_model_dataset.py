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
    def __init__(
        self,
        games : List[str],
        tokens_dir : str = "checkpoints/rsvq_tokens",
        replay_dir : str = "data",
        seq_len : int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        
        self.all_tokens = []        # (N, 3) arrays, 1x per game
        self.all_actions = []       # (N,) arrays, 1x per game
        self.valid_starts = []      # (game_idx, start_idx) tuples 

        for game_idx, game in enumerate(games):
            print(f"LOADING GAME {game}...")
            
            """ 1. LOAD HRVQ TOKENS (3 LAYERS) """
            layers = []
            for layer in range(3):
                path = f"{tokens_dir}/vq_tokens_ALE_{game}_layer{layer}.npy"
                t = np.load(path).squeeze() # (100000, 1, 1) -> (100000,)
                layers.append(t)
            tokens_3layer = np.stack(layers, axis = 1).astype(np.int64) # (100000, 3)
            
            """ 2. LOAD ACTIONS (FROM REPLAY BUFFER) """
            buf = np.load(f"{replay_dir}/replay_buffer_ALE_{game}.npz") 
            actions = buf['actions'].astype(np.int64) # (100000,)
            dones = buf['dones']
            
            N = len(actions)
            assert len(tokens_3layer) == N, f"Token and action lengths do not match for game {game}!"
            
            """ 3. STORE GAME DATA """
            
            """ 4. COMPUTE VALID START INDICES (RESPECTING EPISODE BOUNDARIES) """
            
            pass
            
            
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        pass
    

def create_dataloders():
    pass

if __name__ == "__main__":
    pass