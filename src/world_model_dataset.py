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
            self.all_tokens.append(tokens_3layer) 
            self.all_actions.append(actions)
            
            """ 4. COMPUTE VALID START INDICES (RESPECTING EPISODE BOUNDARIES) """
            cumulativesum = np.concatenate([ [0], np.cumsum(dones.astype(bool)) ]) # (100001,)
            num_candidates = N - self.seq_len + 1
            boundaries = cumulativesum[self.seq_len - 1 : N] - cumulativesum[:num_candidates]
            valid_idx = np.where(boundaries == 0)[0] # indices where no episode boundary is crossed
            
            for s in valid_idx:
                self.valid_starts.append((game_idx, int(s)))
            print(f" {N} TIMESTEPS, {len(valid_idx)} VALID STARTS")
            
            
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        game_idx, start = self.valid_starts[idx]
        end = start + self.seq_len
        
        tokens = torch.from_numpy(self.all_tokens[game_idx][start:end]) # (64, 3)
        actions = torch.from_numpy(self.all_actions[game_idx][start:end]) # (64,)
        
        return tokens, actions
    

def create_dataloders(
    config_path : str = "configs/worldmodel.yaml",
    seed : int = 42,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = WorldModelDataset(
        games = config['data']['games'],
        tokens_dir = config['data']['tokens_dir'],
        replay_dir = config['data']['replay_dir'],
        seq_len = config['training']['seq_len'],
    )
    
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator = torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
    )
    
    val_loader = DataLoader(
    )
    
    info = {
    }
    
    return train_loader, val_loader, info

if __name__ == "__main__":
    pass