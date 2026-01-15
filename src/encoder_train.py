"""
Train Simple CNN Encoder with Contrastive Learning
Expected training time: 30-45 minutes on T4 GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import yaml

from encoder_v1 import AtariCNNEncoder

""" CONFIG LOADING """
def load_config(
    config_path : str = "configs/encoder.yaml",
):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



class TemporalContrastiveDataset(Dataset):
    """
    Dataset for temporal contrastive learning
    
    Positive pairs: consecutive frames (t, t+1)
    Negative pairs: random distant frames
    """
    
    def __init__(self, frames: np.ndarray, temporal_distance: int = 1):
        """
        Args:
            frames: (N, 4, 84, 84) stacked frames from replay buffer
            temporal_distance: frames apart for positive pairs
        """
        self.frames = frames
        self.temporal_distance = temporal_distance
        self.n = len(frames) - temporal_distance
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Anchor: frame at time t
        anchor = self.frames[idx]  # (4, 84, 84)
        
        # Positive: frame at time t+1
        positive = self.frames[idx + self.temporal_distance]
        
        # Negative: random frame far from anchor
        neg_idx = np.random.randint(0, len(self.frames))
        while abs(neg_idx - idx) < 10:  # Ensure temporal distance
            neg_idx = np.random.randint(0, len(self.frames))
        negative = self.frames[neg_idx]
        
        # Normalize to [0, 1]
        anchor = torch.from_numpy(anchor.astype(np.float32) / 255.0)
        positive = torch.from_numpy(positive.astype(np.float32) / 255.0)
        negative = torch.from_numpy(negative.astype(np.float32) / 255.0)
        
        return anchor, positive, negative


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Contrastive loss (InfoNCE)
    
    Maximizes similarity between anchor and positive
    Minimizes similarity between anchor and negative
    
    Args:
        anchor, positive, negative: (B, D) normalized embeddings
        temperature: temperature scaling parameter
    
    Returns:
        loss: scalar loss value
    """
    
    # Cosine similarities (embeddings already L2-normalized)
    pos_sim = (anchor * positive).sum(dim=-1) / temperature  # (B,)
    neg_sim = (anchor * negative).sum(dim=-1) / temperature  # (B,)
    
    # InfoNCE loss
    loss = -torch.log(
        torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))
    )
    
    return loss.mean()


def train_encoder(
    data_paths: list = ['data/replay_buffer_ALE_PongNoFrameskip-v4.npz', 'data/replay_buffer_ALE_BreakoutNoFrameskip-v4.npz', 'data/replay_buffer_ALE_SpaceInvadersNoFrameskip-v4.npz'],
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    save_dir: str = 'checkpoints',
    save_every: int = 10,
):
    """
    Train CNN encoder with contrastive learning
    
    Args:
        data_paths: List of paths to replay buffers
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Adam learning rate
        weight_decay: L2 regularization
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    print("-" * 70)
    print("TRAINING ATARI CNN ENCODER")
    print(f"\nDevice: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    # Load and concatenate data from multiple games
    print(f"\nLoading data from {len(data_paths)} game(s)...")
    all_frames = []

    for i, path in enumerate(data_paths, 1):
        print(f"  [{i}/{len(data_paths)}] Loading {path}...")
        with np.load(path) as data:
            game_frames = data['states']
        print(f"      → {game_frames.shape[0]:,} frames")
        all_frames.append(game_frames)
    # Concatenate along batch dimension
    frames = np.concatenate(all_frames, axis=0)

    print(f"\nCombined Dataset:")
    print(f"  Total frames: {frames.shape[0]:,}")
    print(f"  Shape: {frames.shape}")
    print(f"  Memory: {frames.nbytes / 1024**3:.2f} GB")
    print(f"  Dtype: {frames.dtype}")
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = TemporalContrastiveDataset(frames, temporal_distance=1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset):,}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    print(f"\nInitializing model...")
    model = AtariCNNEncoder(input_channels=4, embedding_dim=384)
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.count_parameters() * 4 / 1024**2:.2f} MB (float32)")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"STARTING TRAINING ({num_epochs} epochs)")
    print(f"{'-'*70}\n")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=100
        )
        
        for anchor, positive, negative in pbar:
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            # Compute loss
            loss = contrastive_loss(emb_anchor, emb_positive, emb_negative)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Time elapsed: {elapsed/60:.1f} min")
        print(f"  Est. remaining: {(elapsed/(epoch+1))*(num_epochs-epoch-1)/60:.1f} min")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(save_dir) / 'encoder_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_path)
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = Path(save_dir) / f'encoder_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint: {ckpt_path}")
        
        print()
    
    # Final save
    final_path = Path(save_dir) / 'encoder_final.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)
    
    total_time = time.time() - start_time
    

    print("TRAINING COMPLETE")
    print("-" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nSaved models:")
    print(f"  Best:  {Path(save_dir) / 'encoder_best.pt'}")
    print(f"  Final: {final_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model


if __name__ == "__main__":
    # Load config
    config = load_config('configs/encoder.yaml')
    
    # Train encoder
    model = train_encoder(
        data_paths=config['data']['replay_buffers'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_dir=config['logging']['save_dir'],
        save_every=config['logging']['save_every'],
    )
    
    print("\nTRAINING FINISHED, MODEL READY FOR USE.")