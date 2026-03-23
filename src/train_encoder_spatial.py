"""
TRAIN SPATIAL ENCODER — Joint training of SpatialAtariEncoder + SpatialHRVQTokenizer.

Loss: reconstruction of patch embeddings after encode → quantize.
  total_loss = l0_weight * MSE(q_l0, e_l0) + l1_weight * MSE(q_l1, e_l1)
             + l2_weight * MSE(q_l2, e_l2) + vq_commitment_loss

Unlike the pixel decoder approach (train_decoder.py), this trains on embedding-space
reconstruction — the encoder is evaluated on how well the VQ can represent the patches.

Usage:
    python src/train_encoder_spatial.py --config configs/encoder_spatial.yaml
    python src/train_encoder_spatial.py --config configs/encoder_spatial.yaml --wandb
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder_v2 import SpatialAtariEncoder
from vq_spatial import SpatialHRVQTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class AtariFrameDataset(torch.utils.data.Dataset):
    """
    Loads raw Atari frames (stacked 4×84×84) from replay .npz files.

    Expects the same format as the existing replay buffer:
        data/{game}/frames.npy   — (N, 4, 84, 84) uint8
    or
        data/{game}/obs.npy      — same format
    """

    def __init__(self, replay_dir: str, games: list, max_frames_per_game: int = 100_000):
        self.frames = []

        for game in games:
            game_dir = Path(replay_dir) / game

            # Try common file names
            for fname in ['frames.npy', 'obs.npy', 'observations.npy']:
                fpath = game_dir / fname
                if fpath.exists():
                    data = np.load(str(fpath), mmap_mode='r')
                    n = min(len(data), max_frames_per_game)
                    self.frames.append(data[:n])
                    print(f"  Loaded {n:,} frames from {fpath}")
                    break
            else:
                # Try to reconstruct from replay buffer format (sequences of obs)
                seq_path = game_dir / 'sequences.npz'
                if seq_path.exists():
                    d = np.load(str(seq_path))
                    obs = d['obs']  # (N, seq_len, 4, 84, 84)
                    obs = obs.reshape(-1, 4, 84, 84)[:max_frames_per_game]
                    self.frames.append(obs)
                    print(f"  Loaded {len(obs):,} frames from {seq_path}")
                else:
                    print(f"  WARNING: No frame data found for {game} in {game_dir}")

        if not self.frames:
            raise FileNotFoundError(
                f"No frame data found in {replay_dir} for games {games}. "
                "Run data collection first."
            )

        self.all_frames = np.concatenate(self.frames, axis=0)
        print(f"\n  Total frames: {len(self.all_frames):,}")

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        frame = self.all_frames[idx].astype(np.float32) / 255.0
        return torch.from_numpy(frame)


def build_dataloaders(config: dict, batch_size: int):
    """Build train and validation DataLoaders."""
    dataset = AtariFrameDataset(
        replay_dir=config['data']['replay_dir'],
        games=config['data']['games'],
        max_frames_per_game=config['data'].get('frames_per_game', 100_000),
    )

    val_split = config['training'].get('val_split', 0.05)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config['training']['seed'])
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def reconstruction_loss(
    encoder_feats: dict,
    quant_feats: dict,
    vq_loss: torch.Tensor,
    l0_weight: float,
    l1_weight: float,
    l2_weight: float,
) -> torch.Tensor:
    """
    MSE between encoder patch embeddings and quantised embeddings (STE).

    The STE ensures gradients flow through to the encoder.
    """
    loss_l0 = F.mse_loss(quant_feats['l0'], encoder_feats['l0'].detach()) \
            + F.mse_loss(encoder_feats['l0'], quant_feats['l0'].detach())
    loss_l1 = F.mse_loss(quant_feats['l1'], encoder_feats['l1'].detach()) \
            + F.mse_loss(encoder_feats['l1'], quant_feats['l1'].detach())
    loss_l2 = F.mse_loss(quant_feats['l2'], encoder_feats['l2'].detach()) \
            + F.mse_loss(encoder_feats['l2'], quant_feats['l2'].detach())

    return (l0_weight * loss_l0
            + l1_weight * loss_l1
            + l2_weight * loss_l2
            + vq_loss)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config_path: str, use_wandb: bool = False, device_str: str = 'cuda'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"\nSPATIAL ENCODER TRAINING")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Models
    d_model = config['model']['d_model']
    encoder = SpatialAtariEncoder(
        input_channels=config['model']['input_channels'],
        d_model=d_model,
    ).to(device)

    tokenizer = SpatialHRVQTokenizer(
        d_model=d_model,
        num_codes=config['tokenizer']['num_codes'],
        commitment_costs=config['tokenizer']['commitment_costs'],
        decay=config['tokenizer']['decay'],
        epsilon=config['tokenizer']['epsilon'],
    ).to(device)

    enc_params = encoder.count_parameters()
    tok_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"\nEncoder parameters:   {enc_params:,}")
    print(f"Tokenizer parameters: {tok_params:,}")

    # Optimizer — train encoder + tokenizer jointly
    lr = config['training']['learning_rate']
    betas = tuple(config['training']['betas'])
    optimizer = AdamW(
        list(encoder.parameters()) + list(tokenizer.parameters()),
        lr=lr, betas=betas,
        weight_decay=config['training']['weight_decay'],
    )

    scaler = GradScaler(enabled=config['training'].get('mixed_precision', True)
                        and device.type == 'cuda')

    # Data
    batch_size = config['training']['batch_size']
    print(f"\nLoading data...")
    train_loader, val_loader = build_dataloaders(config, batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Loss weights
    l0_w = config['loss']['l0_weight']
    l1_w = config['loss']['l1_weight']
    l2_w = config['loss']['l2_weight']
    grad_clip = config['training']['grad_clip']

    # WandB
    if use_wandb:
        import wandb
        wandb.init(project=config['logging'].get('wandb_project', 'spatial-encoder'),
                   config=config)

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_every = config['logging'].get('log_every', 100)

    best_val_loss = float('inf')
    step = 0

    num_epochs = config['training']['num_epochs']

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        encoder.train()
        tokenizer.train()
        train_loss_sum = 0.0
        train_steps = 0

        for batch in train_loader:
            frames = batch.to(device)   # (B, 4, 84, 84) float32 [0,1]

            optimizer.zero_grad()
            with autocast(enabled=scaler.is_enabled()):
                spatial_feats = encoder(frames)
                token_dict, vq_loss, quant_dict = tokenizer(spatial_feats)
                loss = reconstruction_loss(
                    spatial_feats, quant_dict, vq_loss, l0_w, l1_w, l2_w
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(tokenizer.parameters()),
                grad_clip
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_steps += 1
            step += 1

            if step % log_every == 0:
                avg = train_loss_sum / train_steps
                print(f"  Epoch {epoch}/{num_epochs}  Step {step}  "
                      f"train_loss={avg:.4f}  vq_loss={vq_loss.item():.4f}")
                if use_wandb:
                    import wandb
                    wandb.log({'train_loss': avg, 'vq_loss': vq_loss.item(), 'step': step})

        # --- Validation ---
        eval_every = config['training'].get('eval_every', 1)
        if epoch % eval_every == 0:
            encoder.eval()
            tokenizer.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch.to(device)
                    spatial_feats = encoder(frames)
                    token_dict, vq_loss_v, quant_dict = tokenizer(spatial_feats)
                    loss_v = reconstruction_loss(
                        spatial_feats, quant_dict, vq_loss_v, l0_w, l1_w, l2_w
                    )
                    val_loss_sum += loss_v.item()
                    val_steps += 1

            val_loss = val_loss_sum / max(val_steps, 1)
            usage = tokenizer.get_codebook_usage(token_dict)
            print(f"\nEpoch {epoch} — val_loss={val_loss:.4f}")
            for key, stat in usage.items():
                print(f"  codebook {key}: {stat['usage_pct']:.1f}% utilised")

            if use_wandb:
                import wandb
                wandb.log({'val_loss': val_loss, 'epoch': epoch})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'tokenizer_state_dict': tokenizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(ckpt, save_dir / 'spatial_encoder_best.pt')
                print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        save_every = config['logging'].get('save_every', 10)
        if epoch % save_every == 0:
            ckpt = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'tokenizer_state_dict': tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }
            torch.save(ckpt, save_dir / f'spatial_encoder_epoch{epoch}.pt')

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/encoder_spatial.yaml')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train(args.config, use_wandb=args.wandb, device_str=args.device)
