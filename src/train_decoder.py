"""
DECODER PRE-TRAINING - TRAIN FRAMEDECODER ON REAL REPLAY DATA

Usage:
    cd dreamer/
    python src/train_decoder.py --epochs 20 --batch_size 256 --lr 1e-3
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vq import HRVQTokenizer
from decoder import FrameDecoder


# DATASET
class ReplayDecoderDataset(Dataset):
    """ PAIRS OF (TOKEN_TRIPLE, TARGET_FRAME) FOR SUPERVISED DECODER TRAINING """

    def __init__(
        self,
        replay_path : str,
        tokens_dir  : str,
        game        : str = "ALE_Pong-v5",
    ):
        replay = np.load(replay_path)
        # LAST CHANNEL OF 4-FRAME STACK (MOST RECENT OBSERVATION)
        self.frames  = replay['states'][:, 3, :, :].astype(np.float32) / 255.0  # (N, 84, 84)
        self.rewards = replay['rewards'].astype(np.float32)                       # (N,)

        # PRE-COMPUTED TOKEN INDICES
        l0 = np.load(f"{tokens_dir}/vq_tokens_{game}_layer0.npy").squeeze()  # (N,)
        l1 = np.load(f"{tokens_dir}/vq_tokens_{game}_layer1.npy").squeeze()  # (N,)
        l2 = np.load(f"{tokens_dir}/vq_tokens_{game}_layer2.npy").squeeze()  # (N,)
        self.tokens = np.stack([l0, l1, l2], axis=1).astype(np.int64)        # (N, 3)

        assert len(self.frames) == len(self.tokens), "Frame/token count mismatch"
        print(f"  Dataset: {len(self.frames):,} samples loaded")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame  = torch.from_numpy(self.frames[idx]).unsqueeze(0)  # (1, 84, 84)
        tokens = torch.from_numpy(self.tokens[idx])               # (3,)
        reward = self.rewards[idx]
        return frame, tokens, reward


# LOSS
def weighted_recon_loss(
    pred_coarse : torch.Tensor,
    pred_mid    : torch.Tensor,
    pred_full   : torch.Tensor,
    target      : torch.Tensor,
    w_coarse    : float = 0.2,
    w_mid       : float = 0.3,
    w_full      : float = 0.5,
) -> torch.Tensor:
    """ WEIGHTED MSE ACROSS THREE RECONSTRUCTION LEVELS """
    loss  = w_coarse * F.mse_loss(pred_coarse, target)
    loss += w_mid    * F.mse_loss(pred_mid,    target)
    loss += w_full   * F.mse_loss(pred_full,   target)
    return loss


# TRAINING LOOP
def train_decoder(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDECODER PRE-TRAINING")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")

    # LOAD FROZEN HRVQ
    print("\nLoading frozen HRVQ tokenizer...")
    hrvq = HRVQTokenizer(
        input_dim=384, num_codes_per_layer=256, num_layers=3,
        commitment_costs=[0.05, 0.25, 0.60],
    ).to(device)
    state = torch.load(args.hrvq_path, map_location=device, weights_only=False)
    hrvq.load_state_dict(state)
    hrvq.eval()
    for p in hrvq.parameters():
        p.requires_grad = False
    print(f"  HRVQ loaded and FROZEN")

    # DATASET AND DATALOADER
    print("\nLoading replay dataset...")
    dataset = ReplayDecoderDataset(
        replay_path = args.replay_path,
        tokens_dir  = args.tokens_dir,
        game        = args.game,
    )
    val_size  = min(5000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {train_size:,}  Val: {val_size:,}")

    # MODEL
    decoder = FrameDecoder(emb_dim=384, base_ch=128).to(device)
    print(f"\n  FrameDecoder params: {decoder.count_parameters():,}")

    optimizer = Adam(decoder.parameters(), lr=args.lr, eps=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # MAIN LOOP
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        train_loss = 0.0
        n_batches  = 0

        for frames, tokens, _ in train_loader:
            frames = frames.to(device)          # (B, 1, 84, 84)
            tokens = tokens.to(device)          # (B, 3)

            # CODEBOOK EMBEDDING LOOKUP (NO GRAD - CODEBOOK IS FROZEN)
            with torch.no_grad():
                emb_l0 = hrvq.vq_layers[0].codebook(tokens[:, 0])  # (B, 384)
                emb_l1 = hrvq.vq_layers[1].codebook(tokens[:, 1])  # (B, 384)
                emb_l2 = hrvq.vq_layers[2].codebook(tokens[:, 2])  # (B, 384)

            # THREE RECONSTRUCTION LEVELS
            pred_coarse = decoder(emb_l0)                      # L0 ONLY
            pred_mid    = decoder(emb_l0 + emb_l1)             # L0 + L1
            pred_full   = decoder(emb_l0 + emb_l1 + emb_l2)   # L0 + L1 + L2

            loss = weighted_recon_loss(pred_coarse, pred_mid, pred_full, frames)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)

        # VALIDATION
        decoder.eval()
        val_loss   = 0.0
        full_psnrs = []

        with torch.no_grad():
            for frames, tokens, _ in val_loader:
                frames = frames.to(device)
                tokens = tokens.to(device)
                emb_l0 = hrvq.vq_layers[0].codebook(tokens[:, 0])
                emb_l1 = hrvq.vq_layers[1].codebook(tokens[:, 1])
                emb_l2 = hrvq.vq_layers[2].codebook(tokens[:, 2])
                pred_coarse = decoder(emb_l0)
                pred_mid    = decoder(emb_l0 + emb_l1)
                pred_full   = decoder(emb_l0 + emb_l1 + emb_l2)
                loss = weighted_recon_loss(pred_coarse, pred_mid, pred_full, frames)
                val_loss += loss.item()
                # PSNR FOR FULL RECONSTRUCTION
                mse = F.mse_loss(pred_full, frames).item()
                psnr = -10.0 * np.log10(mse + 1e-8)
                full_psnrs.append(psnr)

        val_loss /= max(len(val_loader), 1)
        psnr_mean = float(np.mean(full_psnrs))

        scheduler.step()

        print(f"  Epoch {epoch:>3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"PSNR(full)={psnr_mean:.2f}dB")

        # CHECKPOINT
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(args.save_dir, "decoder_best.pt")
            torch.save({
                'epoch'     : epoch,
                'val_loss'  : val_loss,
                'psnr_full' : psnr_mean,
                'model_state_dict': decoder.state_dict(),
                'config'    : {'emb_dim': 384, 'base_ch': 128},
            }, path)
            print(f"    Saved best checkpoint (val_loss={val_loss:.5f})")

    # SAVE RECONSTRUCTION GRID
    _save_reconstruction_grid(
        decoder  = decoder,
        hrvq     = hrvq,
        dataset  = dataset,
        device   = device,
        save_dir = args.save_dir,
        n        = 8,
    )

    print(f"\nDECODER TRAINING COMPLETE")
    print(f"  Best val_loss: {best_val_loss:.5f}")
    print(f"  Checkpoint: {os.path.join(args.save_dir, 'decoder_best.pt')}")
    print(f"  Reconstruction grid: {os.path.join(args.save_dir, 'decoder_recon_grid.png')}")

    return best_val_loss


def _save_reconstruction_grid(decoder, hrvq, dataset, device, save_dir, n=8):
    """ SAVE GRID OF ORIGINAL AND RECONSTRUCTED FRAMES AT THREE LEVELS """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping reconstruction grid)")
        return

    decoder.eval()
    indices = np.random.choice(len(dataset), size=n, replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    col_titles = ['Original', 'Coarse (L0)', 'Mid (L0+L1)', 'Full (L0+L1+L2)']

    for row, idx in enumerate(indices):
        frame, tokens, reward = dataset[idx]
        frame  = frame.unsqueeze(0).to(device)    # (1, 1, 84, 84)
        tokens = tokens.unsqueeze(0).to(device)   # (1, 3)

        with torch.no_grad():
            emb_l0 = hrvq.vq_layers[0].codebook(tokens[:, 0])
            emb_l1 = hrvq.vq_layers[1].codebook(tokens[:, 1])
            emb_l2 = hrvq.vq_layers[2].codebook(tokens[:, 2])
            coarse = decoder(emb_l0).cpu().squeeze().numpy()
            mid    = decoder(emb_l0 + emb_l1).cpu().squeeze().numpy()
            full   = decoder(emb_l0 + emb_l1 + emb_l2).cpu().squeeze().numpy()

        orig = frame.cpu().squeeze().numpy()
        rew_str = f"r={reward:.0f}" if abs(reward) > 0.01 else ""

        for col, (img, title) in enumerate(zip(
            [orig, coarse, mid, full], col_titles
        )):
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)
            if col == 0 and rew_str:
                axes[row, col].set_ylabel(rew_str, fontsize=9)
            axes[row, col].axis('off')

    plt.suptitle("Decoder Reconstruction Quality\n"
                 "Good: paddle bar + ball dot visible in 'Full' column\n"
                 "Bad: uniform grey blur (spatial info lost at tokenization)",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'decoder_recon_grid.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Reconstruction grid saved -> {path}")


# ENTRY POINT
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str,   default="cuda")
    p.add_argument("--game",        type=str,   default="ALE_Pong-v5")
    p.add_argument("--replay_path", type=str,   default="data/replay_buffer_ALE_Pong-v5.npz")
    p.add_argument("--tokens_dir",  type=str,   default="checkpoints/rsvq_tokens")
    p.add_argument("--hrvq_path",   type=str,   default="checkpoints/rsvq_model_best.pth")
    p.add_argument("--save_dir",    type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_decoder(args)
