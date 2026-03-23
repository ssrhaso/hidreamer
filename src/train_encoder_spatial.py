"""
TRAIN SPATIAL ENCODER — Joint training of SpatialAtariEncoder + SpatialHRVQTokenizer.

COLLAPSE DIAGNOSIS:
  Pong frames are ~90% black background.  Plain MSE on embeddings is minimised by
  outputting the same "background" embedding for every patch → codebook collapses
  to 2-5 codes → loss = 0.0000 from epoch 1.

THREE FIXES (training objective only — architecture unchanged):

  1. Foreground-weighted pixel auxiliary loss
     A small training-only linear head (nn.Linear(384→1)) predicts each patch's
     mean pixel brightness.  Patches with foreground pixels get 50× higher gradient
     weight.  This is the key signal that forces the encoder to produce DIFFERENT
     embeddings for foreground vs background patches.

  2. Codebook entropy diversity penalty
     Soft codebook assignment entropy is computed per batch.  Low entropy (collapse)
     is penalised directly.  Drives the codebook to spread over all 256 entries.

  3. K-means codebook initialisation
     Before epoch 1, encoder outputs from 100 random batches are collected and
     a random subset of 256 is used to initialise each VQ layer's codebook.
     This puts the codebook in the right neighbourhood from the start and avoids
     the all-background degenerate fixed point.

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
# Dataset — returns raw uint8 frames (needed for foreground mask)
# ---------------------------------------------------------------------------
class AtariFrameDataset(torch.utils.data.Dataset):
    """
    Loads frames from data/{game}/frames.npy — shape (N, 4, 84, 84) uint8.
    Returns float32 [0,1] tensors AND the raw uint8 frame for fg-mask computation.
    """

    def __init__(self, replay_dir: str, games: list, max_frames_per_game: int = 100_000):
        arrays = []
        for game in games:
            game_dir = Path(replay_dir) / game
            loaded = False
            for fname in ['frames.npy', 'obs.npy', 'observations.npy']:
                fpath = game_dir / fname
                if fpath.exists():
                    data = np.load(str(fpath), mmap_mode='r')
                    n = min(len(data), max_frames_per_game)
                    arrays.append(data[:n])
                    print(f"  {game}: {n:,} frames from {fpath}")
                    loaded = True
                    break
            if not loaded:
                print(f"  WARNING: no frame data found for {game} in {game_dir}")

        if not arrays:
            raise FileNotFoundError(
                f"No frame data found in {replay_dir}. "
                "Expected data/{{game}}/frames.npy with shape (N, 4, 84, 84) uint8."
            )

        self.frames = np.concatenate(arrays, axis=0)   # (N_total, 4, 84, 84) uint8
        print(f"  Total frames: {len(self.frames):,}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        raw = self.frames[idx]                                    # (4, 84, 84) uint8
        f32 = torch.from_numpy(raw.astype(np.float32) / 255.0)   # [0,1]
        return f32


def build_dataloaders(config: dict, batch_size: int):
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
    return train_loader, val_loader, dataset


# ---------------------------------------------------------------------------
# Training-only auxiliary pixel head
# ---------------------------------------------------------------------------
class PixelAuxHead(nn.Module):
    """
    Predicts mean pixel brightness of a spatial patch from its embedding.
    Training-only — discarded after training completes.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, N, d_model) → (B, N) predicted mean pixel [0,1]"""
        return torch.sigmoid(self.linear(feats)).squeeze(-1)


# ---------------------------------------------------------------------------
# Foreground mask utilities
# ---------------------------------------------------------------------------
def compute_fg_masks(
    frames: torch.Tensor,    # (B, 4, 84, 84) float32 [0,1]
    fg_threshold: float,
) -> dict:
    """
    Returns per-patch foreground fraction at L0 (2×2) and L1/L2 (4×4) resolutions.
    Uses the mean across the 4 stacked frames as the foreground signal.

    Returns
    -------
    dict:
        'l0': (B, 4)  — fg fraction per L0 patch
        'l1': (B, 16) — fg fraction per L1/L2 patch
    """
    # Mean across frame stack: (B, 84, 84) → (B, 1, 84, 84)
    mean_frame = frames.mean(dim=1, keepdim=True)   # (B, 1, 84, 84)
    fg_mask = (mean_frame > fg_threshold).float()   # (B, 1, 84, 84)

    fg_l0 = F.adaptive_avg_pool2d(fg_mask, (2, 2)).squeeze(1).reshape(-1, 4)    # (B, 4)
    fg_l1 = F.adaptive_avg_pool2d(fg_mask, (4, 4)).squeeze(1).reshape(-1, 16)   # (B, 16)

    return {'l0': fg_l0, 'l1': fg_l1}


def compute_patch_mean_pixels(
    frames: torch.Tensor,    # (B, 4, 84, 84) float32 [0,1]
) -> dict:
    """
    Target mean pixel brightness per spatial patch for the pixel aux loss.

    Returns
    -------
    dict:
        'l0': (B, 4)  — mean pixel per L0 patch (2×2 grid)
        'l1': (B, 16) — mean pixel per L1 patch (4×4 grid)
        'l2': (B, 16) — mean pixel per L2 patch (4×4 grid)
    """
    mean_frame = frames.mean(dim=1, keepdim=True)   # (B, 1, 84, 84)

    p_l0 = F.adaptive_avg_pool2d(mean_frame, (2, 2)).squeeze(1).reshape(-1, 4)
    p_l1 = F.adaptive_avg_pool2d(mean_frame, (4, 4)).squeeze(1).reshape(-1, 16)

    return {'l0': p_l0, 'l1': p_l1, 'l2': p_l1}  # L1 and L2 share same 4×4 target


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def weighted_per_patch_mse(
    feats_enc: torch.Tensor,    # (B, N, D)
    feats_q:   torch.Tensor,    # (B, N, D)
    weights:   torch.Tensor,    # (B, N) — per-patch weights
) -> torch.Tensor:
    """
    Per-patch MSE weighted by foreground content.

    Standard MSE = mean(||enc - q||²).
    Weighted MSE = mean(w * ||enc - q||²) where w is larger for foreground patches.
    """
    # (B, N) — per-patch squared error averaged over embedding dim
    per_patch = (feats_enc - feats_q).pow(2).mean(dim=-1)
    return (weights * per_patch).mean()


def pixel_aux_loss(
    pixel_head: PixelAuxHead,
    quant_feats: torch.Tensor,   # (B, N, D) — STE: grads flow back to encoder
    target_pixels: torch.Tensor, # (B, N) — actual mean pixel brightness
    patch_weights: torch.Tensor, # (B, N) — fg-weighted
) -> torch.Tensor:
    """
    MSE between predicted patch brightness (from quantised embedding) and actual.
    Uses quant_feats (not encoder_feats) so gradient flows through STE to encoder.
    """
    pred = pixel_head(quant_feats)                  # (B, N)
    per_patch = (pred - target_pixels).pow(2)       # (B, N)
    return (patch_weights * per_patch).mean()


def codebook_diversity_loss(
    vq_layer,
    feats: torch.Tensor,    # (B, N, D) — encoder embeddings (pre-quantise)
    temperature: float,
) -> torch.Tensor:
    """
    Entropy-based codebook diversity penalty.

    Computes soft assignment distribution over the codebook for each patch in
    the batch.  Low entropy = few codes used = penalty is high.
    Returns a value in [0, 1]: 0 = fully spread, 1 = all patches → same code.
    """
    B, N, D = feats.shape
    flat = feats.detach().reshape(-1, D)      # (B*N, D) — no grad through this term

    # Squared distances to each codebook entry
    dists = (
        flat.pow(2).sum(1, keepdim=True)               # (B*N, 1)
        + vq_layer.codebook.weight.pow(2).sum(1)        # (num_codes,)
        - 2.0 * flat @ vq_layer.codebook.weight.t()    # (B*N, num_codes)
    )  # (B*N, num_codes)

    # Soft assignments
    soft = F.softmax(-dists / temperature, dim=-1)     # (B*N, num_codes)
    marginal = soft.mean(dim=0)                        # (num_codes,) — average usage

    # Normalised entropy: 0=collapsed, 1=uniform
    eps = 1e-8
    entropy = -(marginal * (marginal + eps).log()).sum()
    max_entropy = torch.log(
        torch.tensor(float(vq_layer.num_codebook_entries), device=feats.device)
    )
    # Loss = 1 - normalised_entropy (minimise → maximise entropy)
    return 1.0 - entropy / max_entropy.clamp(min=eps)


def compute_loss(
    frames:          torch.Tensor,   # (B, 4, 84, 84) [0,1]
    encoder_feats:   dict,
    quant_feats:     dict,
    vq_loss:         torch.Tensor,
    tokenizer:       SpatialHRVQTokenizer,
    pixel_heads:     dict,           # {'l0': PixelAuxHead, 'l1': ..., 'l2': ...}
    cfg_loss:        dict,
) -> tuple:
    """
    Full loss combining:
      1. Foreground-weighted commitment MSE (encoder ↔ codebook)
      2. Pixel auxiliary loss (predicting patch brightness via STE)
      3. Codebook entropy diversity penalty
    """
    l0_w      = cfg_loss['l0_weight']
    l1_w      = cfg_loss['l1_weight']
    l2_w      = cfg_loss['l2_weight']
    fg_boost  = cfg_loss['fg_boost']
    fg_thr    = cfg_loss['fg_threshold']
    px_w      = cfg_loss['pixel_aux_weight']
    div_w     = cfg_loss['diversity_weight']
    div_temp  = cfg_loss['diversity_temp']

    # --- Foreground masks and pixel targets ---
    fg = compute_fg_masks(frames, fg_thr)               # 'l0':(B,4), 'l1':(B,16)
    px_target = compute_patch_mean_pixels(frames)        # 'l0','l1','l2' each (B,N)

    # Per-patch weights: background=1.0, foreground=fg_boost
    def patch_w(fg_frac):   # (B, N) → (B, N)
        return 1.0 + (fg_boost - 1.0) * fg_frac

    pw_l0 = patch_w(fg['l0'])        # (B, 4)
    pw_l1 = patch_w(fg['l1'])        # (B, 16)

    # --- 1. Foreground-weighted commitment MSE ---
    mse_l0 = weighted_per_patch_mse(encoder_feats['l0'], quant_feats['l0'].detach(), pw_l0) \
            + weighted_per_patch_mse(quant_feats['l0'], encoder_feats['l0'].detach(), pw_l0)
    mse_l1 = weighted_per_patch_mse(encoder_feats['l1'], quant_feats['l1'].detach(), pw_l1) \
            + weighted_per_patch_mse(quant_feats['l1'], encoder_feats['l1'].detach(), pw_l1)
    mse_l2 = weighted_per_patch_mse(encoder_feats['l2'], quant_feats['l2'].detach(), pw_l1) \
            + weighted_per_patch_mse(quant_feats['l2'], encoder_feats['l2'].detach(), pw_l1)

    commit_loss = l0_w * mse_l0 + l1_w * mse_l1 + l2_w * mse_l2 + vq_loss

    # --- 2. Pixel auxiliary loss ---
    # gradient flows: quant_feats (STE) → pixel_head → MSE → encoder
    px_l0 = pixel_aux_loss(pixel_heads['l0'], quant_feats['l0'], px_target['l0'], pw_l0)
    px_l1 = pixel_aux_loss(pixel_heads['l1'], quant_feats['l1'], px_target['l1'], pw_l1)
    px_l2 = pixel_aux_loss(pixel_heads['l2'], quant_feats['l2'], px_target['l2'], pw_l1)
    aux_loss = l0_w * px_l0 + l1_w * px_l1 + l2_w * px_l2

    # --- 3. Codebook entropy diversity penalty ---
    div_l0 = codebook_diversity_loss(tokenizer.vq_l0, encoder_feats['l0'], div_temp)
    div_l1 = codebook_diversity_loss(tokenizer.vq_l1, encoder_feats['l1'], div_temp)
    div_l2 = codebook_diversity_loss(tokenizer.vq_l2, encoder_feats['l2'], div_temp)
    div_loss = (div_l0 + div_l1 + div_l2) / 3.0

    total = commit_loss + px_w * aux_loss + div_w * div_loss

    info = {
        'loss_commit': commit_loss.item(),
        'loss_pixel':  aux_loss.item(),
        'loss_div':    div_loss.item(),
        'loss_total':  total.item(),
        'entropy_l0':  1.0 - div_l0.item(),   # 0=collapsed, 1=uniform
        'entropy_l1':  1.0 - div_l1.item(),
        'entropy_l2':  1.0 - div_l2.item(),
    }
    return total, info


# ---------------------------------------------------------------------------
# K-means codebook initialisation
# ---------------------------------------------------------------------------
@torch.no_grad()
def kmeans_init_codebooks(
    encoder:    SpatialAtariEncoder,
    tokenizer:  SpatialHRVQTokenizer,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    num_batches: int,
):
    """
    Collect encoder outputs from num_batches random batches and use a random
    subset of 256 embeddings to initialise each VQ layer's codebook.

    This breaks the all-background fixed point by starting the codebook in the
    data manifold rather than at the default uniform-random initialisation.
    """
    print(f"\n  K-means codebook init ({num_batches} batches)...")
    encoder.eval()

    buffers = {'l0': [], 'l1': [], 'l2': []}

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        frames = batch.to(device)
        feats = encoder(frames)
        buffers['l0'].append(feats['l0'].reshape(-1, feats['l0'].size(-1)).cpu())
        buffers['l1'].append(feats['l1'].reshape(-1, feats['l1'].size(-1)).cpu())
        buffers['l2'].append(feats['l2'].reshape(-1, feats['l2'].size(-1)).cpu())

    for key, vq_layer in [('l0', tokenizer.vq_l0),
                           ('l1', tokenizer.vq_l1),
                           ('l2', tokenizer.vq_l2)]:
        all_embs = torch.cat(buffers[key], dim=0)   # (N_total, D)
        num_codes = vq_layer.num_codebook_entries

        if len(all_embs) < num_codes:
            print(f"    {key}: only {len(all_embs)} embeddings < {num_codes} codes, skipping")
            continue

        # Random subset as initial codebook entries
        idx = torch.randperm(len(all_embs))[:num_codes]
        init_codes = all_embs[idx].to(device)       # (num_codes, D)

        vq_layer.codebook.weight.data.copy_(init_codes)
        vq_layer.ema_weight.copy_(init_codes)
        # Reset cluster sizes to uniform so EMA starts fresh
        vq_layer.ema_cluster_size.fill_(1.0)

        print(f"    {key}: initialised from {len(all_embs):,} embeddings  "
              f"(norm range [{init_codes.norm(dim=1).min():.3f}, "
              f"{init_codes.norm(dim=1).max():.3f}])")

    encoder.train()


# ---------------------------------------------------------------------------
# Codebook utilisation measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_codebook_usage(
    encoder:   SpatialAtariEncoder,
    tokenizer: SpatialHRVQTokenizer,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    num_batches: int = 20,
) -> dict:
    """
    Run num_batches through the encoder+tokenizer in eval mode and return
    the fraction of codebook entries that were assigned at least once.
    """
    encoder.eval()
    tokenizer.eval()

    used = {'l0': set(), 'l1': set(), 'l2': set()}
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        frames = batch.to(device)
        feats = encoder(frames)
        tokens = tokenizer.encode(feats)
        for key in ['l0', 'l1', 'l2']:
            used[key].update(tokens[key].cpu().numpy().flatten().tolist())

    totals = {'l0': tokenizer.num_codes_l0, 'l1': tokenizer.num_codes_l1, 'l2': tokenizer.num_codes_l2}
    stats = {
        key: {
            'unique': len(used[key]),
            'pct':    100.0 * len(used[key]) / totals[key],
        }
        for key in ['l0', 'l1', 'l2']
    }
    encoder.train()
    tokenizer.train()
    return stats


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
    print(f"\nSPATIAL ENCODER TRAINING  (collapse-fixed)")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    d_model = config['model']['d_model']

    # Build models
    encoder = SpatialAtariEncoder(
        input_channels=config['model']['input_channels'],
        d_model=d_model,
    ).to(device)

    tokenizer = SpatialHRVQTokenizer(
        d_model=d_model,
        num_codes_l0=config['tokenizer']['num_codes_l0'],
        num_codes_l1=config['tokenizer']['num_codes_l1'],
        num_codes_l2=config['tokenizer']['num_codes_l2'],
        commitment_costs=config['tokenizer']['commitment_costs'],
        decay=config['tokenizer']['decay'],
        epsilon=config['tokenizer']['epsilon'],
    ).to(device)

    # Training-only pixel auxiliary heads (NOT saved in checkpoint)
    pixel_heads = {
        'l0': PixelAuxHead(d_model).to(device),
        'l1': PixelAuxHead(d_model).to(device),
        'l2': PixelAuxHead(d_model).to(device),
    }

    enc_params = encoder.count_parameters()
    tok_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    px_params  = sum(p.numel() for h in pixel_heads.values()
                     for p in h.parameters() if p.requires_grad)
    print(f"\nEncoder parameters:         {enc_params:,}")
    print(f"Tokenizer parameters:       {tok_params:,}")
    print(f"Pixel head parameters:      {px_params:,}  (training-only, discarded)")

    # Optimizer — encoder + tokenizer + pixel heads trained jointly
    lr    = config['training']['learning_rate']
    betas = tuple(config['training']['betas'])
    all_params = (
        list(encoder.parameters())
        + list(tokenizer.parameters())
        + [p for h in pixel_heads.values() for p in h.parameters()]
    )
    optimizer = AdamW(all_params, lr=lr, betas=betas,
                      weight_decay=config['training']['weight_decay'])

    scaler = GradScaler(
        enabled=config['training'].get('mixed_precision', True) and device.type == 'cuda'
    )

    # Data
    batch_size = config['training']['batch_size']
    print(f"\nLoading data...")
    train_loader, val_loader, dataset = build_dataloaders(config, batch_size)
    print(f"Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # K-means codebook initialisation
    ki_cfg = config.get('kmeans_init', {})
    if ki_cfg.get('enabled', True):
        kmeans_init_codebooks(
            encoder, tokenizer, train_loader, device,
            num_batches=ki_cfg.get('init_batches', 100),
        )

    # WandB
    if use_wandb:
        import wandb
        wandb.init(project=config['logging'].get('wandb_project', 'spatial-encoder'),
                   config=config)

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_every  = config['logging'].get('log_every', 100)
    save_every = config['logging'].get('save_every', 10)
    eval_every = config['training'].get('eval_every', 1)

    cfg_loss  = config['loss']
    grad_clip = config['training']['grad_clip']
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    step = 0

    # Running averages for step-level logging
    running = {k: 0.0 for k in ['loss_commit', 'loss_pixel', 'loss_div', 'loss_total']}
    running_n = 0

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        tokenizer.train()
        for h in pixel_heads.values():
            h.train()

        for batch in train_loader:
            frames = batch.to(device)   # (B, 4, 84, 84) float32 [0,1]

            optimizer.zero_grad()
            with autocast(enabled=scaler.is_enabled()):
                spatial_feats = encoder(frames)
                token_dict, vq_loss, quant_dict = tokenizer(spatial_feats)
                loss, info = compute_loss(
                    frames, spatial_feats, quant_dict, vq_loss,
                    tokenizer, pixel_heads, cfg_loss,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate running averages
            for k in running:
                running[k] += info.get(k, 0.0)
            running_n += 1
            step += 1

            if step % log_every == 0:
                avgs = {k: running[k] / running_n for k in running}
                running = {k: 0.0 for k in running}
                running_n = 0
                print(
                    f"  Ep {epoch}/{num_epochs}  Step {step:6d} | "
                    f"total={avgs['loss_total']:.4f}  "
                    f"commit={avgs['loss_commit']:.4f}  "
                    f"pixel={avgs['loss_pixel']:.4f}  "
                    f"div={avgs['loss_div']:.4f}  "
                    f"entropy=[{info['entropy_l0']:.2f},"
                    f"{info['entropy_l1']:.2f},{info['entropy_l2']:.2f}]"
                )
                if use_wandb:
                    import wandb
                    wandb.log({**avgs, 'step': step,
                               'entropy_l0': info['entropy_l0'],
                               'entropy_l1': info['entropy_l1'],
                               'entropy_l2': info['entropy_l2']})

        # --- Validation & codebook stats ---
        if epoch % eval_every == 0:
            encoder.eval()
            tokenizer.eval()
            for h in pixel_heads.values():
                h.eval()

            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch.to(device)
                    spatial_feats = encoder(frames)
                    token_dict_v, vq_loss_v, quant_dict_v = tokenizer(spatial_feats)
                    loss_v, _ = compute_loss(
                        frames, spatial_feats, quant_dict_v, vq_loss_v,
                        tokenizer, pixel_heads, cfg_loss,
                    )
                    val_loss_sum += loss_v.item()
                    val_n += 1

            val_loss = val_loss_sum / max(val_n, 1)

            # Codebook utilisation (over 20 train batches in eval mode)
            usage = measure_codebook_usage(encoder, tokenizer, train_loader, device)
            print(f"\nEpoch {epoch} — val_loss={val_loss:.4f}")
            for key, stat in usage.items():
                print(f"  codebook {key}: {stat['unique']:3d}/256 ({stat['pct']:.1f}%) used")

            if use_wandb:
                import wandb
                log_dict = {'val_loss': val_loss, 'epoch': epoch}
                for key, stat in usage.items():
                    log_dict[f'codebook_usage_{key}_pct'] = stat['pct']
                    log_dict[f'codebook_usage_{key}_unique'] = stat['unique']
                wandb.log(log_dict)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    'epoch': epoch,
                    'encoder_state_dict':   encoder.state_dict(),
                    'tokenizer_state_dict': tokenizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss':        best_val_loss,
                    'config':               config,
                    # NOTE: pixel_heads NOT saved — training-only
                }
                torch.save(ckpt, save_dir / 'spatial_encoder_best.pt')
                print(f"  Saved best  (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % save_every == 0:
            ckpt = {
                'epoch': epoch,
                'encoder_state_dict':   encoder.state_dict(),
                'tokenizer_state_dict': tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config':               config,
            }
            torch.save(ckpt, save_dir / f'spatial_encoder_epoch{epoch}.pt')

    print(f"\nTraining complete.  Best val_loss: {best_val_loss:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/encoder_spatial.yaml')
    parser.add_argument('--wandb',  action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train(args.config, use_wandb=args.wandb, device_str=args.device)
