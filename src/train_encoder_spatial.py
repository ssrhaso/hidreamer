"""
TRAIN SPATIAL ENCODER - JOINT TRAINING OF SPATIALATARENCODER AND SPATIALHRVQTOKENIZER

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


# DATASET - RETURNS FLOAT32 [0,1] FRAMES
class AtariFrameDataset(torch.utils.data.Dataset):
    """ LOADS FRAMES FROM DATA/{GAME}/FRAMES.NPY AND RETURNS FLOAT32 [0,1] TENSORS """

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


# TRAINING-ONLY AUXILIARY PIXEL HEAD
class PixelAuxHead(nn.Module):
    """ PREDICTS MEAN PIXEL BRIGHTNESS OF A PATCH FROM ITS EMBEDDING """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(feats)).squeeze(-1)


# FOREGROUND MASK UTILITIES
def compute_fg_masks(
    frames: torch.Tensor,
    fg_threshold: float,
) -> dict:
    """ RETURN PER-PATCH FOREGROUND FRACTION AT L0 AND L1/L2 RESOLUTIONS """
    # MEAN ACROSS FRAME STACK
    mean_frame = frames.mean(dim=1, keepdim=True)   # (B, 1, 84, 84)
    fg_mask = (mean_frame > fg_threshold).float()   # (B, 1, 84, 84)

    fg_l0 = F.adaptive_avg_pool2d(fg_mask, (2, 2)).squeeze(1).reshape(-1, 4)    # (B, 4)
    fg_l1 = F.adaptive_avg_pool2d(fg_mask, (4, 4)).squeeze(1).reshape(-1, 16)   # (B, 16)

    return {'l0': fg_l0, 'l1': fg_l1}


def compute_patch_mean_pixels(
    frames: torch.Tensor,
) -> dict:
    """ RETURN TARGET MEAN PIXEL BRIGHTNESS PER SPATIAL PATCH """
    mean_frame = frames.mean(dim=1, keepdim=True)   # (B, 1, 84, 84)

    p_l0 = F.adaptive_avg_pool2d(mean_frame, (2, 2)).squeeze(1).reshape(-1, 4)
    p_l1 = F.adaptive_avg_pool2d(mean_frame, (4, 4)).squeeze(1).reshape(-1, 16)

    return {'l0': p_l0, 'l1': p_l1, 'l2': p_l1}  # L1 AND L2 SHARE SAME 4x4 TARGET


# LOSS FUNCTIONS
def weighted_per_patch_mse(
    feats_enc: torch.Tensor,
    feats_q:   torch.Tensor,
    weights:   torch.Tensor,
) -> torch.Tensor:
    """ PER-PATCH MSE WEIGHTED BY FOREGROUND CONTENT """
    # PER-PATCH SQUARED ERROR AVERAGED OVER EMBEDDING DIM
    per_patch = (feats_enc - feats_q).pow(2).mean(dim=-1)
    return (weights * per_patch).mean()


def pixel_aux_loss(
    pixel_head: PixelAuxHead,
    quant_feats: torch.Tensor,
    target_pixels: torch.Tensor,
    patch_weights: torch.Tensor,
) -> torch.Tensor:
    """ MSE BETWEEN PREDICTED PATCH BRIGHTNESS AND ACTUAL """
    pred = pixel_head(quant_feats)                  # (B, N)
    per_patch = (pred - target_pixels).pow(2)       # (B, N)
    return (patch_weights * per_patch).mean()


def patch_diversity_loss_fn(
    encoder_feats: dict,
) -> torch.Tensor:
    """ PENALIZE HIGH COSINE SIMILARITY BETWEEN PATCHES IN SAME FRAME """
    device = next(iter(encoder_feats.values())).device
    total = torch.tensor(0.0, device=device)
    n_levels = 0

    for key in ['l0', 'l1', 'l2']:
        feats = encoder_feats[key]   # (B, N, D)
        B, N, D = feats.shape

        # PAIRWISE COSINE SIMILARITY
        sim = torch.bmm(feats, feats.transpose(1, 2))

        # ZERO OUT DIAGONAL
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        off_diag_sum = sim.masked_fill(eye, 0.0).sum()
        mean_sim = off_diag_sum / (B * N * (N - 1))

        total = total + mean_sim
        n_levels += 1

    return total / n_levels


def token_entropy_loss_fn(
    encoder_feats: dict,
    tokenizer,
    temperature: float = 0.5,
) -> torch.Tensor:
    """ PER-FRAME CODEBOOK ENTROPY LOSS - PENALIZES LOW ENTROPY """
    device = next(iter(encoder_feats.values())).device
    total = torch.tensor(0.0, device=device)
    n_levels = 0

    for key, vq_layer in [('l0', tokenizer.vq_l0),
                           ('l1', tokenizer.vq_l1),
                           ('l2', tokenizer.vq_l2)]:
        feats = encoder_feats[key]   # (B, N, D)
        B, N, D = feats.shape
        flat = feats.reshape(B * N, D)

        # SQUARED DISTANCES
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            + vq_layer.codebook.weight.pow(2).sum(1)
            - 2.0 * flat @ vq_layer.codebook.weight.t()
        )

        soft = F.softmax(-dists / temperature, dim=-1)  # (B*N, num_codes)
        soft = soft.reshape(B, N, -1)                   # (B, N, num_codes)

        # PER-FRAME MARGINAL
        marginal = soft.mean(dim=1)

        eps = 1e-8
        entropy = -(marginal * (marginal + eps).log()).sum(dim=-1)
        max_ent = torch.log(
            torch.tensor(float(vq_layer.num_codebook_entries), device=device)
        )
        # LOSS = MEAN(1 - NORMALISED_ENTROPY)
        level_loss = (1.0 - entropy / max_ent.clamp(min=eps)).mean()
        total = total + level_loss
        n_levels += 1

    return total / n_levels


def codebook_diversity_loss(
    vq_layer,
    feats: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """ ENTROPY-BASED CODEBOOK DIVERSITY PENALTY - 0=SPREAD, 1=COLLAPSED """
    B, N, D = feats.shape
    flat = feats.detach().reshape(-1, D)

    # SQUARED DISTANCES TO EACH CODEBOOK ENTRY
    dists = (
        flat.pow(2).sum(1, keepdim=True)
        + vq_layer.codebook.weight.pow(2).sum(1)
        - 2.0 * flat @ vq_layer.codebook.weight.t()
    )

    # SOFT ASSIGNMENTS
    soft = F.softmax(-dists / temperature, dim=-1)
    marginal = soft.mean(dim=0)

    # NORMALISED ENTROPY
    eps = 1e-8
    entropy = -(marginal * (marginal + eps).log()).sum()
    max_entropy = torch.log(
        torch.tensor(float(vq_layer.num_codebook_entries), device=feats.device)
    )
    return 1.0 - entropy / max_entropy.clamp(min=eps)


def compute_loss(
    frames:          torch.Tensor,
    encoder_feats:   dict,
    quant_feats:     dict,
    vq_loss:         torch.Tensor,
    token_dict:      dict,
    tokenizer:       SpatialHRVQTokenizer,
    pixel_heads:     dict,
    cfg_loss:        dict,
    cfg_training:    dict,
) -> tuple:
    """ FULL LOSS COMBINING COMMITMENT, PIXEL AUX, ENTROPY, AND DIVERSITY TERMS """
    l0_w      = cfg_loss['l0_weight']
    l1_w      = cfg_loss['l1_weight']
    l2_w      = cfg_loss['l2_weight']
    fg_boost  = cfg_loss['fg_boost']
    fg_thr    = cfg_loss['fg_threshold']
    px_w      = cfg_loss['pixel_aux_weight']
    div_w     = cfg_loss['diversity_weight']
    div_temp  = cfg_loss['diversity_temp']
    patch_div_w = cfg_training.get('patch_diversity_weight', 0.0)
    ent_w       = cfg_training.get('entropy_weight', 0.0)

    # FOREGROUND MASKS AND PIXEL TARGETS
    fg = compute_fg_masks(frames, fg_thr)
    px_target = compute_patch_mean_pixels(frames)

    # PER-PATCH WEIGHTS: BACKGROUND=1.0, FOREGROUND=FG_BOOST
    def patch_w(fg_frac):
        return 1.0 + (fg_boost - 1.0) * fg_frac

    pw_l0 = patch_w(fg['l0'])        # (B, 4)
    pw_l1 = patch_w(fg['l1'])        # (B, 16)

    # 1. FOREGROUND-WEIGHTED COMMITMENT MSE
    mse_l0 = weighted_per_patch_mse(encoder_feats['l0'], quant_feats['l0'].detach(), pw_l0) \
            + weighted_per_patch_mse(quant_feats['l0'], encoder_feats['l0'].detach(), pw_l0)
    mse_l1 = weighted_per_patch_mse(encoder_feats['l1'], quant_feats['l1'].detach(), pw_l1) \
            + weighted_per_patch_mse(quant_feats['l1'], encoder_feats['l1'].detach(), pw_l1)
    mse_l2 = weighted_per_patch_mse(encoder_feats['l2'], quant_feats['l2'].detach(), pw_l1) \
            + weighted_per_patch_mse(quant_feats['l2'], encoder_feats['l2'].detach(), pw_l1)

    commit_loss = l0_w * mse_l0 + l1_w * mse_l1 + l2_w * mse_l2 + vq_loss

    # 2. PIXEL AUXILIARY LOSS
    px_l0 = pixel_aux_loss(pixel_heads['l0'], quant_feats['l0'], px_target['l0'], pw_l0)
    px_l1 = pixel_aux_loss(pixel_heads['l1'], quant_feats['l1'], px_target['l1'], pw_l1)
    px_l2 = pixel_aux_loss(pixel_heads['l2'], quant_feats['l2'], px_target['l2'], pw_l1)
    aux_loss = l0_w * px_l0 + l1_w * px_l1 + l2_w * px_l2

    # 3. CODEBOOK ENTROPY DIVERSITY PENALTY
    div_l0 = codebook_diversity_loss(tokenizer.vq_l0, encoder_feats['l0'], div_temp)
    div_l1 = codebook_diversity_loss(tokenizer.vq_l1, encoder_feats['l1'], div_temp)
    div_l2 = codebook_diversity_loss(tokenizer.vq_l2, encoder_feats['l2'], div_temp)
    div_loss = (div_l0 + div_l1 + div_l2) / 3.0

    # 4. PATCH DIVERSITY LOSS
    patch_div = patch_diversity_loss_fn(encoder_feats)

    # 5. PER-FRAME TOKEN ENTROPY LOSS
    ent_loss = token_entropy_loss_fn(encoder_feats, tokenizer, temperature=div_temp)

    total = (
        commit_loss
        + px_w      * aux_loss
        + div_w     * div_loss
        + patch_div_w * patch_div
        + ent_w     * ent_loss
    )

    info = {
        'loss_commit':     commit_loss.item(),
        'loss_pixel':      aux_loss.item(),
        'loss_div':        div_loss.item(),
        'loss_patch_div':  patch_div.item(),
        'loss_entropy':    ent_loss.item(),
        'loss_total':      total.item(),
        'entropy_l0':      1.0 - div_l0.item(),
        'entropy_l1':      1.0 - div_l1.item(),
        'entropy_l2':      1.0 - div_l2.item(),
    }
    return total, info


# K-MEANS CODEBOOK INITIALISATION
@torch.no_grad()
def kmeans_init_codebooks(
    encoder:    SpatialAtariEncoder,
    tokenizer:  SpatialHRVQTokenizer,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    num_batches: int,
):
    """ INITIALISE CODEBOOKS FROM ENCODER OUTPUTS TO BREAK BACKGROUND FIXED POINT """
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
        all_embs = torch.cat(buffers[key], dim=0)
        num_codes = vq_layer.num_codebook_entries

        if len(all_embs) < num_codes:
            print(f"    {key}: only {len(all_embs)} embeddings < {num_codes} codes, skipping")
            continue

        # RANDOM SUBSET AS INITIAL CODEBOOK ENTRIES
        idx = torch.randperm(len(all_embs))[:num_codes]
        init_codes = all_embs[idx].to(device)

        vq_layer.codebook.weight.data.copy_(init_codes)
        if hasattr(vq_layer, 'ema_weight'):
            vq_layer.ema_weight.copy_(init_codes)
            # RESET CLUSTER SIZES TO UNIFORM
            vq_layer.ema_cluster_size.fill_(1.0)

        print(f"    {key}: initialised from {len(all_embs):,} embeddings  "
              f"(norm range [{init_codes.norm(dim=1).min():.3f}, "
              f"{init_codes.norm(dim=1).max():.3f}])")

    encoder.train()


# CODEBOOK UTILISATION MEASUREMENT
@torch.no_grad()
def measure_codebook_usage(
    encoder:   SpatialAtariEncoder,
    tokenizer: SpatialHRVQTokenizer,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    num_batches: int = 20,
) -> dict:
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
            'total':  totals[key],
            'pct':    100.0 * len(used[key]) / totals[key],
        }
        for key in ['l0', 'l1', 'l2']
    }
    encoder.train()
    tokenizer.train()
    return stats


# MAIN TRAINING LOOP
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

    d_model = config['model']['d_model']

    # BUILD MODELS
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
        use_gradient_vq=config['training'].get('use_gradient_vq', False),
    ).to(device)

    # TRAINING-ONLY PIXEL AUXILIARY HEADS (NOT SAVED IN CHECKPOINT)
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

    # OPTIMIZER - ENCODER + TOKENIZER + PIXEL HEADS TRAINED JOINTLY
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

    # DATA
    batch_size = config['training']['batch_size']
    print(f"\nLoading data...")
    train_loader, val_loader, dataset = build_dataloaders(config, batch_size)
    print(f"Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # K-MEANS CODEBOOK INITIALISATION
    ki_cfg = config.get('kmeans_init', {})
    if ki_cfg.get('enabled', True):
        kmeans_init_codebooks(
            encoder, tokenizer, train_loader, device,
            num_batches=ki_cfg.get('init_batches', 100),
        )

    # WANDB
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

    # RUNNING AVERAGES FOR STEP-LEVEL LOGGING
    running = {k: 0.0 for k in [
        'loss_commit', 'loss_pixel', 'loss_div',
        'loss_patch_div', 'loss_entropy', 'loss_total',
    ]}
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
                    token_dict, tokenizer, pixel_heads, cfg_loss,
                    config['training'],
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # ACCUMULATE RUNNING AVERAGES
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
                    f"patch_div={avgs['loss_patch_div']:.4f}  "
                    f"ent={avgs['loss_entropy']:.4f}  "
                    f"entropy=[{info['entropy_l0']:.2f},"
                    f"{info['entropy_l1']:.2f},{info['entropy_l2']:.2f}]"
                )
                if use_wandb:
                    import wandb
                    wandb.log({**avgs, 'step': step,
                               'entropy_l0': info['entropy_l0'],
                               'entropy_l1': info['entropy_l1'],
                               'entropy_l2': info['entropy_l2']})

        # VALIDATION AND CODEBOOK STATS
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
                        token_dict_v, tokenizer, pixel_heads, cfg_loss,
                        config['training'],
                    )
                    val_loss_sum += loss_v.item()
                    val_n += 1

            val_loss = val_loss_sum / max(val_n, 1)

            # CODEBOOK UTILISATION OVER 20 TRAIN BATCHES IN EVAL MODE
            usage = measure_codebook_usage(encoder, tokenizer, train_loader, device)
            print(f"\nEpoch {epoch} - val_loss={val_loss:.4f}")
            for key, stat in usage.items():
                print(f"  codebook {key}: {stat['unique']:3d}/{stat['total']:3d} ({stat['pct']:.1f}%) used")

            if use_wandb:
                import wandb
                log_dict = {'val_loss': val_loss, 'epoch': epoch}
                for key, stat in usage.items():
                    log_dict[f'codebook_usage_{key}_pct'] = stat['pct']
                    log_dict[f'codebook_usage_{key}_unique'] = stat['unique']
                wandb.log(log_dict)

            # SAVE BEST
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    'epoch': epoch,
                    'encoder_state_dict':   encoder.state_dict(),
                    'tokenizer_state_dict': tokenizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss':        best_val_loss,
                    'config':               config,
                    # NOTE: PIXEL HEADS NOT SAVED - TRAINING-ONLY
                }
                torch.save(ckpt, save_dir / 'spatial_encoder_best.pt')
                print(f"  Saved best  (val_loss={best_val_loss:.4f})")

        # PERIODIC CHECKPOINT
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/encoder_spatial.yaml')
    parser.add_argument('--wandb',  action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train(args.config, use_wandb=args.wandb, device_str=args.device)
