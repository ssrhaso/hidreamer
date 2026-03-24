"""
TRAINING SCRIPT FOR SPATIAL HIERARCHICAL WORLD MODEL

Usage:
    python src/train_world_model_spatial.py \\
        --config configs/worldmodel_spatial.yaml \\
        --wandb --device cuda

    python src/train_world_model_spatial.py \\
        --resume checkpoints/world_model_spatial/checkpoint_epoch_10.pt \\
        --config configs/worldmodel_spatial.yaml --wandb --device cuda
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torch import GradScaler
from tqdm import tqdm

import wandb

from world_model_spatial import SpatialHierarchicalWorldModel, SpatialWorldModelConfig
from world_model_dataset_spatial import create_spatial_dataloaders


# LR SCHEDULE

def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """LINEAR WARMUP + COSINE DECAY"""
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(decay_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


# METRIC UTILITIES

@torch.no_grad()
def compute_metrics_summary(metrics_list: list[dict]) -> dict:
    """AVERAGE METRICS OVER ALL BATCHES IN ONE EPOCH"""
    summary = {}
    for key in metrics_list[0]:
        summary[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    return summary


def _compute_accuracy(
    logits: torch.Tensor,   # (..., num_codes)
    targets: torch.Tensor,  # (...) matching leading dims
    num_codes: int,
) -> float:
    """ARGMAX ACCURACY: flattens over all batch/time/patch dimensions"""
    pred = logits.detach().float().reshape(-1, num_codes).argmax(dim=-1)
    true = targets.detach().reshape(-1)
    return (pred == true).float().mean().item()


# TRAIN / VALIDATE

def train_one_epoch(
    model: SpatialHierarchicalWorldModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict,
    model_config: SpatialWorldModelConfig,
    global_step: int,
    total_steps: int,
    device: torch.device,
    epoch: int,
    use_wandb: bool = True,
) -> tuple:

    model.train()

    accum_steps  = config['training']['accumulation_steps']
    grad_clip    = config['training']['grad_clip']
    max_lr       = config['training']['learning_rate']
    warmup_steps = config['training']['warmup_steps']
    use_amp      = config['training']['mixed_precision'] and device.type == 'cuda'

    running_loss = 0.0
    all_metrics  = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]", leave=True)
    optimizer.zero_grad()

    for batch_idx, (tokens_l0, tokens_l1, tokens_l2, actions) in enumerate(pbar):
        tokens_l0 = tokens_l0.to(device)
        tokens_l1 = tokens_l1.to(device)
        tokens_l2 = tokens_l2.to(device)
        actions   = actions.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(tokens_l0, tokens_l1, tokens_l2, actions)
            loss, metrics = model.compute_loss(
                out['logits_l0'], out['logits_l1'], out['logits_l2'],
                tokens_l0, tokens_l1, tokens_l2,
            )
            loss = loss / accum_steps

        # ACCURACY: computed on detached logits before backward frees the graph.
        # L0: skip t=0 because its prediction is unconditioned (zeros as prior).
        with torch.no_grad():
            acc_l0 = _compute_accuracy(
                out['logits_l0'][:, 1:], tokens_l0[:, 1:], model_config.num_codes_l0
            )
            acc_l1 = _compute_accuracy(out['logits_l1'], tokens_l1, model_config.num_codes_l1)
            acc_l2 = _compute_accuracy(out['logits_l2'], tokens_l2, model_config.num_codes_l2)

        metrics['acc_l0'] = acc_l0
        metrics['acc_l1'] = acc_l1
        metrics['acc_l2'] = acc_l2

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            lr = get_lr(
                step=global_step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                max_lr=max_lr,
            )
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            if use_wandb:
                wandb.log({
                    'train/loss_step': metrics['loss_total'],
                    'train/lr':        lr,
                    'train/global_step': global_step,
                }, step=global_step)

        running_loss += metrics['loss_total']
        all_metrics.append(metrics)

        pbar.set_postfix({
            'loss':   f"{metrics['loss_total']:.4f}",
            'acc_l0': f"{acc_l0:.3f}",
            'acc_l1': f"{acc_l1:.3f}",
            'acc_l2': f"{acc_l2:.3f}",
            'lr':     f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    avg_loss = running_loss / len(train_loader)
    return avg_loss, global_step, all_metrics


@torch.no_grad()
def validate_one_epoch(
    model: SpatialHierarchicalWorldModel,
    val_loader,
    config: dict,
    model_config: SpatialWorldModelConfig,
    device: torch.device,
    epoch: int,
) -> tuple:

    model.eval()
    use_amp = config['training']['mixed_precision'] and device.type == 'cuda'

    running_loss = 0.0
    all_metrics  = []
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]", leave=True)

    for tokens_l0, tokens_l1, tokens_l2, actions in pbar:
        tokens_l0 = tokens_l0.to(device)
        tokens_l1 = tokens_l1.to(device)
        tokens_l2 = tokens_l2.to(device)
        actions   = actions.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            out = model(tokens_l0, tokens_l1, tokens_l2, actions)
            _, metrics = model.compute_loss(
                out['logits_l0'], out['logits_l1'], out['logits_l2'],
                tokens_l0, tokens_l1, tokens_l2,
            )

        # Inside @torch.no_grad() — no separate context needed
        acc_l0 = _compute_accuracy(
            out['logits_l0'][:, 1:], tokens_l0[:, 1:], model_config.num_codes_l0
        )
        acc_l1 = _compute_accuracy(out['logits_l1'], tokens_l1, model_config.num_codes_l1)
        acc_l2 = _compute_accuracy(out['logits_l2'], tokens_l2, model_config.num_codes_l2)

        metrics['acc_l0'] = acc_l0
        metrics['acc_l1'] = acc_l1
        metrics['acc_l2'] = acc_l2

        running_loss += metrics['loss_total']
        all_metrics.append(metrics)

        pbar.set_postfix({
            'val_loss': f"{metrics['loss_total']:.4f}",
            'acc_l0':   f"{acc_l0:.3f}",
            'acc_l1':   f"{acc_l1:.3f}",
            'acc_l2':   f"{acc_l2:.3f}",
        })

    avg_loss = running_loss / len(val_loader)
    return avg_loss, all_metrics


# CHECKPOINTING

def save_checkpoint(
    model: SpatialHierarchicalWorldModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    save_path: str,
):
    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'epoch':                epoch,
        'global_step':          global_step,
        'best_val_loss':        best_val_loss,
    }
    torch.save(checkpoint, save_path)
    print(f"    CHECKPOINT SAVED to : {save_path}")


def load_checkpoint(
    path: str,
    model: SpatialHierarchicalWorldModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    print(f"    CHECKPOINT LOADED from : {path}")
    return checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint['best_val_loss']


# MAIN TRAINING FUNCTION

def train(
    config_path: str = "configs/worldmodel_spatial.yaml",
    resume_from: str = None,
    use_wandb: bool = False,
    device_str: str = None,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(
        device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    """ 1. DATALOADERS """
    train_loader, val_loader, data_info = create_spatial_dataloaders(
        config_path=config_path,
        seed=config['training']['seed'],
    )

    """ 2. MODEL """
    model_config = SpatialWorldModelConfig.from_yaml(path=config_path)
    model = SpatialHierarchicalWorldModel(config=model_config).to(device)
    print(f"\n{model_config}")
    print(f"Parameters: {model.count_parameters():,}")

    """ 3. OPTIMIZER """
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config['training']['learning_rate'],
        betas=tuple(config['training']['betas']),
        weight_decay=config['training']['weight_decay'],
    )

    """ 4. AMP SCALER """
    use_amp = config['training']['mixed_precision'] and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    """ 5. LR SCHEDULE: compute total optimizer steps """
    accum_steps     = config['training']['accumulation_steps']
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps     = steps_per_epoch * config['training']['num_epochs']

    """ 6. CHECKPOINT RESUMPTION """
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float('inf')

    if resume_from and os.path.isfile(resume_from):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    """ 7. SAVE DIR + WANDB """
    save_dir = config['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            resume="allow" if resume_from else False,
        )
        wandb.watch(models=model, log='gradients', log_freq=100)

    # Safe defaults for variables referenced after the epoch loop
    METRIC_KEYS = ('loss_l0', 'loss_l1', 'loss_l2', 'loss_total', 'acc_l0', 'acc_l1', 'acc_l2')
    epoch          = start_epoch - 1
    avg_train_loss = float('nan')
    avg_val_loss   = float('nan')
    train_summary  = {k: float('nan') for k in METRIC_KEYS}
    val_summary    = {k: float('nan') for k in METRIC_KEYS}

    print(f"\nDATA  : {data_info}")
    print(f"DEVICE: {device}  AMP: {use_amp}")
    print(f"STEPS : {total_steps} ({steps_per_epoch}/epoch × {config['training']['num_epochs']} epochs)")
    print()

    """ MAIN TRAINING LOOP """
    for epoch in range(start_epoch, config['training']['num_epochs']):

        avg_train_loss, global_step, train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            model_config=model_config,
            global_step=global_step,
            total_steps=total_steps,
            device=device,
            epoch=epoch,
            use_wandb=use_wandb,
        )

        avg_val_loss, val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            config=config,
            model_config=model_config,
            device=device,
            epoch=epoch,
        )

        train_summary = compute_metrics_summary(train_metrics)
        val_summary   = compute_metrics_summary(val_metrics)

        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"  TRAIN LOSS : {avg_train_loss:.4f} | VAL LOSS : {avg_val_loss:.4f}")
        print(f"  TRAIN  L0/L1/L2 ACC : {train_summary['acc_l0']:.3f} / {train_summary['acc_l1']:.3f} / {train_summary['acc_l2']:.3f}")
        print(f"  VAL    L0/L1/L2 ACC : {val_summary['acc_l0']:.3f} / {val_summary['acc_l1']:.3f} / {val_summary['acc_l2']:.3f}")

        if use_wandb:
            wandb.log({
                'epoch':              epoch + 1,
                'train/loss_epoch':   avg_train_loss,
                'val/loss_epoch':     avg_val_loss,
                'train/loss_l0':      train_summary['loss_l0'],
                'train/loss_l1':      train_summary['loss_l1'],
                'train/loss_l2':      train_summary['loss_l2'],
                'val/loss_l0':        val_summary['loss_l0'],
                'val/loss_l1':        val_summary['loss_l1'],
                'val/loss_l2':        val_summary['loss_l2'],
                'train/acc_l0':       train_summary['acc_l0'],
                'train/acc_l1':       train_summary['acc_l1'],
                'train/acc_l2':       train_summary['acc_l2'],
                'val/acc_l0':         val_summary['acc_l0'],
                'val/acc_l1':         val_summary['acc_l1'],
                'val/acc_l2':         val_summary['acc_l2'],
            }, step=global_step)

        # BEST CHECKPOINT: val loss_total is the selection metric.
        # Using the raw sum is clean — no auxiliary terms that can go negative
        # (the encoder had a bug where patch_diversity_loss went negative and
        # corrupted best-checkpoint selection; we avoid that here).
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model=model, optimizer=optimizer, scaler=scaler,
                epoch=epoch, global_step=global_step,
                best_val_loss=best_val_loss,
                save_path=os.path.join(save_dir, "best_model.pt"),
            )

            if use_wandb:
                artifact = wandb.Artifact(
                    name='hi-dreamer-spatial-wm-best',
                    type='model',
                    metadata={
                        'epoch':      epoch + 1,
                        'val_loss':   avg_val_loss,
                        'val_acc_l0': val_summary['acc_l0'],
                        'val_acc_l1': val_summary['acc_l1'],
                        'val_acc_l2': val_summary['acc_l2'],
                    },
                )
                artifact.add_file(os.path.join(save_dir, 'best_model.pt'))
                wandb.log_artifact(artifact)

        if (epoch + 1) % config['logging']['save_every'] == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, scaler=scaler,
                epoch=epoch, global_step=global_step,
                best_val_loss=best_val_loss,
                save_path=os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )

    """ POST-TRAINING CLEANUP """
    save_checkpoint(
        model=model, optimizer=optimizer, scaler=scaler,
        epoch=epoch, global_step=global_step,
        best_val_loss=best_val_loss,
        save_path=os.path.join(save_dir, "final_model.pt"),
    )

    stats = {
        'final_train_loss':   float(avg_train_loss),
        'final_val_loss':     float(avg_val_loss),
        'final_train_acc_l0': float(train_summary['acc_l0']),
        'final_val_acc_l0':   float(val_summary['acc_l0']),
        'final_train_acc_l1': float(train_summary['acc_l1']),
        'final_val_acc_l1':   float(val_summary['acc_l1']),
        'final_train_acc_l2': float(train_summary['acc_l2']),
        'final_val_acc_l2':   float(val_summary['acc_l2']),
    }
    with open(os.path.join(save_dir, "final_metrics.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    if use_wandb:
        wandb.finish()

    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spatial hierarchical world model")
    parser.add_argument('--config', type=str, default='configs/worldmodel_spatial.yaml',
                        help="Path to worldmodel_spatial.yaml")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--wandb', action='store_true',
                        help="Enable wandb logging")
    parser.add_argument('--device', type=str, default=None,
                        help="Device override: 'cuda' or 'cpu' (auto-detected if omitted)")
    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_from=args.resume,
        use_wandb=args.wandb,
        device_str=args.device,
    )
