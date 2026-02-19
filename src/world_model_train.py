""" TRAINING SCRIPT FOR WORLD MODEL MODULE """
import os
import sys
import math
import time
import json
import argparse
from pathlib import Path


import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from world_model import HierarchicalWorldModel, WorldModelConfig, hierarchical_loss
from world_model_dataset import create_dataloaders


def get_lr(
    step : int,
    warmup_steps : int,
    total_steps : int,
    max_lr : float,
    min_lr : float = 1e-6,
) -> float:
    """ LEARNING RATE SCHEDULE (LINEAR WARMUP)"""

    # 1. LINEAR INTERPOLATION WARMUP
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    
    # 2. COSINE DECAY 
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(decay_steps, 1) # progress %
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

@torch.no_grad()
def compute_metrics_summary(
    metrics_list : list[dict]
) -> dict:
    """ UTILITY FUNCTION TO AVERAGE METRICS OVER MULTIPLE BATCHES IN ONE EPOCH"""
    
    summary = {}
    
    for key in metrics_list[0]:
        summary[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    
    return summary

def train_one_epoch(
    model : HierarchicalWorldModel,
    train_loader, 
    optimizer : torch.optim.Optimizer,
    scaler : GradScaler,
    config : dict,
    global_step : int,
    total_steps : int,
    device : torch.device,
    epoch : int,
    use_wandb : bool = True,
) -> tuple:
    
    # 1. SET MODEL TO TRAIN MODE (ENABLES DROPOUT, BATCHNORM UPDATES, ETC.)
    model.train()
    
    # 2. CONFIG EXTRACTION
    
    accum_steps   = config['training']['accumulation_steps']
    grad_clip     = config['training']['grad_clip']
    max_lr        = config['training']['learning_rate']
    warmup_steps  = config['training']['warmup_steps']
    layer_weights = config['model']['layer_weights']
    use_amp       = config['training']['mixed_precision'] and device.type == 'cuda'
    
    running_loss = 0.0
    all_metrics  = []

    pass


@torch.no_grad()
def validate_one_epoch(
):
    pass

def save_checkpoint(
):
    pass

def load_checkpoint(
):
    pass

def train(
):
    pass


if __name__ == "__main__":
    pass

    