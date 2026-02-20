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
    
    """ 1. SET MODEL TO TRAIN MODE (ENABLES DROPOUT, BATCHNORM UPDATES, ETC.)"""
    model.train()
    
    """ 2. CONFIG EXTRACTION"""
    
    accum_steps   = config['training']['accumulation_steps']
    grad_clip     = config['training']['grad_clip']
    max_lr        = config['training']['learning_rate']
    warmup_steps  = config['training']['warmup_steps']
    layer_weights = config['model']['layer_weights']
    use_amp       = config['training']['mixed_precision'] and device.type == 'cuda'
    
    """ 3. PRE LOOP INITIALISATION"""
    running_loss = 0.0
    all_metrics  = []
    pbar = tqdm(train_loader, desc = f"Epoch {epoch+1} [TRAIN]", leave = True)
    optimizer.zero_grad() # RESET GRADIENTS 1 TIME BEFORE STARTING EPOCH
    
    """ 4. MAIN TRAINING LOOP""" 
    
    for batch_idx, (tokens, actions) in enumerate(pbar):
        # tokens = (B, T, 3) - HRVQ TOKENS
        # actions = (B, T) - DISCRETE ACTIONS
        
        tokens = tokens.to(device)
        actions = actions.to(device)
        
        """ FORWARD PASS """
        with autocast("cuda", enabled = use_amp):
            
            # MODEL OUTPUTS
            logits_l0, logits_l1, logits_l2 = model(tokens, actions) 
            
            # CROSS ENTROPY HIERARCHICAL LOSS
            loss, metrics = hierarchical_loss(
                logits_l0 = logits_l0, logits_l1 = logits_l1, logits_l2 = logits_l2,
                tokens = tokens,
                layer_weights = layer_weights,
            )
            
            # scale down so gradients from multiple mini-batches sum to correct magnitude
            loss = loss / accum_steps
            
        """ BACKWARD PASS """
        
        scaler.scale(loss).backward()
        # multiply loss by scale factor (prevent fp16 underflow) and compute gradients
        
        # GRADIENT ACCUMULATION STEP 
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer) # unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip) # gradient clipping
            
            # self maintenance optimizer with gradscaler
            scaler.step(optimizer) # update parameters
            scaler.update() # update scale factor for next iteration
            optimizer.zero_grad() # reset gradients for next accumulation
            global_step += 1
    
            # UPDATE LEARNING RATE FOR THIS STEP
            lr = get_lr(
                step = global_step,
                warmup_steps = warmup_steps,
                total_steps = total_steps,
                max_lr = max_lr,
            )
            
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = lr # UPDATE LEARNING RATE MANUALLY FOR EACH PARAMETER GROUP
            
            # WANDB LOGGING
            if use_wandb:
                wandb.log({
                    
                    'train/loss_step' :     metrics['loss_total'],
                    'train/lr' :            lr,    
                    'train/global_step' :   global_step,
    
                }, step = global_step)
        
        
        
        """ LOGGING AND PBAR UPDATE """
        
        running_loss += metrics['loss_total'] 
        all_metrics.append(metrics)
        
        pbar.set_postfix({
            'loss' : f"{metrics['loss_total']:.4f}",
            'acc_l0' : f"{metrics['acc_l0']:.3f}",
            'acc_l1' : f"{metrics['acc_l1']:.3f}",
            'acc_l2' : f"{metrics['acc_l2']:.3f}",
            'lr' : f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        
        
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

    