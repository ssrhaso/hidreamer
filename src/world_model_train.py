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
from torch import autocast
from torch import GradScaler
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
            'acc_l0' : f"{metrics['accuracy_l0']:.3f}",
            'acc_l1' : f"{metrics['accuracy_l1']:.3f}",
            'acc_l2' : f"{metrics['accuracy_l2']:.3f}",
            'lr' : f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        
        
    """ 5. EPOCH SUMMARY """
    avg_loss = running_loss / len(train_loader)
    return avg_loss, global_step, all_metrics


@torch.no_grad()
def validate_one_epoch(
    model : HierarchicalWorldModel,
    val_loader, 
    config : dict,
    device : torch.device,
    epoch : int,
) -> tuple:
    
    """ 1. SET MODEL TO EVAL MODE (DISABLES DROPOUT, BATCHNORM FIXED, ETC.)"""
    model.eval()
    
    
    """ 2. CONFIG EXTRACTION"""
    
    layer_weights = config['model']['layer_weights']
    use_amp       = config['training']['mixed_precision'] and device.type == 'cuda'
    
    
    """ 3. PRE LOOP INITIALISATION"""
    running_loss = 0.0
    all_metrics  = []
    pbar = tqdm(val_loader, desc = f"Epoch {epoch+1} [VAL]", leave = True)    
    
    """ 4. MAIN VALIDATION LOOP""" 
    
    for tokens, actions in pbar:
        
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
        
        # NO BACKWARD PASS OR OPTIMIZER STEP DURING VALIDATION, JUST METRICS COMPUTATION
        
        """ LOGGING AND PBAR UPDATE """
        running_loss += metrics['loss_total'] 
        all_metrics.append(metrics)
        
        pbar.set_postfix({
            'val_loss' : f"{metrics['loss_total']:.4f}",
            'acc_l0' : f"{metrics['accuracy_l0']:.3f}",
            'acc_l1' : f"{metrics['accuracy_l1']:.3f}",
            'acc_l2' : f"{metrics['accuracy_l2']:.3f}",
        })
        
        
    """ 5. EPOCH SUMMARY """
    
    avg_loss = running_loss / len(val_loader)
    return avg_loss, all_metrics

def save_checkpoint(
    model : HierarchicalWorldModel,
    optimizer : torch.optim.Optimizer,
    scaler : GradScaler,
    epoch : int,
    global_step : int,
    best_val_loss : float,
    save_path : str,
):
    checkpoint = {
        
        'model_state_dict' : model.state_dict(),
        # ORDERED DICT of all model parameter tensors and their values (weights, biases, etc.)
        
        'optimizer_state_dict' : optimizer.state_dict(),
        # ADAMW Internal states (adaptive lr rate)
        
        'scaler_state_dict' : scaler.state_dict(),
        # GRADSCALER Internal states (current scale factor, growth interval, etc.)
        
        'epoch' : epoch,
        'global_step' : global_step,
        'best_val_loss' : best_val_loss,
        # epoch, lr scheduler step counter, best validation loss for early stopping or model selection
    }
    
    torch.save(checkpoint, save_path)
    print(f"    CHECKPOINT SAVED to : {save_path}")


def load_checkpoint(
    path : str,
    model : HierarchicalWorldModel,
    optimizer : torch.optim.Optimizer,
    scaler : GradScaler,
    device : torch.device,
) -> tuple:
    
    checkpoint = torch.load(path, map_location = device, weights_only = False)
    
    # DESERLIASE WEIGHTS / BIASES 
    model.load_state_dict(state_dict = checkpoint['model_state_dict'])
    
    # DESERIALISE OPTIMIZER AND SCALER STATES (LEARNING RATES, MOMENTUM, SCALE FACTOR, ETC.)
    optimizer.load_state_dict(state_dict = checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(state_dict = checkpoint['scaler_state_dict'])
    
    print(f"    CHECKPOINT LOADED from : {path}")
    
    return checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint['best_val_loss']

def train(
    config_path : str = "configs/worldmodel.yaml",
    resume_from : str = None,
    use_wandb : bool = True,
):
    """ LOAD CONFIG """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """ BUILD DATALOADERS """
    train_loader, val_loader, data_info = create_dataloaders(config_path = config_path, seed = config['training']['seed'])

    
    pass


if __name__ == "__main__":
    pass
