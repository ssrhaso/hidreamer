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
):
    pass

@torch.no_grad()
def compute_metrics_summary(
):
    pass

def train_one_epoch(
):
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

    