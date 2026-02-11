""" DATASET CLASS FOR WORLD MODEL TRAINING

1. RESHAPES
2. RESPECT EPISODE BOUNDARIES
3. MULTI-GAME MIXING
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import yaml
