""" POLICY NETWORKS FOR ACTOR CRITIC

ALL TRAINABLE COMPONENTS FOR IMAGINATION BASED RL

World Model is FROZEN during PPO training - only these networks are updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Optional
import math

def symlog():
    pass

def symexp():
    pass   

class HierarchicalFeatureExtractor(nn.Module):
    pass

class PolicyNetwork(nn.Module):
    pass

class ValueNetwork(nn.Module):
    pass

class RewardPredictor(nn.Module):
    pass

class ContinueNetwork(nn.Module):
    pass

class SlowValueTarget(nn.Module):
    pass

def compute_lambda_returns():
    pass

class ReturnNormalizer:
    pass

def count_policy_params():
    pass