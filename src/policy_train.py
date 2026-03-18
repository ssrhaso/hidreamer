"""
TRAIN POLICY — Entry point for Hi-Dreamer actor-critic training.

Usage:
    python train_policy.py --config configs/policy.yaml --game Pong-v5
    python train_policy.py --config configs/policy.yaml --offline  # no env needed

Requires pre-trained frozen models:
    - World model checkpoint
    - HRVQ tokenizer checkpoint  
    - CNN encoder checkpoint
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_model import HierarchicalWorldModel, WorldModelConfig
from vq import HRVQTokenizer
from encoder_v1 import AtariCNNEncoder

from policy import (
    ActorNetwork, CriticNetwork, RewardNetwork, ContinueNetwork,
    HierarchicalFeatureExtractor, count_policy_params,
)
from imagination import ImagineRollout
from replay_buffer import TokenReplayBuffer
from actor_critic_train import ActorCriticTrainer


def parse_args():
    """ Parse command-line arguments for policy training."""
    
    parser = argparse.ArgumentParser(description = "Hi-Dreamer Policy Training")
    parser.add_argument("--config", type = str, default = "configs/policy.yaml")
    parser.add_argument("--game", type = str, default = None, help="Override game from config")
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--wandb", action = "store_true", default = False)
    parser.add_argument("--offline", action = "store_true", default = False, help = "Force offline mode")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Resume from checkpoint")
    parser.add_argument("--seed", type = int, default = 42)
    return parser.parse_args()

def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_offline_buffer(config: dict, game: str, device: torch.device) -> TokenReplayBuffer:
    """Load pre-collected data into replay buffer."""
    
    print(f"\nLoading offline buffer for {game}...")
    buffer = TokenReplayBuffer.from_numpy_data(
        tokens_dir=config['data']['tokens_dir'],
        replay_dir=config['data']['replay_dir'],
        game=game,
        capacity=100_000,
        seq_len=64,
        device=device,
    )
    print(f"  Buffer ready: {len(buffer)} transitions")
    return buffer

def load_frozen_models(
    config : dict,
    device : torch.device
):
    """ LOAD and FREEZE all pre-trained models
    
    - World Model 
    - HRVQ Tokenizer
    - CNN Encoder
    """
    
    paths = config['frozen_models']
    
    # 1. World Model
    print("\nLoading frozen WORLD MODEL...")
    world_model_config = WorldModelConfig.from_yaml("configs/worldmodel.yaml")
    world_model = HierarchicalWorldModel(world_model_config).to(device)
    
    world_model_checkpoint = torch.load(paths['world_model'], map_location=device, weights_only=False)
    world_model.load_state_dict(world_model_checkpoint['model_state_dict'])
    world_model.eval()
    
    # FREEZE
    for p in world_model.parameters():
        p.requires_grad = False
    
    world_model_params = sum(p.numel() for p in world_model.parameters())
    print(f"  World Model loaded with {world_model_params:,} parameters (FROZEN)")
    print(f"  Loaded from epoch {world_model_checkpoint.get('epoch', '?')}, val_loss={world_model_checkpoint.get('best_val_loss', '?')}")
    
    
    # 2. HRVQ Tokenizer
    print("\nLoading frozen HRVQ TOKENIZER...")
    hrvq = HRVQTokenizer(
        input_dim=384, num_codes_per_layer=256, num_layers=3,
        commitment_costs=[0.05, 0.25, 0.60],
    ).to(device)
    
    hrvq.load_state_dict(torch.load(paths['hrvq_tokenizer'], map_location=device, weights_only=False))
    hrvq.eval()
    
    # FREEZE
    for p in hrvq.parameters():
        p.requires_grad = False
    print(f"  HRVQ Tokenizer loaded and FROZEN")
    
    # 3. CNN Encoder
    print("\nLoading frozen CNN ENCODER...")
    encoder = AtariCNNEncoder(input_channels=4, embedding_dim=384).to(device)
    
    
    encoder_checkpoint = torch.load(paths['encoder'], map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"  CNN encoder: {encoder_params:,} params (FROZEN)")
    
    # NOTE: count_parameters() filters by requires_grad, so returns 0 after freezing, hence manual counting.
    
    return world_model, hrvq, encoder

def make_env(
    game : str,
    seed : int = 42
):
    """ Create Gymnasium ALE environment with standard Atari preprocessing. """
    
    try:
        import ale_py
        import gymnasium as gym
        
        env_name = f"ALE/{game}"
        env = gym.make(env_name)
        env = gym.wrappers.AtariPreprocessing(
            env, frame_skip=1, screen_size=84,
            grayscale_obs=True, scale_obs=False,
        )
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.reset(seed=seed)
        
        num_actions = env.action_space.n
        print(f"  Environment: {env_name}, {num_actions} actions")
        return env, num_actions
    except Exception as e:
        print(f"  WARNING: Could not create environment ({e})")
        print(f"  Running in offline mode only")
        return None, None


def build_trainable_networks(
    config : dict,
    hrvq,
    num_actions : int,
    device : torch.device
):
    """ INSTANTIATE all trainable policy components
    
    - Feature extractor (uses frozen HRVQ codebooks)
    - Actor Network
    - Critic Network
    - Reward Predictor
    - Continue Predictor
    """
    
    policy_config = config['policy']
    mode = policy_config.get('feature_mode', 'concat')
    
    # 1. FEATURE EXTRACTOR (Frozen HRVQ codebooks)
    feature_extractor = HierarchicalFeatureExtractor(
        hrvq_tokenizer = hrvq,
        mode = mode,
        d_model = 384,
    ).to(device)
    
    # EXTRACT
    feat_dim = feature_extractor.feat_dim
    hidden_dim = policy_config.get('hidden_dim', 512)
    
    # 2. ACTOR NETWORK
    policy = ActorNetwork(
        feat_dim = feat_dim,
        num_actions = num_actions,
        hidden_dim = hidden_dim,
        unimix = policy_config.get('unimix', 0.01),
    ).to(device)
    
    # 3. CRITIC NETWORK
    critic = CriticNetwork(
        feat_dim = feat_dim,
        hidden_dim = hidden_dim,
    ).to(device)
    
    # 4. REWARD NETWORK
    reward_net = RewardNetwork(
        feat_dim = feat_dim,
        hidden_dim = hidden_dim,
    ).to(device)
    
    # 5. CONTINUE NETWORK
    continue_net = ContinueNetwork(
        feat_dim = feat_dim,
        hidden_dim = 256,
    ).to(device)
    
    # PARAM COUNTS
    counts = count_policy_params(
        critic = critic,
        policy = policy,    
        reward_net = reward_net,
        continue_net = continue_net,
        feature_extractor = feature_extractor,
    )
    print(f"\n  Trainable networks ({mode} mode, feat_dim={feat_dim}):")
    for name, count in counts.items():
        print(f"    {name}: {count:,}")
    
    return feature_extractor, policy, critic, reward_net, continue_net


def main():
    
    # SETUP 
    args = parse_args()
    config = load_config(args.config)
    
    # SEED
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # DEVICE AND INIT OUTPUT
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print()
    print(f"Hi-DREAMER POLICY TRAINING")
    print()
    print(f"Device: {device}")
    print()
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # OVERRIDE GAME IF SPECIFIED
    game = args.game or config['policy']['game']
    num_actions = config['policy']['num_actions']
    offline_mode = args.offline or config['policy'].get('offline_mode', True)

    print()
    print(f"GAME: {game}")
    print(f"NUMBER OF ACTIONS: {num_actions}")
    print(f"OFFLINE TRAINING" if offline_mode else "ONLINE TRAINING") 
    print()
    
    # LOAD FROZEN MODELS
    
    print(f"Loading FROZEN models... ")
    print()
    world_model, hrvq_tokenizer, cnn_encoder = load_frozen_models(
        config = config, device = device
    )
    print(f"FROZEN models loaded.")
    print()
    
    # BUILD NETWORKS (TRAINABLE)
    print(f"Building TRAINABLE networks... ")
    feature_extractor, policy, critic, reward_net, continue_net = build_trainable_networks(
        config = config,
        hrvq = hrvq_tokenizer,
        num_actions = num_actions,
        device = device,
    )
    print()
    print(f"TRAINABLE networks built.")
    print()
    
    # ENVIRONMENT (if not offline-only)
    env = None
    if not offline_mode:
        print(f"Building ENVIRONMENT (ALE)... ")
        env, env_actions = make_env(
            game = game,
            seed = args.seed,
        )
        print()
        
        # CHECK ACTION SPACE
        if env_actions and env_actions != num_actions:
            print(f"  WARNING: config num_actions={num_actions} but env has {env_actions}")
            num_actions = env_actions
            print()
        
        print(f"ENVIRONMENT ready.")
        print()
        
    # REPLAY BUFFER
    print(f"Building REPLAY BUFFER... ")
    if offline_mode:
        buffer = load_offline_buffer(
            config = config,
            game = game,
            device = device,
        )
    
    else:
        buffer = TokenReplayBuffer(
            capacity = 100_000,
            seq_len = 64,
            device = device,
        )
    print()
    print(f"REPLAY BUFFER ready.")
    print()
    
    """ IMAGINATION ROLLOUT MODULE"""
    print(f"Building IMAGINATION module... ")
    
    imagination = ImagineRollout(
        world_model = world_model,
        feature_extractor = feature_extractor,
        actor_network = policy,
        critic_network = critic,
        reward_network = reward_net,
        continue_network = continue_net,
        max_horizon = config['policy'].get('max_horizon', 30),
        temperature = config['policy']['temperature'],
        device = device,
    )
    print()
    print(f"IMAGINATION module ready.")
    print()
    
    # TRAINER (ACTOR-CRITIC)
    print(f"Initializing TRAINER... ")
    trainer = ActorCriticTrainer(
        world_model = world_model,
        hrvq_tokenizer = hrvq_tokenizer,
        encoder = cnn_encoder,
        feature_extractor = feature_extractor,
        policy = policy,
        critic = critic,
        reward_network = reward_net,
        continue_network = continue_net,
        imagination = imagination,
        replay_buffer = buffer,
        config = config,
        env = env,
        device = device,
    )
    print()
    print(f"TRAINER initialized.")
    print()
    
    # TRAIN
    print(f"STARTING TRAINING... ")
    trainer.train(
        total_steps = config['policy']['total_steps'],
        use_wandb = args.wandb,
        save_directory = config['logging']['save_dir'],
        eval_interval = config['policy']['eval_every'],
        log_interval = config['policy']['log_every'],
        prefill_steps = config['policy'].get('prefill_steps', 0),
        offline_mode = offline_mode,
    )
    print()
    print(f"POLICY TRAINING COMPLETE.")
    print()

if __name__ == "__main__":
    main()
    
    