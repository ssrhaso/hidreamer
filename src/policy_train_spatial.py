"""
TRAIN POLICY — SPATIAL HI-DREAMER ENTRY POINT

Usage:
    python src/policy_train_spatial.py --config configs/policy_spatial.yaml --wandb
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_model_spatial import SpatialHierarchicalWorldModel, SpatialWorldModelConfig

from policy import (
    ActorNetwork, CriticNetwork, RewardNetwork, ContinueNetwork,
    HiddenStateFeatureExtractor,
    count_policy_params,
)
from imagination_spatial import SpatialImagineRollout
from replay_buffer_spatial import SpatialTokenReplayBuffer
from actor_critic_train_spatial import SpatialActorCriticTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Hi-Dreamer Policy Training")
    parser.add_argument("--config",   type=str, default="configs/policy_spatial.yaml")
    parser.add_argument("--game",     type=str, default=None)
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--wandb",    action="store_true", default=False)
    parser.add_argument("--seed",     type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_frozen_models(config: dict, device: torch.device):
    """Load and freeze the spatial world model."""
    paths      = config['frozen_models']
    wm_config  = SpatialWorldModelConfig.from_yaml(paths['spatial_world_model_config'])
    wm         = SpatialHierarchicalWorldModel(wm_config).to(device)

    ckpt = torch.load(paths['spatial_world_model'], map_location=device, weights_only=False)
    wm.load_state_dict(ckpt['model_state_dict'])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in wm.parameters())
    print(f"  Spatial World Model: {n_params:,} params (FROZEN)")
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', '?')}")
    return wm


def load_offline_buffer(config: dict, game: str, device: torch.device) -> SpatialTokenReplayBuffer:
    print(f"\nLoading offline buffer for {game}...")
    buffer = SpatialTokenReplayBuffer.from_numpy_data(
        tokens_dir = config['data']['tokens_dir'],
        game       = game,
        capacity   = 100_000,
        seq_len    = 16,
        device     = device,
    )
    print(f"  Buffer ready: {len(buffer)} transitions")
    return buffer


def build_trainable_networks(config: dict, num_actions: int, device: torch.device):
    """Instantiate all trainable policy components (hidden_state mode only)."""
    policy_config = config['policy']

    feature_extractor = HiddenStateFeatureExtractor(
        d_model        = 384,
        use_projection = True,
    ).to(device)

    feat_dim   = feature_extractor.feat_dim
    hidden_dim = policy_config.get('hidden_dim', 512)

    policy_net = ActorNetwork(
        feat_dim    = feat_dim,
        num_actions = num_actions,
        hidden_dim  = hidden_dim,
        unimix      = policy_config.get('unimix', 0.01),
    ).to(device)

    critic = CriticNetwork(feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)

    reward_net = RewardNetwork(feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)

    continue_net = ContinueNetwork(feat_dim=feat_dim, hidden_dim=hidden_dim // 2).to(device)

    counts = count_policy_params(
        critic=critic, actor=policy_net,
        reward_net=reward_net, continue_net=continue_net,
        feature_extractor=feature_extractor,
    )
    print(f"\n  Trainable networks (hidden_state mode, feat_dim={feat_dim}):")
    for name, count in counts.items():
        print(f"    {name}: {count:,}")

    return feature_extractor, policy_net, critic, reward_net, continue_net


def main():
    args   = parse_args()
    config = load_config(args.config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print()
    print("SPATIAL HI-DREAMER POLICY TRAINING")
    print()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    game        = args.game or config['policy']['game']
    num_actions = config['policy']['num_actions']

    print()
    print(f"GAME: {game}")
    print(f"ACTIONS: {num_actions}")
    print("OFFLINE MODE")
    print()

    print("Loading FROZEN spatial world model...")
    world_model = load_frozen_models(config, device)
    print()

    print("Building TRAINABLE networks...")
    feature_extractor, policy_net, critic, reward_net, continue_net = build_trainable_networks(
        config, num_actions, device
    )
    print()

    print("Building REPLAY BUFFER...")
    buffer = load_offline_buffer(config, game, device)
    print()

    print("Building IMAGINATION module...")
    imagination = SpatialImagineRollout(
        world_model      = world_model,
        feature_extractor = feature_extractor,
        actor_network    = policy_net,
        critic_network   = critic,
        reward_network   = reward_net,
        continue_network = continue_net,
        max_horizon      = config['policy'].get('max_horizon', 15),
        temperature      = config['policy']['temperature'],
        device           = device,
        use_amp          = True,
        max_context_t    = 10,
        compile_wm       = True,
    )
    print()

    print("Initializing TRAINER...")
    trainer = SpatialActorCriticTrainer(
        world_model      = world_model,
        feature_extractor = feature_extractor,
        policy           = policy_net,
        critic           = critic,
        reward_network   = reward_net,
        continue_network = continue_net,
        imagination      = imagination,
        replay_buffer    = buffer,
        config           = config,
        device           = device,
    )
    print()

    print("STARTING TRAINING...")
    trainer.train(
        total_steps    = config['policy']['total_steps'],
        use_wandb      = args.wandb,
        save_directory = config['logging']['save_dir'],
        log_interval   = config['policy']['log_every'],
    )
    print()
    print("SPATIAL POLICY TRAINING COMPLETE.")


if __name__ == "__main__":
    main()
