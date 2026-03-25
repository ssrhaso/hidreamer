"""
STANDALONE EVALUATION SCRIPT FOR SPATIAL HI-DREAMER POLICY

Usage:
    python src/eval_spatial.py \
        --checkpoint checkpoints/policy_spatial/policy_step_50000.pt \
        --config configs/policy_spatial.yaml \
        --episodes 10 \
        --device cuda
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_model_spatial import SpatialHierarchicalWorldModel, SpatialWorldModelConfig
from encoder_v2 import SpatialAtariEncoder
from vq_spatial import SpatialHRVQTokenizer
from policy import ActorNetwork, HiddenStateFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Spatial Hi-Dreamer Policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="configs/policy_spatial.yaml")
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--verbose",    action="store_true", default=False)
    return parser.parse_args()


def load_all(checkpoint_path: str, config: dict, device: torch.device):
    """Load frozen world model, encoder, tokenizer, and trained policy."""
    paths = config['frozen_models']

    # SPATIAL WORLD MODEL
    wm_config = SpatialWorldModelConfig.from_yaml(paths['spatial_world_model_config'])
    wm = SpatialHierarchicalWorldModel(wm_config).to(device)
    wm_ckpt = torch.load(paths['spatial_world_model'], map_location=device, weights_only=False)
    wm.load_state_dict(wm_ckpt['model_state_dict'])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    print(f"  World model loaded from {paths['spatial_world_model']}")

    # SPATIAL ENCODER + TOKENIZER
    with open(paths['spatial_encoder_config'], 'r') as f:
        enc_config = yaml.safe_load(f)

    encoder = SpatialAtariEncoder(
        input_channels=enc_config['model']['input_channels'],
        d_model=enc_config['model']['d_model'],
    ).to(device)

    tokenizer = SpatialHRVQTokenizer(
        d_model=enc_config['model']['d_model'],
        num_codes_l0=enc_config['tokenizer']['num_codes_l0'],
        num_codes_l1=enc_config['tokenizer']['num_codes_l1'],
        num_codes_l2=enc_config['tokenizer']['num_codes_l2'],
        commitment_costs=enc_config['tokenizer']['commitment_costs'],
        decay=enc_config['tokenizer']['decay'],
        epsilon=enc_config['tokenizer']['epsilon'],
        use_gradient_vq=enc_config['training'].get('use_gradient_vq', False),
    ).to(device)

    enc_ckpt = torch.load(paths['spatial_encoder'], map_location=device, weights_only=False)
    encoder.load_state_dict(enc_ckpt['encoder_state_dict'])
    tokenizer.load_state_dict(enc_ckpt['tokenizer_state_dict'])
    encoder.eval()
    tokenizer.eval()
    for p in list(encoder.parameters()) + list(tokenizer.parameters()):
        p.requires_grad = False
    print(f"  Encoder loaded from {paths['spatial_encoder']}")

    # FEATURE EXTRACTOR AND ACTOR
    policy_config = config['policy']
    feat_ext = HiddenStateFeatureExtractor(d_model=384, use_projection=True).to(device)
    actor = ActorNetwork(
        feat_dim=feat_ext.feat_dim,
        num_actions=policy_config['num_actions'],
        hidden_dim=policy_config.get('hidden_dim', 512),
        unimix=policy_config.get('unimix', 0.03),
    ).to(device)

    # LOAD TRAINED WEIGHTS
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feat_ext.load_state_dict(ckpt['feature_extractor_state_dict'])
    actor.load_state_dict(ckpt['policy_state_dict'])
    feat_ext.eval()
    actor.eval()
    print(f"  Policy loaded from {checkpoint_path} (step {ckpt.get('step', '?')})")

    return wm, encoder, tokenizer, feat_ext, actor


@torch.no_grad()
def encode_obs(obs, encoder, tokenizer, device, use_amp):
    """Raw Atari observation to spatial token dict."""
    frame = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    with autocast(device_type=device.type, enabled=use_amp):
        spatial_feats = encoder(frame)
        token_dict    = tokenizer.encode(spatial_feats)
    return {
        'l0': token_dict['l0'].squeeze(0),   # (4,)
        'l1': token_dict['l1'].squeeze(0),   # (16,)
        'l2': token_dict['l2'].squeeze(0),   # (16,)
    }


@torch.no_grad()
def get_features(wm, feat_ext, token_history_l0, token_history_l1,
                 token_history_l2, action_history, device, use_amp, max_ctx=5):
    """Run frozen spatial WM on context to get features for the last timestep."""
    ctx_l0 = torch.stack(token_history_l0[-max_ctx:]).unsqueeze(0).to(device)
    ctx_l1 = torch.stack(token_history_l1[-max_ctx:]).unsqueeze(0).to(device)
    ctx_l2 = torch.stack(token_history_l2[-max_ctx:]).unsqueeze(0).to(device)
    ctx_a  = torch.tensor(action_history[-max_ctx:]).unsqueeze(0).to(device)

    # MATCH THE IMAGINATION CONTEXT 
    # During training, features are from 3 pass cascade, but at eval time we only have one pass, so we need to match the context length to preserve feature semantics

    ctx_l2[:, -1, :] = 0
    ctx_a[:, -1]     = 0

    with autocast(device_type=device.type, enabled=use_amp):
        out = wm(ctx_l0, ctx_l1, ctx_l2, ctx_a)
    x = out['hidden']
    D = x.size(-1)
    T = ctx_l0.size(1)

    x_ts = x.reshape(1, T, 37, D)
    last = x_ts[:, -1, :, :]

    mean_l0 = last[:, 0:4,   :].mean(dim=1)
    mean_l1 = last[:, 4:20,  :].mean(dim=1)
    mean_l2 = last[:, 20:36, :].mean(dim=1)
    hidden_3 = torch.stack([mean_l0, mean_l1, mean_l2], dim=1)
    return feat_ext(hidden_3)


def run_episode(wm, encoder, tokenizer, feat_ext, actor, env,
                device, use_amp, max_ctx=5, verbose=False):
    """Run one episode with greedy action selection."""
    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    done = False

    token_history_l0 = []
    token_history_l1 = []
    token_history_l2 = []
    action_history   = []

    while not done:
        tokens = encode_obs(obs, encoder, tokenizer, device, use_amp)
        token_history_l0.append(tokens['l0'].cpu())
        token_history_l1.append(tokens['l1'].cpu())
        token_history_l2.append(tokens['l2'].cpu())

        while len(action_history) < len(token_history_l0):
            action_history.append(0)

        features = get_features(
            wm, feat_ext,
            token_history_l0, token_history_l1, token_history_l2,
            action_history, device, use_amp, max_ctx=max_ctx,
        )

        dist = actor(features)
        action = dist.probs.argmax(dim=-1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        action_history[-1] = action   # replace placeholder with real action
        done = terminated or truncated
        episode_return += reward
        episode_length += 1
        obs = next_obs

        # TRIM HISTORIES
        if len(token_history_l0) > max_ctx:
            token_history_l0 = token_history_l0[-max_ctx:]
            token_history_l1 = token_history_l1[-max_ctx:]
            token_history_l2 = token_history_l2[-max_ctx:]
            action_history   = action_history[-max_ctx:]

    if verbose:
        print(f"  Episode: return={episode_return:.0f}, length={episode_length}")

    return episode_return, episode_length


def main():
    args = parse_args()

    import ale_py
    import gymnasium as gym

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'

    print()
    print("SPATIAL HI-DREAMER POLICY EVALUATION")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.episodes}")
    print()

    # VALIDATE CHECKPOINT EXISTS
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("Loading models...")
    wm, encoder, tokenizer, feat_ext, actor = load_all(args.checkpoint, config, device)
    print()

    # CREATE ENV
    game = config['policy']['game']
    ale_game = f"ALE/{game}" if not game.startswith("ALE/") else game
    env = gym.make(ale_game)

    with open(config['frozen_models']['spatial_encoder_config'], 'r') as f:
        enc_config = yaml.safe_load(f)
    input_channels = enc_config['model']['input_channels']

    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=84,
                                           grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=input_channels)
    print(f"  Env: {ale_game}")

    max_ctx = config['policy']['seed_context_len']

    # RUN EVAL EPISODES
    print(f"\nRunning {args.episodes} episodes...")
    returns = []
    lengths = []

    for ep in range(args.episodes):
        ret, length = run_episode(
            wm, encoder, tokenizer, feat_ext, actor, env,
            device, use_amp, max_ctx=max_ctx, verbose=args.verbose,
        )
        returns.append(ret)
        lengths.append(length)
        print(f"  Episode {ep+1}/{args.episodes}: return={ret:.0f}, length={length}")

    env.close()

    # SUMMARY
    print()
    print(f"RESULTS ({args.episodes} episodes)")
    print(f"  Return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"  Length: {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"  Min/Max return: {np.min(returns):.0f} / {np.max(returns):.0f}")


if __name__ == "__main__":
    main()
