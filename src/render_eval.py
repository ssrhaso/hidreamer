"""
Render evaluation episodes to GIF for debugging.
Usage: python src/render_eval.py --checkpoint checkpoints/policy/best_policy.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import ale_py
import gymnasium as gym
import yaml

from world_model import HierarchicalWorldModel, WorldModelConfig
from vq import HRVQTokenizer
from encoder_v1 import AtariCNNEncoder
from policy import ActorNetwork, HiddenStateFeatureExtractor


def load_all(checkpoint_path, config_path="configs/policy.yaml", device="cuda"):
    """Load frozen models + trained policy from checkpoint."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    paths = config['frozen_models']
    
    # World model
    wm_config = WorldModelConfig.from_yaml("configs/worldmodel.yaml")
    wm = HierarchicalWorldModel(wm_config).to(device)
    wm_ckpt = torch.load(paths['world_model'], map_location=device, weights_only=False)
    wm.load_state_dict(wm_ckpt['model_state_dict'])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    
    # HRVQ
    hrvq = HRVQTokenizer(input_dim=384, num_codes_per_layer=256, num_layers=3,
                          commitment_costs=[0.05, 0.25, 0.60]).to(device)
    hrvq.load_state_dict(torch.load(paths['hrvq_tokenizer'], map_location=device, weights_only=False))
    hrvq.eval()
    for p in hrvq.parameters():
        p.requires_grad = False
    
    # Encoder
    encoder = AtariCNNEncoder(input_channels=4, embedding_dim=384).to(device)
    enc_ckpt = torch.load(paths['encoder'], map_location=device, weights_only=False)
    encoder.load_state_dict(enc_ckpt['model_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    # Feature extractor + actor
    feat_ext = HiddenStateFeatureExtractor(d_model=384, use_projection=True).to(device)
    actor = ActorNetwork(feat_dim=1152, num_actions=6, hidden_dim=512, unimix=0.01).to(device)
    
    # Load trained weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feat_ext.load_state_dict(ckpt['feature_extractor_state_dict'])
    actor.load_state_dict(ckpt['policy_state_dict'])
    feat_ext.eval()
    actor.eval()
    
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}, best_return={ckpt.get('best_return', '?')}")
    
    return wm, hrvq, encoder, feat_ext, actor, config


@torch.no_grad()
def encode_obs(obs, encoder, hrvq, device):
    """Raw obs -> 3 HRVQ tokens."""
    frame = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    emb = encoder(frame)
    emb = emb.unsqueeze(1).unsqueeze(2)
    token_list = hrvq.encode(emb)
    t0 = token_list[0].squeeze()
    t1 = token_list[1].squeeze()
    t2 = token_list[2].squeeze()
    return torch.stack([t0, t1, t2], dim=-1)  # (3,)


@torch.no_grad()
def get_features(wm, tokens_history, actions_history, feat_ext, device, max_ctx=16):
    """Run WM on context to get hidden-state features."""
    ctx_tokens = torch.stack(tokens_history[-max_ctx:]).unsqueeze(0).to(device)
    ctx_actions = torch.tensor(actions_history[-max_ctx:]).unsqueeze(0).to(device)
    
    x = wm.embedding(ctx_tokens, ctx_actions)
    mask = wm._get_mask(x.size(1), x.device)
    for block in wm.blocks:
        x = block(x, mask=mask)
    x = wm.ln_final(x)
    
    t = ctx_tokens.size(1)
    last_start = (t - 1) * 4
    hidden = x[:, last_start:last_start+3, :]
    return feat_ext(hidden)


def run_episode(wm, hrvq, encoder, feat_ext, actor, env, device, greedy=True):
    """Run one episode, return frames + actions + rewards."""
    obs, _ = env.reset()
    
    frames = []
    actions_taken = []
    rewards_got = []
    action_history = []
    token_history = []
    done = False
    
    while not done:
        # Save raw frame (the preprocessed 84x84 grayscale)
        # Use last channel of the frame stack for display
        frames.append(obs[-1].copy())
        
        tokens = encode_obs(obs, encoder, hrvq, device)
        token_history.append(tokens.cpu())
        
        while len(action_history) < len(token_history):
            action_history.append(0)
        
        features = get_features(wm, token_history, action_history, feat_ext, device)
        
        dist = actor(features)
        if greedy:
            action = dist.probs.argmax(dim=-1).item()
        else:
            action = dist.sample().item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        action_history.append(action)
        done = terminated or truncated
        
        actions_taken.append(action)
        rewards_got.append(reward)
        obs = next_obs
    
    return frames, actions_taken, rewards_got


def frames_to_gif(frames, path, fps=30):
    """Save list of numpy frames as GIF."""
    imgs = []
    for f in frames:
        # Scale up for visibility (84x84 is tiny)
        img = Image.fromarray(f, mode='L')
        img = img.resize((336, 336), Image.NEAREST)
        imgs.append(img)
    
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=1000//fps, loop=0)
    print(f"  Saved {path} ({len(frames)} frames)")


def run_random_episode(env):
    """Run one random episode for comparison."""
    obs, _ = env.reset()
    frames = []
    rewards = []
    done = False
    while not done:
        frames.append(obs[-1].copy())
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return frames, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy/best_policy.pt")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="eval_renders")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load everything
    wm, hrvq, encoder, feat_ext, actor, config = load_all(args.checkpoint, device=device)
    
    # Create env
    env = gym.make("ALE/Pong-v5")
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, screen_size=84,
                                           grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    # ACTION NAMES for Pong
    action_names = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}
    
    # Run policy episodes
    print(f"\n=== POLICY EPISODES (greedy) ===")
    for ep in range(args.num_episodes):
        frames, actions, rewards = run_episode(wm, hrvq, encoder, feat_ext, actor, env, device, greedy=True)
        total_return = sum(rewards)
        
        # Action distribution
        from collections import Counter
        action_counts = Counter(actions)
        total_actions = len(actions)
        
        print(f"\nEpisode {ep+1}: return={total_return:.0f}, length={len(frames)}")
        print(f"  Action distribution:")
        for a in sorted(action_counts.keys()):
            pct = 100 * action_counts[a] / total_actions
            print(f"    {action_names.get(a, f'ACT_{a}')}: {action_counts[a]} ({pct:.1f}%)")
        
        # Count reward events
        pos_rewards = sum(1 for r in rewards if r > 0)
        neg_rewards = sum(1 for r in rewards if r < 0)
        print(f"  Points scored: {pos_rewards}, Points conceded: {neg_rewards}")
        
        frames_to_gif(frames, os.path.join(args.output_dir, f"policy_ep{ep+1}.gif"))
    
    # Run random baseline for comparison
    print(f"\n=== RANDOM BASELINE ===")
    for ep in range(2):
        frames, rewards = run_random_episode(env)
        total_return = sum(rewards)
        print(f"Random episode {ep+1}: return={total_return:.0f}, length={len(frames)}")
        frames_to_gif(frames, os.path.join(args.output_dir, f"random_ep{ep+1}.gif"))
    
    env.close()
    print(f"\nAll renders saved to {args.output_dir}/")
    print("Download the GIFs to see what the agent is doing.")


if __name__ == "__main__":
    main()
Ccc
