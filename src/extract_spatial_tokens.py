"""
EXTRACT SPATIAL TOKENS — Pre-compute spatial token sequences for all games.

Loads the trained SpatialAtariEncoder + SpatialHRVQTokenizer and tokenizes
all 100k frames per game, saving compact token arrays for world model training.

Output format (per game, in tokens_dir/{game}/):
    spatial_tokens_l0.npy  — (N, 4)  int16  L0 token indices
    spatial_tokens_l1.npy  — (N, 16) int16  L1 token indices
    spatial_tokens_l2.npy  — (N, 16) int16  L2 token indices
    actions.npy            — (N,)    int16  action taken at each step
    rewards.npy            — (N,)    float32 reward received
    dones.npy              — (N,)    bool   episode termination flags

Usage:
    python src/extract_spatial_tokens.py \\
        --checkpoint checkpoints/spatial_encoder/spatial_encoder_best.pt \\
        --config configs/encoder_spatial.yaml \\
        --replay_dir data \\
        --output_dir checkpoints/spatial_tokens \\
        --games Pong-v5 Breakout-v5 MsPacman-v5
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder_v2 import SpatialAtariEncoder
from vq_spatial import SpatialHRVQTokenizer


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_models(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained encoder + tokenizer from checkpoint."""
    d_model = config['model']['d_model']

    encoder = SpatialAtariEncoder(
        input_channels=config['model']['input_channels'],
        d_model=d_model,
    ).to(device)

    tokenizer = SpatialHRVQTokenizer(
        d_model=d_model,
        num_codes=config['tokenizer']['num_codes'],
        commitment_costs=config['tokenizer']['commitment_costs'],
        decay=config['tokenizer']['decay'],
        epsilon=config['tokenizer']['epsilon'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    tokenizer.load_state_dict(ckpt['tokenizer_state_dict'])

    encoder.eval()
    tokenizer.eval()

    val_loss = ckpt.get('best_val_loss', float('nan'))
    epoch = ckpt.get('epoch', '?')
    print(f"  Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.5f}")

    return encoder, tokenizer


def load_game_data(replay_dir: str, game: str):
    """
    Load replay data for a single game.

    Returns
    -------
    obs : (N, 4, 84, 84) uint8
    actions : (N,) int16
    rewards : (N,) float32
    dones : (N,) bool
    """
    game_dir = Path(replay_dir) / game

    # Try common formats
    obs = None
    for fname in ['frames.npy', 'obs.npy', 'observations.npy']:
        p = game_dir / fname
        if p.exists():
            obs = np.load(str(p))
            print(f"  Frames: {obs.shape} from {p}")
            break

    if obs is None:
        # Try npz sequences
        seq_path = game_dir / 'sequences.npz'
        if seq_path.exists():
            d = np.load(str(seq_path))
            obs = d['obs'].reshape(-1, 4, 84, 84)
            print(f"  Frames (from sequences): {obs.shape}")

    if obs is None:
        raise FileNotFoundError(f"No observation data found in {game_dir}")

    # Load metadata — use zeros if not available
    N = len(obs)

    actions_path = game_dir / 'actions.npy'
    rewards_path = game_dir / 'rewards.npy'
    dones_path   = game_dir / 'dones.npy'

    actions = np.load(str(actions_path)).flatten()[:N].astype(np.int16) \
              if actions_path.exists() else np.zeros(N, dtype=np.int16)
    rewards = np.load(str(rewards_path)).flatten()[:N].astype(np.float32) \
              if rewards_path.exists() else np.zeros(N, dtype=np.float32)
    dones   = np.load(str(dones_path)).flatten()[:N].astype(bool) \
              if dones_path.exists() else np.zeros(N, dtype=bool)

    print(f"  Actions: {actions.shape}, Rewards: {rewards.shape}, Dones: {dones.shape}")
    return obs, actions, rewards, dones


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_tokens_for_game(
    encoder: SpatialAtariEncoder,
    tokenizer: SpatialHRVQTokenizer,
    obs: np.ndarray,          # (N, 4, 84, 84) uint8
    batch_size: int,
    device: torch.device,
):
    """
    Tokenise all frames in batches.

    Returns
    -------
    tokens_l0 : (N, 4)  int16
    tokens_l1 : (N, 16) int16
    tokens_l2 : (N, 16) int16
    """
    N = len(obs)
    all_l0 = np.empty((N, 4),  dtype=np.int16)
    all_l1 = np.empty((N, 16), dtype=np.int16)
    all_l2 = np.empty((N, 16), dtype=np.int16)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(obs[start:end].astype(np.float32) / 255.0).to(device)

        spatial_feats = encoder(batch)
        token_dict = tokenizer.encode(spatial_feats)

        all_l0[start:end] = token_dict['l0'].cpu().numpy().astype(np.int16)
        all_l1[start:end] = token_dict['l1'].cpu().numpy().astype(np.int16)
        all_l2[start:end] = token_dict['l2'].cpu().numpy().astype(np.int16)

        if start % 10000 == 0:
            print(f"  Processed {end}/{N} frames...", end='\r')

    print(f"  Processed {N}/{N} frames.   ")
    return all_l0, all_l1, all_l2


def print_codebook_stats(tokens_l0, tokens_l1, tokens_l2, num_codes=256):
    """Print codebook utilisation per level."""
    for name, tokens in [('L0', tokens_l0), ('L1', tokens_l1), ('L2', tokens_l2)]:
        unique = len(np.unique(tokens.reshape(-1)))
        print(f"  {name}: {unique}/{num_codes} codes ({100*unique/num_codes:.1f}%) used")

    # Check that different spatial patches use different codes
    # (if all patches collapse to the same code, the encoder is still degenerate)
    for name, tokens in [('L0', tokens_l0), ('L1', tokens_l1), ('L2', tokens_l2)]:
        # Variance across patch dimension (should be > 0 if spatial info is preserved)
        patch_var = tokens.astype(float).var(axis=1).mean()
        print(f"  {name} inter-patch variance: {patch_var:.2f}  "
              f"(>0 = spatial info preserved)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Extract spatial tokens')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/spatial_encoder/spatial_encoder_best.pt')
    parser.add_argument('--config',    type=str,
                        default='configs/encoder_spatial.yaml')
    parser.add_argument('--replay_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/spatial_tokens')
    parser.add_argument('--games', nargs='+',
                        default=['Pong-v5', 'Breakout-v5', 'MsPacman-v5'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nEXTRACT SPATIAL TOKENS")
    print(f"Device: {device}")
    print(f"Games:  {args.games}")

    print(f"\nLoading models from {args.checkpoint}...")
    encoder, tokenizer = load_models(args.checkpoint, config, device)

    output_dir = Path(args.output_dir)
    num_codes = config['tokenizer']['num_codes']

    for game in args.games:
        print(f"\n{'='*50}")
        print(f"Processing: {game}")
        print(f"{'='*50}")

        # Load observations
        obs, actions, rewards, dones = load_game_data(args.replay_dir, game)

        # Extract tokens
        tokens_l0, tokens_l1, tokens_l2 = extract_tokens_for_game(
            encoder, tokenizer, obs,
            batch_size=args.batch_size,
            device=device,
        )

        # Stats
        print(f"\nCodebook utilisation:")
        print_codebook_stats(tokens_l0, tokens_l1, tokens_l2, num_codes)

        reward_counts = {
            'positive': (rewards > 0).sum(),
            'negative': (rewards < 0).sum(),
            'zero':     (rewards == 0).sum(),
        }
        print(f"\nReward distribution: {reward_counts}")

        # Save
        game_out = output_dir / game
        game_out.mkdir(parents=True, exist_ok=True)

        np.save(str(game_out / 'spatial_tokens_l0.npy'), tokens_l0)
        np.save(str(game_out / 'spatial_tokens_l1.npy'), tokens_l1)
        np.save(str(game_out / 'spatial_tokens_l2.npy'), tokens_l2)
        np.save(str(game_out / 'actions.npy'), actions)
        np.save(str(game_out / 'rewards.npy'), rewards)
        np.save(str(game_out / 'dones.npy'), dones)

        print(f"\nSaved to {game_out}/")
        print(f"  spatial_tokens_l0.npy  — {tokens_l0.shape}")
        print(f"  spatial_tokens_l1.npy  — {tokens_l1.shape}")
        print(f"  spatial_tokens_l2.npy  — {tokens_l2.shape}")
        print(f"  actions.npy, rewards.npy, dones.npy")

    print(f"\nExtraction complete for all games.")


if __name__ == "__main__":
    main()
