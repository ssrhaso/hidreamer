"""
MIX TOKEN DATA — Combine random-policy and heuristic-agent token datasets.

Usage:
    python src/mix_token_data.py \
        --sources checkpoints/spatial_tokens/Pong-v5 data/Pong-v5-heuristic \
        --output checkpoints/spatial_tokens/Pong-v5-mixed \
        --max_total 100000

This concatenates token arrays from multiple source directories, shuffles
episode-aware (preserves within-episode ordering), and saves to a new directory
that from_numpy_data() can load directly.
"""

import os
import argparse
import numpy as np


def load_source(src_dir):
    """Load all token arrays from one source directory."""
    # Try both conventions: spatial_tokens_* and raw arrays
    files = {}
    for name, alternatives in [
        ('l0', ['spatial_tokens_l0.npy']),
        ('l1', ['spatial_tokens_l1.npy']),
        ('l2', ['spatial_tokens_l2.npy']),
        ('actions', ['actions.npy']),
        ('rewards', ['rewards.npy']),
        ('dones', ['dones.npy']),
    ]:
        for alt in alternatives:
            path = os.path.join(src_dir, alt)
            if os.path.exists(path):
                files[name] = np.load(path)
                break
        if name not in files:
            raise FileNotFoundError(f"Missing {name} in {src_dir}")

    N = len(files['actions'])
    print(f"  {src_dir}: {N} transitions, "
          f"{(files['rewards'] > 0).sum()} pos rewards, "
          f"{(files['rewards'] < 0).sum()} neg rewards")
    return files


def split_episodes(dones):
    """Return list of (start, end) index pairs for each episode."""
    ends = np.where(dones)[0]
    episodes = []
    start = 0
    for end in ends:
        episodes.append((start, end + 1))
        start = end + 1
    if start < len(dones):
        episodes.append((start, len(dones)))
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Mix spatial token datasets")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="Source directories to combine")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--max_total", type=int, default=100_000,
                        help="Maximum total transitions (default: 100k)")
    parser.add_argument("--shuffle_episodes", action="store_true", default=True,
                        help="Shuffle episode order (preserves within-episode order)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading sources...")
    all_sources = [load_source(s) for s in args.sources]

    # Concatenate all sources
    combined = {}
    for key in ['l0', 'l1', 'l2', 'actions', 'rewards', 'dones']:
        combined[key] = np.concatenate([s[key] for s in all_sources], axis=0)

    total = len(combined['actions'])
    print(f"\nCombined: {total} transitions")
    print(f"  Positive rewards: {(combined['rewards'] > 0).sum()}")
    print(f"  Negative rewards: {(combined['rewards'] < 0).sum()}")

    # Shuffle by episode to maintain temporal coherence
    if args.shuffle_episodes:
        episodes = split_episodes(combined['dones'])
        np.random.shuffle(episodes)

        # Rebuild arrays in shuffled episode order
        shuffled = {k: [] for k in combined}
        count = 0
        for start, end in episodes:
            if count + (end - start) > args.max_total:
                # Take partial episode to fill up to max_total
                take = args.max_total - count
                if take > 0:
                    for k in combined:
                        shuffled[k].append(combined[k][start:start + take])
                    count += take
                break
            for k in combined:
                shuffled[k].append(combined[k][start:end])
            count += end - start

        combined = {k: np.concatenate(v, axis=0) for k, v in shuffled.items()}
    else:
        # Just truncate
        for k in combined:
            combined[k] = combined[k][:args.max_total]

    N = len(combined['actions'])
    print(f"\nFinal dataset: {N} transitions")
    print(f"  Positive rewards: {(combined['rewards'] > 0).sum()}")
    print(f"  Negative rewards: {(combined['rewards'] < 0).sum()}")
    print(f"  Reward density: {(combined['rewards'].abs() > 0).sum() / N * 100:.2f}%")

    # Save
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "spatial_tokens_l0.npy"),
            combined['l0'].astype(np.int16))
    np.save(os.path.join(args.output, "spatial_tokens_l1.npy"),
            combined['l1'].astype(np.int16))
    np.save(os.path.join(args.output, "spatial_tokens_l2.npy"),
            combined['l2'].astype(np.int16))
    np.save(os.path.join(args.output, "actions.npy"),
            combined['actions'].astype(np.int16))
    np.save(os.path.join(args.output, "rewards.npy"),
            combined['rewards'].astype(np.float32))
    np.save(os.path.join(args.output, "dones.npy"),
            combined['dones'].astype(bool))

    print(f"\nSaved to {args.output}/")
    print("  spatial_tokens_l0.npy, spatial_tokens_l1.npy, spatial_tokens_l2.npy")
    print("  actions.npy, rewards.npy, dones.npy")


if __name__ == "__main__":
    main()
