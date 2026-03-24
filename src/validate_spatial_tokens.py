"""
VALIDATE SPATIAL TOKENS — Diagnostic for encoder quality.

Checks codebook utilization, inter-patch variance, and unique state tuples.
Run after extract_spatial_tokens.py to verify the encoder is producing
distinguishable game states.

Usage:
    python src/validate_spatial_tokens.py --tokens_dir checkpoints/spatial_tokens --game Pong-v5
"""

import argparse
import numpy as np
from pathlib import Path


def validate(tokens_dir: str, game: str, sample_size: int = 500):
    game_dir = Path(tokens_dir) / game

    l0 = np.load(str(game_dir / 'spatial_tokens_l0.npy'))
    l1 = np.load(str(game_dir / 'spatial_tokens_l1.npy'))
    l2 = np.load(str(game_dir / 'spatial_tokens_l2.npy'))

    N = len(l0)
    print(f"\nSPATIAL TOKEN VALIDATION: {game}")
    print(f"{'='*50}")
    print(f"Frames: {N:,}")

    # Per-level stats
    for name, tokens, total in [('L0', l0, 16), ('L1', l1, 64), ('L2', l2, 64)]:
        unique = len(np.unique(tokens))
        patch_var = tokens.astype(float).var(axis=1).mean()
        print(f"\n  {name}:")
        print(f"    Shape:       {tokens.shape}")
        print(f"    Codes used:  {unique}/{total} ({100*unique/total:.1f}%)")
        print(f"    Patch var:   {patch_var:.2f}  (>0 = spatial info preserved)")

    # Unique full-state tuples
    combined = np.concatenate([l0, l1, l2], axis=1)

    sample_n = min(sample_size, N)
    sample_tuples = set(tuple(row) for row in combined[:sample_n])
    all_tuples = set(tuple(row) for row in combined)

    print(f"\n  Unique state tuples:")
    print(f"    First {sample_n}: {len(sample_tuples)}/{sample_n} ({100*len(sample_tuples)/sample_n:.1f}%)")
    print(f"    All {N:,}:  {len(all_tuples)}/{N} ({100*len(all_tuples)/N:.1f}%)")

    # Verdict
    ratio = len(sample_tuples) / sample_n
    print(f"\n  {'='*50}")
    if ratio > 0.8:
        print(f"  PASS — strong state discrimination ({ratio:.1%})")
    elif ratio > 0.5:
        print(f"  MARGINAL — moderate discrimination ({ratio:.1%}), may limit policy")
    else:
        print(f"  FAIL — poor discrimination ({ratio:.1%}), encoder needs improvement")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens_dir', type=str, default='checkpoints/spatial_tokens')
    parser.add_argument('--game', type=str, default='Pong-v5')
    parser.add_argument('--sample_size', type=int, default=500)
    args = parser.parse_args()

    validate(args.tokens_dir, args.game, args.sample_size)