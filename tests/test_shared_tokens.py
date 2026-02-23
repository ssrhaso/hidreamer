"""
Pytest conversion of testing/analyze_shared_tokens.py

Uses the tiny_rsvq_token_paths fixture from conftest.py which generates dummy
token files with a known number of shared tokens — no real checkpoint needed.

The original script was analysis-only (print statements, no assertions).
We preserve all the computation logic and add meaningful assertions.
"""

import numpy as np
import pytest
from collections import Counter


""" HELPER: replicate the original script's analysis """

def _analyse_shared_tokens(token_paths: dict, games: list) -> dict:
    """Run the full shared-token analysis from the original script.

    Parameters
    ----------
    token_paths : dict mapping game name → path to layer-0 .npy file
    games       : ordered list of game names

    Returns
    -------
    dict with keys: shared_ids, per_game_shared_usage, temporal_early_shared
    """
    tokens_by_game = {
        g: np.load(token_paths[g]).flatten() for g in games
    }

    # Shared token set (set intersection)
    shared: set = set(tokens_by_game[games[0]])
    for g in games[1:]:
        shared &= set(tokens_by_game[g])

    per_game_shared_usage: dict = {}
    for game in games:
        toks = tokens_by_game[game]
        per_game_shared_usage[game] = sum(1 for t in toks if t in shared) / len(toks)

    # Temporal: first 1000 frames
    temporal_early: dict = {}
    for game in games:
        toks = tokens_by_game[game][:1000]
        temporal_early[game] = sum(1 for t in toks if t in shared) / len(toks)

    # Frequency analysis
    freq_stats: dict = {}
    for game in games:
        toks     = tokens_by_game[game]
        all_freq = Counter(toks)
        s_freq   = [all_freq[t] for t in shared if t in all_freq]
        ns_keys  = set(all_freq.keys()) - shared
        ns_freq  = [all_freq[t] for t in ns_keys]
        freq_stats[game] = {
            "shared_mean":     float(np.mean(s_freq))   if s_freq  else 0.0,
            "nonshared_mean":  float(np.mean(ns_freq))  if ns_freq else 0.0,
        }

    return {
        "shared_ids":            sorted(shared),
        "num_shared":            len(shared),
        "per_game_shared_usage": per_game_shared_usage,
        "temporal_early_shared": temporal_early,
        "freq_stats":            freq_stats,
    }


""" TESTS """

def test_shared_tokens_analysis_runs(tiny_rsvq_token_paths):
    """The analysis must complete without exceptions and find at least one shared token."""
    info = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])
    assert isinstance(result["num_shared"], int)
    assert result["num_shared"] > 0, "Analysis found zero shared tokens — fixture must guarantee overlap"


def test_shared_tokens_count_above_minimum(tiny_rsvq_token_paths):
    """At least `expected_min_shared` tokens must be shared across all 3 games.

    The fixture constructs files with codes [0..49] guaranteed to appear in
    every game, so we must see ≥ 45 shared tokens.
    """
    info   = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])

    assert result["num_shared"] >= info["expected_min_shared"], (
        f"Too few shared tokens: {result['num_shared']} < {info['expected_min_shared']}"
    )


def test_shared_tokens_within_valid_codebook_range(tiny_rsvq_token_paths):
    """All shared token IDs must be valid codebook indices [0, 255]."""
    info   = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])

    bad = [t for t in result["shared_ids"] if not (0 <= t <= 255)]
    assert not bad, f"Shared token IDs out of codebook range: {bad}"


def test_shared_usage_per_game_is_positive(tiny_rsvq_token_paths):
    """Each game must actually use at least some of the shared tokens (> 0%)."""
    info   = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])

    for game, usage in result["per_game_shared_usage"].items():
        assert usage > 0, f"Game '{game}' has 0 shared-token usage"


def test_all_three_games_analysed(tiny_rsvq_token_paths):
    """Analysis result must cover all three configured games."""
    info   = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])

    for game in info["games"]:
        assert game in result["per_game_shared_usage"], (
            f"Game '{game}' missing from per_game_shared_usage"
        )
        assert game in result["temporal_early_shared"], (
            f"Game '{game}' missing from temporal_early_shared"
        )


def test_temporal_early_shared_usage_is_positive(tiny_rsvq_token_paths):
    """Shared-token usage rate for the first 1 000 frames must be > 0.

    The fixture places codes [0..49] in every sequence, so shared tokens
    must appear in every game's first 1 000 frames.
    """
    info   = tiny_rsvq_token_paths
    result = _analyse_shared_tokens(info["paths"], info["games"])

    for game, usage in result["temporal_early_shared"].items():
        assert usage > 0.0, (
            f"Game '{game}' has 0 shared-token usage in first 1000 frames — "
            "fixture guarantees overlap"
        )
