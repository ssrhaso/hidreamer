"""
Pytest conversion of testing/validate_vq.py

This test wraps the helper functions from the original script which require:
  - A trained HRVQ model checkpoint
  - Pre-extracted CNN embedding files for all 3 games

Both are NOT available in CI, so every test is skipped automatically unless
the required files exist locally.  The test logic is 100% preserved.

To run locally:
    pytest tests/test_vq.py -v
"""

from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
from tqdm import tqdm

# required paths
_CHECKPOINT   = Path("checkpoints/vq_model_best.pth")
_EMBEDDINGS   = {
    "Pong":     Path("data/embeddings_ALE_Pong-v5_cnn.npy"),
    "Breakout": Path("data/embeddings_ALE_Breakout-v5_cnn.npy"),
    "MsPacman": Path("data/embeddings_ALE_MsPacman-v5_cnn.npy"),
}

_all_files_present = _CHECKPOINT.exists() and all(p.exists() for p in _EMBEDDINGS.values())

_NEEDS_FILES = pytest.mark.skipif(
    not _all_files_present,
    reason=(
        "Requires trained HRVQ checkpoint + CNN embedding .npy files — "
        "skipped in CI.  Run locally after training."
    ),
)

# Mark every test in this file as 'integration' so the CI filter
# (-m "not integration and not gpu") skips them, and so does
# a quick local run with -m "not integration".
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helper functions inlined from testing/validate_vq.py
# ---------------------------------------------------------------------------

def _test_reconstruction_quality(model, embeddings, num_samples=1000):
    """Test how well VQ preserves embedding information."""
    device = next(model.parameters()).device
    indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    sample_embeddings = embeddings[indices]
    x = torch.from_numpy(sample_embeddings).float().to(device)
    x = x.unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        z_quantized, _, tokens = model(x)
    x_flat = x.squeeze().cpu().numpy()
    z_flat = z_quantized.squeeze().cpu().numpy()
    cosine_sim = np.sum(x_flat * z_flat, axis=1)
    l2_dist = np.linalg.norm(x_flat - z_flat, axis=1)
    return {
        'cosine_similarity_mean': float(cosine_sim.mean()),
        'cosine_similarity_std':  float(cosine_sim.std()),
        'l2_distance_mean':       float(l2_dist.mean()),
    }


def _test_temporal_consistency(model, embeddings, window_size=10000):
    """Test if consecutive frames get similar tokens."""
    device = next(model.parameters()).device
    consecutive_embeddings = embeddings[:window_size]
    x = torch.from_numpy(consecutive_embeddings).float().to(device)
    x = x.unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        _, _, tokens_list = model(x)
    tokens = tokens_list[0].squeeze().cpu().numpy()
    same_token = (tokens[:-1] == tokens[1:]).astype(float)
    temporal_smoothness = float(same_token.mean())
    change_rate = float((tokens[:-1] != tokens[1:]).sum() / len(tokens))
    runs, current_run = [], 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    avg_run_length = float(np.mean(runs)) if runs else 1.0
    return {
        'temporal_smoothness': temporal_smoothness,
        'token_change_rate':   change_rate,
        'avg_run_length':      avg_run_length,
    }


def _test_codebook_statistics(model, embeddings):
    """Full codebook usage analysis (per-layer for HRVQ)."""
    device = next(model.parameters()).device
    num_layers = model.num_layers
    all_tokens_by_layer = [[] for _ in range(num_layers)]
    batch_size = 512
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Tokenizing"):
        batch = embeddings[i:i + batch_size]
        x = torch.from_numpy(batch).float().to(device)
        x = x.unsqueeze(1).unsqueeze(2)
        with torch.no_grad():
            tokens_list = model.encode(x)
        for layer_idx, tokens in enumerate(tokens_list):
            all_tokens_by_layer[layer_idx].append(tokens.squeeze().cpu().numpy())
    all_tokens_by_layer = [np.concatenate(lt) for lt in all_tokens_by_layer]
    layer_results = []
    for layer_idx, all_tokens in enumerate(all_tokens_by_layer):
        token_counts = Counter(all_tokens.flatten())
        num_used = len(token_counts)
        total = len(all_tokens)
        token_probs = np.array([token_counts.get(i, 0) / total for i in range(256)])
        token_probs_nz = token_probs[token_probs > 0]
        entropy = -np.sum(token_probs_nz * np.log(token_probs_nz + 1e-10))
        layer_results.append({
            'layer':          layer_idx,
            'num_used_codes': int(num_used),
            'perplexity':     float(np.exp(entropy)),
        })
    return {'layers': layer_results}


def _test_multi_game_separation(model, embeddings_dict):
    """Test if different games use different token distributions (Layer 0 = shared vocab)."""
    device = next(model.parameters()).device
    game_tokens = {}
    for game_name, embeddings in embeddings_dict.items():
        x = torch.from_numpy(embeddings[:10000]).float().to(device)
        x = x.unsqueeze(1).unsqueeze(2)
        with torch.no_grad():
            tokens_list = model.encode(x)
        game_tokens[game_name] = tokens_list[0].squeeze().cpu().numpy()
    all_used = {game: set(tokens.flatten()) for game, tokens in game_tokens.items()}
    shared = set.intersection(*all_used.values())
    return {'shared_tokens': len(shared)}


# shared session fixture: load model + embeddings once

@pytest.fixture(scope="module")
def vq_model_and_embeddings():
    """Load HRVQ model and all game embeddings (L2-normalised).  Module-scoped."""
    import sys
    sys.path.insert(0, str(Path("src").resolve()))
    from vq import HRVQTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HRVQTokenizer(
        input_dim=384,
        num_codes_per_layer=256,
        num_layers=3,
        commitment_costs=[0.05, 0.25, 0.60],
    ).to(device)
    model.load_state_dict(torch.load(str(_CHECKPOINT), map_location=device))
    model.eval()

    embeddings_dict: dict = {}
    all_embs: list = []
    for game, path in _EMBEDDINGS.items():
        emb = np.load(str(path)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = emb / norms
        embeddings_dict[game] = emb
        all_embs.append(emb)

    all_embeddings = np.concatenate(all_embs, axis=0)
    return model, embeddings_dict, all_embeddings


""" TEST 1  —  Reconstruction quality """

@_NEEDS_FILES
def test_vq_reconstruction_cosine_similarity(vq_model_and_embeddings):
    """Mean cosine similarity between original and VQ-quantised embeddings must ≥ 0.70."""
    model, _, all_embeddings = vq_model_and_embeddings
    result = _test_reconstruction_quality(model, all_embeddings, num_samples=1000)

    assert result["cosine_similarity_mean"] >= 0.70, (
        f"FAIL reconstruction similarity: {result['cosine_similarity_mean']:.4f} < 0.70"
    )


""" TEST 2  —  Temporal consistency """

@_NEEDS_FILES
def test_vq_temporal_smoothness_in_range(vq_model_and_embeddings):
    """Temporal smoothness (consecutive token match rate) must be in [0.30, 0.70]."""
    model, embeddings_dict, _ = vq_model_and_embeddings
    result = _test_temporal_consistency(model, embeddings_dict["Pong"])

    s = result["temporal_smoothness"]
    assert 0.30 <= s <= 0.70, (
        f"FAIL temporal smoothness: {s:.4f} not in [0.30, 0.70]"
    )


""" TEST 3  —  Codebook statistics """

@_NEEDS_FILES
def test_vq_codebook_usage_all_layers(vq_model_and_embeddings):
    """Each layer must use ≥ 200/256 codebook entries and perplexity ≥ 100."""
    model, _, all_embeddings = vq_model_and_embeddings
    result = _test_codebook_statistics(model, all_embeddings)

    for layer_info in result["layers"]:
        idx  = layer_info["layer"]
        used = layer_info["num_used_codes"]
        perp = layer_info["perplexity"]
        assert used >= 200, (
            f"FAIL Layer {idx} codebook collapse: only {used}/256 codes used"
        )
        assert perp >= 100, (
            f"FAIL Layer {idx} low perplexity: {perp:.2f} (expected ≥ 100)"
        )


""" TEST 4  —  Multi-game token separation """

@_NEEDS_FILES
def test_vq_shared_vocabulary_exists(vq_model_and_embeddings):
    """Layer 0 must share ≥ 30 tokens across all 3 games (cross-game vocabulary)."""
    model, embeddings_dict, _ = vq_model_and_embeddings
    result = _test_multi_game_separation(model, embeddings_dict)

    shared = result["shared_tokens"]
    assert shared >= 30, (
        f"FAIL insufficient shared vocabulary: {shared} tokens shared (expected ≥ 30)"
    )
