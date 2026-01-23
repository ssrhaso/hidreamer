"""
Comprehensive VQ Model Validation Script

Tests:
1. Codebook statistics (usage, perplexity)
2. Reconstruction quality (embedding similarity)
3. Temporal consistency (token smoothness)
4. Multi-game separation (token distribution per game)
5. Visualization of token patterns
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from vq import VQTokenizer, HRVQTokenizer


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained HRVQ model"""
    model = HRVQTokenizer(
        input_dim=384,
        num_codes_per_layer=256,
        num_layers=3,
        commitment_costs=[0.15, 0.25, 0.40],
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✓ Loaded HRVQ model (3 layers) from {checkpoint_path}")
    return model


def test_reconstruction_quality(model, embeddings, num_samples=1000):
    """Test how well VQ preserves embedding information"""
    print("\n=== Test 1: Reconstruction Quality ===")
    
    device = next(model.parameters()).device
    indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
    sample_embeddings = embeddings[indices]
    
    # Convert to tensor
    x = torch.from_numpy(sample_embeddings).float().to(device)
    x = x.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 384]
    
    with torch.no_grad():
        z_quantized, _, tokens = model(x)
    
    # Compare original vs quantized
    x_flat = x.squeeze().cpu().numpy()
    z_flat = z_quantized.squeeze().cpu().numpy()
    
    # Cosine similarity (since embeddings are L2-normalized)
    cosine_sim = np.sum(x_flat * z_flat, axis=1)
    
    # L2 distance
    l2_dist = np.linalg.norm(x_flat - z_flat, axis=1)
    
    print(f"  Cosine similarity: {cosine_sim.mean():.4f} ± {cosine_sim.std():.4f}")
    print(f"  L2 distance:       {l2_dist.mean():.4f} ± {l2_dist.std():.4f}")
    print(f"  Range: [{cosine_sim.min():.4f}, {cosine_sim.max():.4f}]")
    
    # Pass/fail criteria
    if cosine_sim.mean() > 0.85:
        print("   PASS: High reconstruction fidelity")
    elif cosine_sim.mean() > 0.70:
        print("    ACCEPTABLE: Moderate information loss")
    else:
        print("   FAIL: Significant information loss")
    
    return {
        'cosine_similarity_mean': float(cosine_sim.mean()),
        'cosine_similarity_std': float(cosine_sim.std()),
        'l2_distance_mean': float(l2_dist.mean()),
    }


def test_temporal_consistency(model, embeddings, window_size=10000):
    """Test if consecutive frames get similar tokens"""
    print("\n=== Test 2: Temporal Consistency ===")
    
    device = next(model.parameters()).device
    
    # Take consecutive embeddings
    consecutive_embeddings = embeddings[:window_size]
    x = torch.from_numpy(consecutive_embeddings).float().to(device)
    x = x.unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        _, _, tokens_list = model(x)
    
    # Analyze Layer 0 (coarse/shared layer) for temporal consistency
    tokens = tokens_list[0].squeeze().cpu().numpy()
    
    # Compute token similarity between consecutive frames
    same_token = (tokens[:-1] == tokens[1:]).astype(float)
    temporal_smoothness = same_token.mean()
    
    # Token change rate
    changes = (tokens[:-1] != tokens[1:]).sum()
    change_rate = changes / len(tokens)
    
    # Average token run length
    runs = []
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    avg_run_length = np.mean(runs) if runs else 1.0
    
    print(f"  Consecutive token similarity: {temporal_smoothness:.4f} ({temporal_smoothness*100:.1f}%)")
    print(f"  Token change rate:            {change_rate:.4f}")
    print(f"  Average token run length:     {avg_run_length:.2f} frames")
    
    # Pass/fail criteria
    if 0.30 <= temporal_smoothness <= 0.70:
        print("   PASS: Balanced temporal dynamics")
    elif temporal_smoothness > 0.70:
        print("    WARNING: Too smooth, may be under-utilizing codebook")
    else:
        print("    WARNING: Too volatile, tokens may be noisy")
    
    return {
        'temporal_smoothness': float(temporal_smoothness),
        'token_change_rate': float(change_rate),
        'avg_run_length': float(avg_run_length),
    }


def test_codebook_statistics(model, embeddings):
    """Full codebook usage analysis (per-layer for HRVQ)"""
    print("\n=== Test 3: Codebook Statistics (Per Layer) ===")
    
    device = next(model.parameters()).device
    num_layers = model.num_layers
    
    # Tokenize all embeddings
    all_tokens_by_layer = [[] for _ in range(num_layers)]
    batch_size = 512
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="  Tokenizing"):
        batch = embeddings[i:i+batch_size]
        x = torch.from_numpy(batch).float().to(device)
        x = x.unsqueeze(1).unsqueeze(2)
        
        with torch.no_grad():
            tokens_list = model.encode(x)
        
        for layer_idx, tokens in enumerate(tokens_list):
            all_tokens_by_layer[layer_idx].append(tokens.squeeze().cpu().numpy())
    
    # Concatenate per layer
    all_tokens_by_layer = [np.concatenate(layer_tokens) for layer_tokens in all_tokens_by_layer]
    
    # Analyze each layer
    layer_results = []
    for layer_idx, all_tokens in enumerate(all_tokens_by_layer):
        print(f"\n  Layer {layer_idx}:")
        
        # Count token frequencies
        token_counts = Counter(all_tokens.flatten())
        num_used = len(token_counts)
        
        # Perplexity
        total = len(all_tokens)
        token_probs = np.array([token_counts.get(i, 0) / total for i in range(256)])
        token_probs_nonzero = token_probs[token_probs > 0]
        entropy = -np.sum(token_probs_nonzero * np.log(token_probs_nonzero + 1e-10))
        perplexity = np.exp(entropy)
        
        print(f"    Codebook usage:    {num_used}/256 ({num_used/256*100:.1f}%)")
        print(f"    Perplexity:        {perplexity:.2f} (max: 256)")
        print(f"    Effective usage:   {perplexity/256*100:.1f}% of theoretical max")
        
        # Token distribution statistics
        frequencies = list(token_counts.values())
        print(f"    Token freq (min/max): {min(frequencies)} / {max(frequencies)}")
        print(f"    Token freq (mean):    {np.mean(frequencies):.1f} ± {np.std(frequencies):.1f}")
        
        # Pass/fail
        if num_used == 256 and perplexity > 200:
            print("     PASS: Excellent codebook utilization")
        elif num_used >= 240 and perplexity > 150:
            print("     ACCEPTABLE: Good codebook utilization")
        else:
            print("     FAIL: Codebook collapse detected")
        
        layer_results.append({
            'layer': layer_idx,
            'num_used_codes': int(num_used),
            'perplexity': float(perplexity),
        })
    
    return {'layers': layer_results}


def test_multi_game_separation(model, embeddings_dict):
    """Test if different games use different token distributions (Layer 0 = shared vocabulary)"""
    print("\n=== Test 4: Multi-Game Token Distribution (Layer 0 Analysis) ===")
    
    device = next(model.parameters()).device
    game_tokens = {}
    
    for game_name, embeddings in embeddings_dict.items():
        # Tokenize game embeddings
        x = torch.from_numpy(embeddings[:10000]).float().to(device)
        x = x.unsqueeze(1).unsqueeze(2)
        
        with torch.no_grad():
            tokens_list = model.encode(x)
        
        # Use Layer 0 (coarse/shared layer) for cross-game analysis
        game_tokens[game_name] = tokens_list[0].squeeze().cpu().numpy()
    
    # Compare token distributions
    print("\n  Token distribution overlap:")
    games = list(game_tokens.keys())
    
    for i, game1 in enumerate(games):
        for game2 in games[i+1:]:
            # Compute distribution overlap (Bhattacharyya coefficient)
            dist1 = np.bincount(game_tokens[game1], minlength=256) / len(game_tokens[game1])
            dist2 = np.bincount(game_tokens[game2], minlength=256) / len(game_tokens[game2])
            
            overlap = np.sum(np.sqrt(dist1 * dist2))
            
            print(f"  {game1} ↔ {game2}: {overlap:.4f}")
    
    # Check unique vs shared tokens
    all_used = {}
    for game, tokens in game_tokens.items():
        unique = set(tokens.flatten())
        all_used[game] = unique
        print(f"\n  {game}: {len(unique)} unique tokens used")
    
    # Shared tokens across all games
    shared = set.intersection(*all_used.values())
    print(f"\n  Shared tokens (all games): {len(shared)}/256")
    
    if len(shared) > 100:
        print("  ✅ Strong shared vocabulary (good generalization)")
    elif len(shared) > 50:
        print("  ⚠️  Moderate shared vocabulary")
    else:
        print("  ❌ Weak shared vocabulary (games may be too separated)")
    
    return {
        'game_tokens': {k: v.tolist()[:1000] for k, v in game_tokens.items()},  # Save subset
        'shared_tokens': len(shared),
    }


def visualize_token_patterns(tokens, save_path='results/token_patterns.png'):
    """Visualize token sequence patterns"""
    print("\n=== Test 5: Token Pattern Visualization ===")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. Token sequence (first 1000 frames)
    axes[0].plot(tokens[:1000], linewidth=0.5, alpha=0.7)
    axes[0].set_title('Token Sequence (First 1000 Frames)', fontsize=12)
    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Token ID')
    axes[0].set_ylim(-5, 260)
    axes[0].grid(alpha=0.3)
    
    # 2. Token histogram
    axes[1].hist(tokens, bins=256, edgecolor='black', alpha=0.7)
    axes[1].set_title('Token Frequency Distribution', fontsize=12)
    axes[1].set_xlabel('Token ID')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)
    
    # 3. Token transition matrix (heatmap of which tokens follow which)
    transition_matrix = np.zeros((256, 256))
    for i in range(len(tokens) - 1):
        transition_matrix[int(tokens[i]), int(tokens[i+1])] += 1
    
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums > 0)
    
    im = axes[2].imshow(transition_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
    axes[2].set_title('Token Transition Matrix (P(token_j | token_i))', fontsize=12)
    axes[2].set_xlabel('Next Token')
    axes[2].set_ylabel('Current Token')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {save_path}")
    plt.close()


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VQ MODEL VALIDATION SUITE")
    print("=" * 60)
    
    # Paths
    checkpoint_path = 'checkpoints/vq_model_best.pth'
    embeddings_paths = {
        'Pong': 'data/embeddings_ALE_Pong-v5_cnn.npy',
        'Breakout': 'data/embeddings_ALE_Breakout-v5_cnn.npy',
        'MsPacman': 'data/embeddings_ALE_MsPacman-v5_cnn.npy',
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings_dict = {}
    all_embeddings = []
    for game, path in embeddings_paths.items():
        emb = np.load(path).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = emb / norms
        embeddings_dict[game] = emb
        all_embeddings.append(emb)
        print(f"  Loaded {game}: {emb.shape}")
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Total embeddings: {all_embeddings.shape}")
    
    # Run tests
    results = {}
    
    results['reconstruction'] = test_reconstruction_quality(model, all_embeddings)
    results['temporal'] = test_temporal_consistency(model, embeddings_dict['Pong'])
    results['codebook'] = test_codebook_statistics(model, all_embeddings)
    results['multi_game'] = test_multi_game_separation(model, embeddings_dict)
    
    # Visualize
    print("\nGenerating visualizations...")
    device_cpu = next(model.parameters()).device
    x = torch.from_numpy(embeddings_dict['Pong'][:10000]).float().to(device_cpu)
    x = x.unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        tokens_list = model.encode(x)
    # Visualize Layer 0 (coarse/shared patterns)
    visualize_token_patterns(tokens_list[0].squeeze().cpu().numpy())
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY (HRVQ)")
    print("=" * 60)
    print(f"✓ Reconstruction quality:  {results['reconstruction']['cosine_similarity_mean']:.4f}")
    print(f"✓ Temporal consistency:    {results['temporal']['temporal_smoothness']:.4f}")
    print(f"✓ Layer 0 codebook usage:  {results['codebook']['layers'][0]['num_used_codes']}/256")
    print(f"✓ Layer 0 perplexity:      {results['codebook']['layers'][0]['perplexity']:.2f}")
    print(f"✓ Layer 1 codebook usage:  {results['codebook']['layers'][1]['num_used_codes']}/256")
    print(f"✓ Layer 2 codebook usage:  {results['codebook']['layers'][2]['num_used_codes']}/256")
    print(f"✓ Shared tokens (Layer 0): {results['multi_game']['shared_tokens']}/256")
    print("=" * 60)
    
    # Save results
    import json
    os.makedirs('results', exist_ok=True)
    with open('results/vq_validation_results.json', 'w') as f:
        # Remove large arrays for JSON
        results_clean = {
            'reconstruction': results['reconstruction'],
            'temporal': results['temporal'],
            'codebook': {k: v for k, v in results['codebook'].items() if k != 'token_counts'},
            'multi_game': {k: v for k, v in results['multi_game'].items() if k != 'game_tokens'},
        }
        json.dump(results_clean, f, indent=2)
    print("\n✓ Saved detailed results to results/vq_validation_results.json")


if __name__ == '__main__':
    main()
