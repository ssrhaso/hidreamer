"""
VALIDATE DINOV2 ENCODER FOR PONG
Tests if frozen DINOv2 embeddings capture game semantics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import seaborn as sns


def load_data(replay_path, embeddings_path, num_samples=500):
    """Load frames and their DINOv2 embeddings"""
    replay = np.load(replay_path)
    
    # Get frames
    if 'states' in replay:
        frames = replay['states']
    elif 'observations' in replay:
        frames = replay['observations']
    else:
        frames = replay['obs']
    
    # Load embeddings
    embeddings = np.load(embeddings_path).astype(np.float32)
    
    # Random sample
    indices = np.random.choice(len(frames), num_samples, replace=False)
    return frames[indices], embeddings[indices], indices


def extract_game_state_labels(frames):
    """
    Extract semantic game state features from Pong frames
    Returns labels for: ball position, paddle positions
    """
    labels = {
        'ball_x': [],
        'ball_y': [],
        'left_paddle_y': [],
        'right_paddle_y': [],
    }
    
    for frame in frames:
        # Take last frame if stacked (C, H, W)
        if frame.ndim == 3 and frame.shape[0] in [3, 4]:
            img = frame[-1]  # Most recent frame
        else:
            img = frame
        
        # Find ball (brightest small region)
        threshold = img.max() * 0.5
        ball_pixels = np.where(img > threshold)
        
        if len(ball_pixels[0]) > 0:
            ball_y = ball_pixels[0].mean()
            ball_x = ball_pixels[1].mean()
        else:
            ball_y, ball_x = -1, -1  # No ball found
        
        # Find paddles (bright vertical edges on left/right)
        left_col = img[:, :10].max(axis=1)  # Left 10 pixels
        right_col = img[:, -10:].max(axis=1)  # Right 10 pixels
        
        left_paddle_pixels = np.where(left_col > threshold)[0]
        right_paddle_pixels = np.where(right_col > threshold)[0]
        
        left_paddle_y = left_paddle_pixels.mean() if len(left_paddle_pixels) > 0 else -1
        right_paddle_y = right_paddle_pixels.mean() if len(right_paddle_pixels) > 0 else -1
        
        labels['ball_x'].append(ball_x)
        labels['ball_y'].append(ball_y)
        labels['left_paddle_y'].append(left_paddle_y)
        labels['right_paddle_y'].append(right_paddle_y)
    
    # Convert to numpy
    for key in labels:
        labels[key] = np.array(labels[key])
    
    return labels


def test_embedding_semantic_correlation(embeddings, labels):
    """
    Test if embedding distances correlate with semantic differences
    """
    print("\n" + "="*60)
    print("TEST 1: EMBEDDING ↔ GAME STATE CORRELATION")
    print("="*60)
    
    # Compute pairwise embedding distances
    embedding_distances = 1 - cosine_similarity(embeddings)
    
    results = {}
    
    for feature_name, feature_values in labels.items():
        # Skip if feature not detected in enough frames
        valid_mask = feature_values >= 0
        if valid_mask.sum() < 50:
            print(f"\n  {feature_name}: Not enough valid detections ({valid_mask.sum()}/500)")
            continue
        
        # Compute pairwise feature differences
        feature_diff = np.abs(feature_values[:, None] - feature_values[None, :])
        
        # Flatten upper triangle (avoid double-counting)
        triu_indices = np.triu_indices(len(embeddings), k=1)
        emb_dist_flat = embedding_distances[triu_indices]
        feat_diff_flat = feature_diff[triu_indices]
        
        # Remove pairs where either frame had invalid detection
        valid_pairs = (feature_values[triu_indices[0]] >= 0) & (feature_values[triu_indices[1]] >= 0)
        emb_dist_valid = emb_dist_flat[valid_pairs]
        feat_diff_valid = feat_diff_flat[valid_pairs]
        
        # Spearman correlation (monotonic relationship)
        correlation, p_value = spearmanr(emb_dist_valid, feat_diff_valid)
        
        results[feature_name] = correlation
        
        # Interpret
        status = "✓" if correlation > 0.3 else "⚠️" if correlation > 0.15 else "✗"
        print(f"\n{status} {feature_name}:")
        print(f"   Correlation: {correlation:.3f} (p={p_value:.4f})")
        print(f"   Valid pairs: {len(emb_dist_valid)}")
        
        if correlation > 0.3:
            print(f"   → GOOD: Embeddings capture {feature_name} well")
        elif correlation > 0.15:
            print(f"   → WEAK: Embeddings partially capture {feature_name}")
        else:
            print(f"   → BAD: Embeddings don't capture {feature_name}")
    
    # Overall score
    avg_corr = np.mean([v for v in results.values() if not np.isnan(v)])
    print(f"\n{'='*60}")
    print(f"OVERALL SEMANTIC CORRELATION: {avg_corr:.3f}")
    
    if avg_corr > 0.4:
        print("✓ VERDICT: DINOv2 captures game state well - PROCEED")
    elif avg_corr > 0.25:
        print("⚠️  VERDICT: DINOv2 partially captures game state - CONSIDER ADAPTER")
    else:
        print("✗ VERDICT: DINOv2 doesn't capture game state - USE TRAINABLE CNN")
    print("="*60)
    
    return results


def test_temporal_consistency(embeddings, indices):
    """
    Test if consecutive frames have similar embeddings
    """
    print("\n" + "="*60)
    print("TEST 2: TEMPORAL CONSISTENCY")
    print("="*60)
    
    # Find pairs of consecutive frames in our sample
    consecutive_pairs = []
    non_consecutive_pairs = []
    
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            if i >= j:
                continue
            
            if abs(idx_i - idx_j) == 1:
                consecutive_pairs.append((i, j))
            elif abs(idx_i - idx_j) > 100:  # Far apart
                non_consecutive_pairs.append((i, j))
    
    # Compute distances
    if len(consecutive_pairs) > 0:
        consec_dists = [
            np.linalg.norm(embeddings[i] - embeddings[j])
            for i, j in consecutive_pairs
        ]
        
        non_consec_dists = [
            np.linalg.norm(embeddings[i] - embeddings[j])
            for i, j in non_consecutive_pairs[:len(consecutive_pairs)]  # Match sample size
        ]
        
        print(f"\nConsecutive frames (n={len(consecutive_pairs)}):")
        print(f"  Mean distance: {np.mean(consec_dists):.4f}")
        print(f"  Std distance:  {np.std(consec_dists):.4f}")
        
        print(f"\nNon-consecutive frames (n={len(non_consec_dists)}):")
        print(f"  Mean distance: {np.mean(non_consec_dists):.4f}")
        print(f"  Std distance:  {np.std(non_consec_dists):.4f}")
        
        ratio = np.mean(consec_dists) / np.mean(non_consec_dists)
        print(f"\nRatio (consecutive/non-consecutive): {ratio:.3f}")
        
        if ratio < 0.7:
            print("✓ GOOD: Consecutive frames are more similar")
        elif ratio < 0.9:
            print("  WEAK: Some temporal consistency")
        else:
            print("✗ BAD: No temporal consistency")
    else:
        print("  Not enough consecutive frames in sample")

def visualize_embedding_space(embeddings, labels, output_path='dinov2_validation.png'):
    """
    Visualize embeddings colored by game state features
    """
    print("\n" + "="*60)
    print("TEST 3: EMBEDDING SPACE VISUALIZATION")
    print("="*60)
    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    features = ['ball_x', 'ball_y', 'left_paddle_y', 'right_paddle_y']
    titles = ['Ball X Position', 'Ball Y Position', 'Left Paddle Y', 'Right Paddle Y']
    
    for ax, feature, title in zip(axes.flat, features, titles):
        valid_mask = labels[feature] >= 0
        
        scatter = ax.scatter(
            embeddings_2d[valid_mask, 0],
            embeddings_2d[valid_mask, 1],
            c=labels[feature][valid_mask],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=ax, label='Pixel Position')
    
    plt.suptitle('DINOv2 Embedding Space Colored by Game State', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    print("  → If you see clear gradients/clusters, embeddings capture semantics")


def test_clustering_quality(embeddings, labels):
    """
    Test if semantically similar states cluster together
    """
    print("\n" + "="*60)
    print("TEST 4: CLUSTERING QUALITY")
    print("="*60)
    
    # Define semantic bins (e.g., "ball in left third", "ball in middle", etc.)
    ball_x = labels['ball_x']
    valid = ball_x >= 0
    
    if valid.sum() < 50:
        print("⚠️  Not enough valid ball detections")
        return
    
    # Divide into 3 zones: left, center, right
    ball_x_valid = ball_x[valid]
    embeddings_valid = embeddings[valid]
    
    zones = np.digitize(ball_x_valid, bins=[ball_x_valid.min() + i*(ball_x_valid.max()-ball_x_valid.min())/3 for i in range(1, 3)])
    
    # Within-zone vs between-zone distances
    within_dists = []
    between_dists = []
    
    for i in range(len(zones)):
        for j in range(i+1, len(zones)):
            dist = np.linalg.norm(embeddings_valid[i] - embeddings_valid[j])
            
            if zones[i] == zones[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)
    
    if len(within_dists) > 0 and len(between_dists) > 0:
        print(f"\nWithin-zone distance: {np.mean(within_dists):.4f}")
        print(f"Between-zone distance: {np.mean(between_dists):.4f}")
        
        separation = np.mean(between_dists) / np.mean(within_dists)
        print(f"Separation ratio: {separation:.3f}")
        
        if separation > 1.2:
            print("✓ GOOD: Semantic clusters are well-separated")
        elif separation > 1.05:
            print("⚠️  WEAK: Some cluster separation")
        else:
            print("✗ BAD: No semantic clustering")


def main():
    # Paths
    REPLAY_PATH = "data/replay_buffer_ALE_Pong-v5.npz"
    EMBEDDINGS_PATH = "data/embeddings_ALE_Pong-v5_dinov2_base.npy"
    
    print("="*60)
    print("DINOV2 ENCODER VALIDATION FOR PONG")
    print("="*60)
    print("\nLoading data...")
    
    frames, embeddings, indices = load_data(REPLAY_PATH, EMBEDDINGS_PATH, num_samples=500)
    print(f"✓ Loaded {len(frames)} frames")
    print(f"  Frame shape: {frames.shape}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Extract ground truth game state labels
    print("\nExtracting game state labels from frames...")
    labels = extract_game_state_labels(frames)
    
    # Run tests
    test_embedding_semantic_correlation(embeddings, labels)
    test_temporal_consistency(embeddings, indices)
    visualize_embedding_space(embeddings, labels)
    test_clustering_quality(embeddings, labels)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check dinov2_validation.png for visual clusters")
    print("2. If correlations > 0.3: proceed with world model")
    print("3. If correlations < 0.3: add trainable adapter layer")
    print("="*60)


if __name__ == "__main__":
    main()
