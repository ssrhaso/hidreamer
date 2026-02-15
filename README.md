[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

# Hi-Dreamer - Hierarchical World Model 

A lightweight world model pipeline for sample-efficient Atari control on a single GPU. Combines a trained CNN encoder, 3-layer Hierarchical Residual Vector Quantizer, and a hierarchical transformer — each component validated independently before integration. Research engineering project exploring whether discrete hierarchical representations enable cross-game transfer in model-based RL.

## Architecture

```
Atari Frames (4×84×84) → CNN Encoder (1.88M params) → 384D Embeddings
    → 3-Layer HRVQ (256 codes/layer) → Discrete Tokens
    → Hierarchical Transformer World Model (in progress)
    → Imagination Rollouts → Policy Optimization
```

## Results

**CNN Encoder (1.88M params, 30 epochs):**

- Temporal consistency: 0.260 (consecutive frames 74% closer than random)
- Embedding diversity: 1.411 (no mode collapse)
- Game state correlation: 0.267 (3x better than DINOv2's 0.08)
- Trained with temporal contrastive learning (InfoNCE) on stacked 4-frame inputs

**VQ Tokenizer (256 codes, 50 epochs):**

- Codebook usage: 256/256 (100% utilization, no dead codes)
- Perplexity: 240.25 (near theoretical max of 256)
- Temporal smoothness: 0.811 (tokens are coherent across frames - 81% persist frame-to-frame)
- Training time: 50 epochs in ~50 seconds

**Multi-Game Generalization (Pong, Breakout, MsPacman):**

* **Unified Embedding Space:** A single encoder learned consistent temporal features across 3 distinct games (Norm: 1.0, Temporal Ratio: 0.22 avg).
* **Shared Vocabulary:** Trained one VQ codebook (256 codes) on 300k combined frames — 100% codebook usage and high perplexity (~240) on the joint dataset within 2 epochs.

**Hierarchical Residual VQ (HRVQ):**

Implemented and trained a 3-layer Hierarchical Residual VQ inspired by SoundStream (2021) and HiTVideo (2025). Each layer quantizes the residual of the previous, with layer-specific commitment costs `[0.05, 0.25, 0.60]` forcing Layer 0 to capture coarse shared representations first. 100% codebook utilization across all layers (perplexities: 240/229/228). Cross-game analysis of Layer 0 tokens shows shared codes for backgrounds and motion primitives, with game-specific mechanics (paddle physics, ghost AI, brick patterns) separated into higher layers — consistent with the hierarchical hypothesis that Atari games share structure at the right abstraction level.

### World Model (In Progress)

**Hierarchical Transformer — Partially Implemented**

Building a discrete token world model that operates on HRVQ token sequences, drawing on ideas from TWISTER (ICLR 2025) and DreamerV3. Target architecture: 6-layer transformer (384D, 6 heads, 1536 FFN) predicting next-step tokens autoregressively across the 3-level hierarchy + actions.

**Completed so far:**

- **Token embedding layer:** Interleaves 3 HRVQ layers + action tokens into `(B, T*4, 384)` sequences with learned level embeddings and positional encodings
- **Hierarchical causal attention mask:** Custom mask enforcing both temporal causality and semantic hierarchy. Beyond standard causal masking, it blocks fine-grained layers (L1, L2) from attending to previous timesteps' detail tokens while allowing all layers to see past coarse physics tokens (L0). Within each timestep, the mask enforces hierarchical dependencies (L2→L1→L0→Action) so predictions build coarse-to-fine. This matches the HRVQ tokenizer's learned abstraction levels to the transformer's information flow
- **Transformer block:** Standard pre-norm transformer block (MHSA + FFN + residuals + LayerNorm)
- **Config + data pipeline:** Full YAML config with training hyperparameters, token data pre-extracted for all 3 games × 3 layers

Recent additions include the full `HierarchicalWorldModel` scaffold with stacked blocks, layer-specific token prediction heads (L0/L1/L2), and a cached hierarchical mask to avoid recomputation on long sequences. A weighted hierarchical loss is also wired to emphasize coarse dynamics over fine detail, aligning the training signal with the HRVQ abstraction ladder.

**Remaining:** Assembling blocks into full model with prediction heads (token/reward/done), loss function, and training loop.

## Lessons Learned

- Initially projected DINOv2 embeddings from 384→128D via MLP, but this introduced information loss. Removing the projection and feeding 384D directly to VQ improved codebook utilization from 77/256 to 153/256.
- Frozen DINOv2 embeddings failed on Atari: only 0.08 correlation to game states (paddle position: 0.05, ball position: 0.02-0.15). While temporal consistency (0.53) showed DINOv2 detects motion, it couldn't capture position — the foundation model's natural image features don't transfer to synthetic game graphics. This motivated pivoting to a trained CNN, confirming task-specific encoders outperform general-purpose foundations under significant domain shift.
- Aggressive per-layer commitment cost scheduling was critical for HRVQ. Without it, all layers learned redundant representations instead of a coarse-to-fine hierarchy.
