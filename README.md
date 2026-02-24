[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

# Hi-Dreamer - Hierarchical World Model

A lightweight world model pipeline for sample-efficient Atari control on a single GPU. Combines a trained CNN encoder, 3-layer Hierarchical Residual Vector Quantizer, and a hierarchical transformer with each component validated independently before integration. Research engineering project exploring whether discrete hierarchical representations enable cross-game transfer in model-based RL.

## Architecture

```
Atari Frames (4×84×84) → CNN Encoder (1.88M params) → 384D Embeddings
    → 3-Layer HRVQ (256 codes/layer) → Discrete Tokens
    → Hierarchical Transformer World Model (11M params)
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

- **Unified Embedding Space:** A single encoder learned consistent temporal features across 3 distinct games (Norm: 1.0, Temporal Ratio: 0.22 avg).
- **Shared Vocabulary:** Trained one VQ codebook (256 codes) on 300k combined frames — 100% codebook usage and high perplexity (~240) on the joint dataset within 2 epochs.

**Hierarchical Residual VQ (HRVQ):**

Implemented and trained a 3-layer Hierarchical Residual VQ inspired by SoundStream (2021) and HiTVideo (2025). Each layer quantizes the residual of the previous, with layer-specific commitment costs `[0.05, 0.25, 0.60]` forcing Layer 0 to capture coarse shared representations first. 100% codebook utilization across all layers (perplexities: 240/229/228). Cross-game analysis of Layer 0 tokens shows shared codes for backgrounds and motion primitives, with game-specific mechanics (paddle physics, ghost AI, brick patterns) separated into higher layers — consistent with the hierarchical hypothesis that Atari games share structure at the right abstraction level.

### World Model - Training Complete

**Hierarchical Transformer - 40 Epochs, T4 GPU**

Built and trained a discrete token world model operating on HRVQ token sequences (6-layer transformer, 384D, 6 heads, 1536 FFN, ~11M params). Predicts next-step tokens autoregressively across the 3-level hierarchy. Architecture inspired by STORM (Zhang et al., 2023) with hierarchical masking novelty.

**Architecture Components:**

- **Token embedding:** Interleaves 3 HRVQ layers + action tokens into `(B, T*4, 384)` sequences with learned level embeddings and positional encodings
- **Hierarchical causal mask:** Custom attention mask enforcing both temporal causality and semantic hierarchy. Beyond standard causal masking, blocks fine-grained layers (L1, L2) from attending to previous timesteps' detail tokens while allowing all layers to see past coarse physics tokens (L0). Within each timestep, enforces hierarchical dependencies (L2→L1→L0→Action) so predictions build coarse-to-fine
- **6 transformer blocks:** Standard pre-norm blocks (MHSA + FFN + residuals + LayerNorm) with cached mask for efficiency
- **Prediction heads:** Layer-specific linear heads predict next HRVQ codes (L0: action→L0_next, L1: L0→L1_same, L2: L1→L2_same)
- **Hierarchical loss:** Weighted cross-entropy `[1.0, 0.5, 0.1]` emphasizing coarse dynamics, with per-layer accuracy tracking

**Dataset Pipeline (`world_model_dataset.py`):**

- Loads 300K timesteps (3 games × 100K) of pre-extracted HRVQ tokens and actions
- Respects episode boundaries using vectorized cumsum-based window validation (0 violations across 246K valid 64-step sequences)
- Multi-game batching: Pong 38%, Breakout 27%, MsPacman 35% — forces Layer 0 to learn shared physics
- Train/val split: 234K/12K sequences, DataLoader with persistent workers for Windows compatibility

**Training Config (`configs/worldmodel.yaml`):**

- Optimizer: AdamW (lr=3e-4, weight_decay=0.1, betas=[0.9, 0.95])
- Scheduler: 1000-step linear warmup → cosine annealing
- Batch: 32 sequences/step × 2 accumulation steps = effective batch 64
- Mixed precision: float16 AMP with GradScaler (T4-compatible)
- 40 epochs, gradient clipping at 1.0

**Training Results:**

| Epoch | Val Loss | L0 Acc | L1 Acc | L2 Acc |
| ----- | -------- | ------ | ------ | ------ |
| 1     | 0.739    | 93.8%  | 78.2%  | 53.8%  |
| 5     | 0.210    | 98.1%  | 91.3%  | 79.4%  |
| 10    | 0.073    | 99.5%  | 97.8%  | 95.9%  |
| 17    | 0.060    | 99.6%  | 98.1%  | 97.6%  |
| 22    | 0.055    | 99.6%  | 98.2%  | 98.0%  |
| 30    | 0.052    | 99.7%  | 98.2%  | 98.3%  |
| 35    | 0.050    | 99.7%  | 98.3%  | 98.4%  |
| 40    | 0.049    | 99.7%  | 98.3%  | 98.5%  |

- Hardware: NVIDIA T4 GPU (Google Colab free tier), ~10 min/epoch, ~7 hours total
- Experiment tracking: WandB (`chocolate-dream-3`)
- Validation loss consistently lower than training loss throughout — healthy dropout signature, no overfitting

**Key Findings:**

- **Fast convergence:** Near-convergence by epoch 22. Val loss dropped 93% (0.739 → 0.055), indicating the architecture efficiently extracted all learnable structure from 300K frames.
- **No overfitting:** Val loss remained below train loss across all 40 epochs — confirmed healthy generalization.
- **L1/L2 inversion:** From epoch 31 onward, L2 accuracy (rendering) consistently exceeded L1 accuracy (mechanics) — 98.5% vs 98.3%. Game mechanics are genuinely harder to predict than rendering details: once physics (L0) is known, sprite positions follow near-deterministically, while ghost AI decisions and ball spin introduce irreducible stochasticity at the mechanics layer. To be explored further in ablation studies.
- **Entropy floor:** The remaining ~0.3% L0 prediction error represents irreducible game stochasticity (episode resets, pseudo-random AI). 100% accuracy is theoretically impossible without data leakage.
- **Codebook health:** All three HRVQ layers maintained 100% codebook utilization (256/256 codes) throughout training.

**Validation Suite:**

- `validate_dataset.py`: End-to-end test (shapes, dtypes, boundary integrity, forward pass + loss)
- `validate_world_model.py`: Model architecture test (output shapes, gradients, NaN checks)
- `validate_mask.py`: Hierarchical attention pattern verification

### Policy - In Progress

**Actor-Critic in Latent Token Space - Pong-v5 (proof of concept)**

Training a TWISTER -style actor-critic that operates entirely on the frozen world model's HRVQ token representations. The world model is frozen —-only the policy networks are trained.

**Feature Representation:**

```
feat = concat(L0_embed, L1_embed, L2_embed)  →  (B, 1152)
```

**Architecture:**

- Actor: MLP (1152→512→512→num_actions), REINFORCE with λ-returns
- Critic: MLP (1152→512→512→1), log-likelihood loss with EMA slow target
- Reward predictor + Continue predictor: auxiliary MLPs for imagination rollouts
- Imagination horizon: H=15 steps, γ=0.997, λ=0.95
- Adapted from TWISTER (Burchi & Timofte, ICLR 2025) / DreamerV3 (Hafner et al., 2023)

**Status:** PPO + actor-critic implementation in progress. Evaluation rollout ablations planned on Pong-v5 first, then pivoting based on results.

## Lessons Learned

- Initially projected DINOv2 embeddings from 384→128D via MLP, but this introduced information loss. Removing the projection and feeding 384D directly to VQ improved codebook utilization from 77/256 to 153/256.
- Frozen DINOv2 embeddings failed on Atari: only 0.08 correlation to game states (paddle position: 0.05, ball position: 0.02-0.15). While temporal consistency (0.53) showed DINOv2 detects motion, it couldn't capture position — the foundation model's natural image features don't transfer to synthetic game graphics. This motivated pivoting to a trained CNN, confirming task-specific encoders outperform general-purpose foundations under significant domain shift.
- Aggressive per-layer commitment cost scheduling was critical for HRVQ. Without it, all layers learned redundant representations instead of a coarse-to-fine hierarchy.
- Validation loss consistently lower than training loss is not overfitting — it is the correct behaviour when dropout is active during training and disabled at validation.
- The L1/L2 accuracy inversion (rendering easier to predict than mechanics) was unexpected. It suggests the HRVQ layer separation is working correctly: L1 captures genuinely stochastic game dynamics, while L2 rendering is near-deterministic given L0 physics context.

## References

- IRIS (Micheli et al., ICLR 2023) — discrete token world models for Atari
- STORM (Zhang et al., 2023) — transformer-based world models, architecture inspiration
- TWISTER (Burchi & Timofte, ICLR 2025) — AC-CPC + DreamerV3 actor-critic, policy reference
- DreamerV3 (Hafner et al., 2023) — actor-critic in latent space
- DIAMOND (Alonso et al., NeurIPS 2024) — diffusion world models, comparison baseline
- DART (Agarwal et al., ICML 2024) — discrete token policy learning
- SoundStream (Zeghidour et al., NeurIPS 2021) — residual VQ, HRVQ inspiration
- HiTVideo (2025) — hierarchical residual VQ for video
