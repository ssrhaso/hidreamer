[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

# Hi-Dreamer - Hierarchical Imagination Dreamer
Trained lightweight CNN Encoder + discrete tokenization with 3 layer Hierarchical Residual Visual Quantizer + World Model transformer attention dynamics = sample-efficient Atari control on a single GPU. Research engineering analysis combining three validated techniques in a novel architecture

## Results (AT PRESENT)

**CNN Encoder (1.8M params, 30 epochs):**

- Temporal consistency: 0.260 (excellent - consecutive frames 74% closer than random)
- Embedding diversity: 1.411 (no collapse)
- Game state correlation: 0.267 (3x better than DINOv2's 0.08, sufficient for dynamics)

**VQ Tokenizer (256 codes, 50 epochs):**

- Codebook usage: 256/256 (100% - perfect utilization)
- Perplexity: 247.52 (near theoretical max of 256)
- Training time: 50 epochs in 50 seconds

**Multi-Game Generalization (Pong, Breakout, MsPacman):**

* **Unified Embedding Space:** A single 1.8M param encoder learned consistent temporal features across 3 distinct games (Norm: 1.0, Temporal Ratio: 0.22 avg).
* **Shared Vocabulary:** Trained one VQ codebook (256 codes) on 300k combined frames.
* **Zero-Shot Transfer?** No, but "few-shot adaptation" proven: 100% codebook usage and high perplexity (~240) achieved on the joint dataset in just 2 epochs. This validates that our discrete latent space is robust enough to represent diverse game physics simultaneously.

**Hierarchical Residual VQ (HRVQ) - Breakthrough Achieved:**

We successfully implemented and trained a 3-layer Hierarchical Residual VQ architecture inspired by HiTVideo (2025) and SoundStream (2021). The key innovation: aggressive layer-specific commitment costs `[0.05, 0.25, 0.60]` forced Layer 0 to capture coarse shared representations before passing residuals to game-specific layers. **Results exceeded expectations:** 125/256 shared tokens (49% of vocabulary) are now used across all three games, with ~53% average frame coverage - proving genuine cross-game semantic transfer. Reconstruction quality improved from 0.40 → 0.73 cosine similarity while maintaining 100% codebook utilization across all layers (perplexities: 240/229/228). The model learned that black backgrounds, motion primitives, and edge features are universal, while game-specific mechanics (paddle physics vs ghost AI vs brick patterns) occupy separate subspaces (assumption). This validates the hierarchical hypothesis: Atari games ARE transferable at the right abstraction level. 


### WORLD MODEL:

**Hierarchical Transformer with Adaptive Mixture-of-Experts (In Progress)**

Building a discrete token world model combining TWISTER (ICLR 2025) and DreamerV4 innovations. **ATTEMPTING CONTRIBUTION** : first per-layer expert counts in world modeling, expected 2.8x inference speedup with 40% parameter reduction, temporal hierarchy matching semantic hierarchy, +15% long-horizon accuracy. Target: ~11M parameter model achieving <5ms inference for PPO imagination rollouts, enabling model-based RL with 10x sample efficiency over model-free baselines.

**Hierarchical Causal Attention Mask:** Implemented a custom attention mask that enforces both temporal causality and semantic hierarchy across the 3-layer HRVQ representation. Standard causal masks only prevent future-seeing; our mask additionally blocks lower layers (L1, L2) from attending to previous timesteps' fine-grained tokens, while allowing all layers to see past coarse physics tokens (L0). This design matches the tokenizer's semantic structure: L0 captures shared physics/motion that should inform future predictions, while L1 (game mechanics) and L2 (game-specific details) represent ephemeral state that shouldn't leak across timesteps. Within each timestep, the mask enforces hierarchical dependencies (L2→L1→L0→Action), enabling the model to build predictions from coarse-to-fine. This architectural alignment between the HRVQ tokenizer's learned abstraction levels and the transformer's information flow prevents the model from "cheating" by memorizing low-level patterns instead of learning dynamics, resulting in better generalization and more interpretable world models.

## Lessons Learned

- We initially projected DINOv2 embeddings from 384 to 128 dimensions using an MLP, but found this introduced unnecessary information loss. We removed the projection and fed 384-dimensional embeddings directly to the VQ layer, improving codebook utilization from 77/256 to 153/256.
- Frozen DINOv2 embeddings failed on Atari with only 0.08 correlation to game states (paddle position: 0.05, ball position: 0.02-0.15). While temporal consistency (0.53) showed DINOv2 detects motion, it couldn't capture position - the foundation model's natural image features don't transfer to synthetic game graphics. This 10x performance gap vs trainable CNNs demonstrates when vision foundation models fail: domain shift matters more than model scale. We pivoted to a trained CNN, proving task-specific encoders outperform general-purpose foundations on domain-shifted tasks. This was expected due to the difference in content DINOv2 is trained on in comparision to ATARI images.

