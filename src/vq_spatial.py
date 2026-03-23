"""
SPATIAL HRVQ TOKENIZER - PER-PATCH VQ FOR DISCRETE SPATIAL TOKEN GRIDS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vq import VQVAE  # REUSE EXISTING EMA-BASED VQ LAYER


NUM_L0_PATCHES = 4
NUM_L1_PATCHES = 16
NUM_L2_PATCHES = 16


# GRADIENT-BASED VQ
class GradientVQ(nn.Module):
    """ GRADIENT-BASED VQ WITH STRAIGHT-THROUGH ESTIMATOR """

    def __init__(self, num_codes: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codebook_entries = num_codes   # ALIAS KEPT FOR COMPATIBILITY
        self.num_codes = num_codes
        self.codebook_dim = dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, dim)
        # UNIFORM INIT SCALED BY 1/NUM_CODES
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])           # (N, D)

        # PAIRWISE DISTANCES TO ALL CODEBOOK ENTRIES
        dists = torch.cdist(flat, self.codebook.weight)  # (N, num_codes)
        indices_flat = dists.argmin(dim=-1)              # (N,)
        quantized_flat = self.codebook(indices_flat)     # (N, D)

        # CODEBOOK LOSS + COMMITMENT LOSS
        codebook_loss   = F.mse_loss(quantized_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, quantized_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # STRAIGHT-THROUGH ESTIMATOR
        quantized_st = flat + (quantized_flat - flat).detach()

        return (
            quantized_st.reshape(orig_shape),
            vq_loss,
            indices_flat.reshape(orig_shape[:-1]),
        )


class SpatialHRVQTokenizer(nn.Module):
    """ PER-LEVEL SPATIAL VQ TOKENIZER WITH DEAD-CODE REVIVAL """

    def __init__(
        self,
        d_model: int = 384,
        num_codes_l0: int = 16,
        num_codes_l1: int = 64,
        num_codes_l2: int = 64,
        commitment_costs: list = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        revival_interval: int = 100,
        revival_noise: float = 1e-3,
        use_gradient_vq: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes_l0 = num_codes_l0
        self.num_codes_l1 = num_codes_l1
        self.num_codes_l2 = num_codes_l2
        self.num_codes = max(num_codes_l0, num_codes_l1, num_codes_l2)
        self.revival_interval = revival_interval
        self.revival_noise = revival_noise
        self.use_gradient_vq = use_gradient_vq

        if commitment_costs is None:
            commitment_costs = [0.05, 0.25, 0.60]

        # SEPARATE VQ LAYER PER SPATIAL LEVEL
        if use_gradient_vq:
            self.vq_l0 = GradientVQ(num_codes_l0, d_model, commitment_costs[0])
            self.vq_l1 = GradientVQ(num_codes_l1, d_model, commitment_costs[1])
            self.vq_l2 = GradientVQ(num_codes_l2, d_model, commitment_costs[2])
        else:
            self.vq_l0 = VQVAE(
                num_codebook_entries=num_codes_l0,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[0],
                decay=decay,
                epsilon=epsilon,
                layer_idx=0,
            )
            self.vq_l1 = VQVAE(
                num_codebook_entries=num_codes_l1,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[1],
                decay=decay,
                epsilon=epsilon,
                layer_idx=1,
            )
            self.vq_l2 = VQVAE(
                num_codebook_entries=num_codes_l2,
                codebook_dim=d_model,
                commitment_cost=commitment_costs[2],
                decay=decay,
                epsilon=epsilon,
                layer_idx=2,
            )

        # DEAD-CODE TRACKING BUFFERS
        self.register_buffer('_usage_l0', torch.zeros(num_codes_l0, dtype=torch.long))
        self.register_buffer('_usage_l1', torch.zeros(num_codes_l1, dtype=torch.long))
        self.register_buffer('_usage_l2', torch.zeros(num_codes_l2, dtype=torch.long))

        # STEP COUNTER
        self._train_step = 0

    @torch.no_grad()
    def _revive_dead_codes(
        self,
        level_name: str,
        vq_layer: VQVAE,
        usage_buf: torch.Tensor,
        encoder_flat: torch.Tensor,
    ) -> int:
        """ REPLACE ZERO-USAGE CODEBOOK ENTRIES WITH SAMPLED ENCODER OUTPUTS """
        dead_mask = (usage_buf == 0)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0

        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        M = encoder_flat.size(0)

        # SAMPLE N_DEAD ENCODER OUTPUTS WITH REPLACEMENT
        sample_idx = torch.randint(0, M, (n_dead,), device=encoder_flat.device)
        new_codes = encoder_flat[sample_idx].clone()

        # ADD NOISE THEN RENORMALIZE
        new_codes = new_codes + self.revival_noise * torch.randn_like(new_codes)
        new_codes = F.normalize(new_codes, p=2, dim=-1)

        # UPDATE CODEBOOK WEIGHTS
        vq_layer.codebook.weight.data[dead_idx] = new_codes

        # EMA-SPECIFIC STATE - ONLY PRESENT ON VQVAE
        active_mask = ~dead_mask
        if hasattr(vq_layer, 'ema_weight'):
            vq_layer.ema_weight[dead_idx] = new_codes
            if active_mask.any():
                mean_active = vq_layer.ema_cluster_size[active_mask].mean()
            else:
                mean_active = torch.tensor(1.0, device=encoder_flat.device)
            vq_layer.ema_cluster_size[dead_idx] = mean_active

        n_active = int(active_mask.sum()) if active_mask.any() else 0
        print(f"  [Revival] Revived {n_dead:3d} dead codes at level {level_name} "
              f"(active before: {n_active}/{vq_layer.num_codebook_entries})")
        return n_dead

    def _accumulate_usage(
        self,
        idx0: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
    ):
        for idx, buf in [(idx0, self._usage_l0),
                         (idx1, self._usage_l1),
                         (idx2, self._usage_l2)]:
            flat = idx.reshape(-1).long()
            ones = torch.ones(flat.size(0), dtype=torch.long, device=flat.device)
            buf.scatter_add_(0, flat, ones)

    def maybe_revive(
        self,
        l0_feats: torch.Tensor,
        l1_feats: torch.Tensor,
        l2_feats: torch.Tensor,
    ) -> dict:
        self._train_step += 1
        counts = {'l0': 0, 'l1': 0, 'l2': 0}

        if self._train_step % self.revival_interval != 0:
            return counts

        # FLAT ENCODER OUTPUTS FOR SAMPLING CANDIDATES
        flat_l0 = l0_feats.detach().reshape(-1, self.d_model)
        flat_l1 = l1_feats.detach().reshape(-1, self.d_model)
        flat_l2 = l2_feats.detach().reshape(-1, self.d_model)

        counts['l0'] = self._revive_dead_codes('l0', self.vq_l0, self._usage_l0, flat_l0)
        counts['l1'] = self._revive_dead_codes('l1', self.vq_l1, self._usage_l1, flat_l1)
        counts['l2'] = self._revive_dead_codes('l2', self.vq_l2, self._usage_l2, flat_l2)

        # RESET USAGE COUNTERS FOR NEXT WINDOW
        self._usage_l0.zero_()
        self._usage_l1.zero_()
        self._usage_l2.zero_()

        return counts

    def forward(self, spatial_feats: dict):
        """ QUANTIZE SPATIAL FEATURES AND RETURN TOKEN INDICES, LOSS, AND QUANTIZED EMBEDDINGS """
        l0_feats = spatial_feats['l0']  # (B, 4, 384)
        l1_feats = spatial_feats['l1']  # (B, 16, 384)
        l2_feats = spatial_feats['l2']  # (B, 16, 384)

        # QUANTIZE EACH LEVEL INDEPENDENTLY
        q0, loss0, idx0 = self.vq_l0(l0_feats)  # q0:(B,4,384) idx0:(B,4)
        q1, loss1, idx1 = self.vq_l1(l1_feats)  # q1:(B,16,384) idx1:(B,16)
        q2, loss2, idx2 = self.vq_l2(l2_feats)  # q2:(B,16,384) idx2:(B,16)

        total_vq_loss = loss0 + loss1 + loss2

        # DEAD-CODE TRACKING AND REVIVAL (TRAINING ONLY)
        if self.training:
            self._accumulate_usage(idx0, idx1, idx2)
            self.maybe_revive(l0_feats, l1_feats, l2_feats)

        token_dict = {'l0': idx0, 'l1': idx1, 'l2': idx2}
        quant_dict  = {'l0': q0,   'l1': q1,   'l2': q2}

        return token_dict, total_vq_loss, quant_dict

    @torch.no_grad()
    def encode(self, spatial_feats: dict) -> dict:
        token_dict, _, _ = self.forward(spatial_feats)
        return token_dict

    @torch.no_grad()
    def decode(self, token_dict: dict) -> dict:
        """ DECODE TOKEN INDICES BACK TO PATCH EMBEDDINGS """
        out = {}
        for key, vq in [('l0', self.vq_l0), ('l1', self.vq_l1), ('l2', self.vq_l2)]:
            out[key] = vq.codebook(token_dict[key])
        return out

    def get_codebook_usage(self, token_dict: dict) -> dict:
        """ RETURN PER-LEVEL CODEBOOK UTILISATION STATS """
        totals = {'l0': self.num_codes_l0, 'l1': self.num_codes_l1, 'l2': self.num_codes_l2}
        stats = {}
        for key in ['l0', 'l1', 'l2']:
            tokens = token_dict[key]
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().numpy()
            unique = len(np.unique(tokens.reshape(-1)))
            total = totals[key]
            stats[key] = {
                'unique_codes': unique,
                'total_codes': total,
                'usage_pct': 100.0 * unique / total,
            }
        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: SpatialHRVQTokenizer (vq_spatial)")
    print("=" * 60)

    from encoder_v2 import SpatialAtariEncoder

    B = 4
    encoder = SpatialAtariEncoder(input_channels=4, d_model=384)
    tokenizer = SpatialHRVQTokenizer(d_model=384, num_codes_l0=16, num_codes_l1=64, num_codes_l2=64)

    params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"Tokenizer parameters: {params:,}")

    x = torch.randint(0, 256, (B, 4, 84, 84)).float()
    spatial_feats = encoder(x)

    # ENCODE IN EVAL MODE FIRST SO WE CAN ASSERT DETERMINISM
    tokenizer.eval()
    token_dict_eval = tokenizer.encode(spatial_feats)

    # TRAINING FORWARD (EMA UPDATES CODEBOOK)
    tokenizer.train()
    token_dict, vq_loss, quant_dict = tokenizer(spatial_feats)

    print(f"\nForward pass:")
    print(f"  vq_loss: {vq_loss.item():.4f}")
    for key in ['l0', 'l1', 'l2']:
        t = token_dict[key]
        q = quant_dict[key]
        print(f"  {key} tokens: {list(t.shape)}  quant: {list(q.shape)}")

    assert token_dict['l0'].shape == (B, 4),   f"l0 token shape wrong"
    assert token_dict['l1'].shape == (B, 16),  f"l1 token shape wrong"
    assert token_dict['l2'].shape == (B, 16),  f"l2 token shape wrong"
    assert quant_dict['l0'].shape == (B, 4,  384)
    assert quant_dict['l1'].shape == (B, 16, 384)
    assert quant_dict['l2'].shape == (B, 16, 384)

    # VERIFY EVAL ENCODE IS DETERMINISTIC
    tokenizer.eval()
    token_dict2 = tokenizer.encode(spatial_feats)
    tokenizer.train()
    assert (token_dict2['l0'] == token_dict_eval['l0']).all(), "Eval encode not deterministic"

    # DECODE
    recon = tokenizer.decode(token_dict)
    for key in ['l0', 'l1', 'l2']:
        assert recon[key].shape == quant_dict[key].shape

    # CODEBOOK USAGE
    usage = tokenizer.get_codebook_usage(token_dict)
    print(f"\nCodebook usage (before revival):")
    for key, stat in usage.items():
        print(f"  {key}: {stat['unique_codes']}/{stat['total_codes']} codes ({stat['usage_pct']:.1f}%)")

    # REVIVAL SMOKE TEST
    print(f"\nRevival smoke test (interval={tokenizer.revival_interval} steps):")
    tokenizer.train()
    tokenizer._train_step = tokenizer.revival_interval - 1

    # ACCUMULATE FAKE ZERO USAGE TO MAXIMISE DEAD CODES
    tokenizer._usage_l0.zero_()
    tokenizer._usage_l1.zero_()
    tokenizer._usage_l2.zero_()

    # ONE MORE FORWARD TRIGGERS REVIVAL
    token_dict2, _, _ = tokenizer(spatial_feats)

    usage2 = tokenizer.get_codebook_usage(token_dict2)
    print(f"\nCodebook usage (after revival):")
    for key, stat in usage2.items():
        print(f"  {key}: {stat['unique_codes']}/{stat['total_codes']} codes ({stat['usage_pct']:.1f}%)")

    # VERIFY EMA CLUSTER SIZES AFTER REVIVAL
    for vq_layer, name in [(tokenizer.vq_l0, 'l0'),
                            (tokenizer.vq_l1, 'l1'),
                            (tokenizer.vq_l2, 'l2')]:
        if hasattr(vq_layer, 'ema_cluster_size'):
            assert (vq_layer.ema_cluster_size > 0).all(), \
                f"Some {name} cluster sizes are still 0 after revival"
            print(f"  {name}: all cluster sizes > 0  OK")
        else:
            print(f"  {name}: GradientVQ - no EMA cluster sizes, skipping that check  OK")

    print("\nSpatialHRVQTokenizer (EMA mode): PASSED")

    # GRADIENTVQ MODE SMOKE TEST
    print("\n" + "=" * 60)
    print("SMOKE TEST: SpatialHRVQTokenizer with GradientVQ")
    print("=" * 60)

    tokenizer_gvq = SpatialHRVQTokenizer(
        d_model=384,
        num_codes_l0=16,
        num_codes_l1=64,
        num_codes_l2=64,
        use_gradient_vq=True,
    )

    # VERIFY EACH VQ LAYER IS A GRADIENTVQ INSTANCE
    assert isinstance(tokenizer_gvq.vq_l0, GradientVQ), "vq_l0 should be GradientVQ"
    assert isinstance(tokenizer_gvq.vq_l1, GradientVQ), "vq_l1 should be GradientVQ"
    assert isinstance(tokenizer_gvq.vq_l2, GradientVQ), "vq_l2 should be GradientVQ"
    print("  VQ layer types: GradientVQ  OK")

    tokenizer_gvq.train()
    token_dict_gvq, vq_loss_gvq, quant_dict_gvq = tokenizer_gvq(spatial_feats)

    # SHAPE CHECKS
    assert token_dict_gvq['l0'].shape == (B, 4),   "GradientVQ l0 token shape wrong"
    assert token_dict_gvq['l1'].shape == (B, 16),  "GradientVQ l1 token shape wrong"
    assert token_dict_gvq['l2'].shape == (B, 16),  "GradientVQ l2 token shape wrong"
    assert quant_dict_gvq['l0'].shape == (B, 4,  384)
    assert quant_dict_gvq['l1'].shape == (B, 16, 384)
    assert quant_dict_gvq['l2'].shape == (B, 16, 384)
    print("  Token and quant shapes: OK")

    # VQ_LOSS SHOULD BE POSITIVE
    assert vq_loss_gvq.item() > 0.0, f"GradientVQ vq_loss should be > 0, got {vq_loss_gvq.item()}"
    print(f"  vq_loss = {vq_loss_gvq.item():.4f}  (> 0)  OK")

    # VQ_LOSS SHOULD BE DIFFERENTIABLE
    assert vq_loss_gvq.grad_fn is not None, "GradientVQ vq_loss should have grad_fn"
    print("  vq_loss has grad_fn (differentiable)  OK")

    print("\nSpatialHRVQTokenizer (GradientVQ mode): PASSED")
    print("=" * 60)
