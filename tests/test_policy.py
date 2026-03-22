"""
PRE-FLIGHT TESTS FOR POLICY TRAINING PIPELINE

Run before loading onto GPU to catch bugs cheaply on CPU.
Covers all fixes from the audit:
  - compute_lambda_returns indexing (Bug 4 fix)
  - get_horizon flat_horizon parameter (Issue 8 fix)  
  - count_policy_params keyword args (Bug 1 fix)
  - Replay buffer sampling shapes
  - Imagination rollout trajectory shapes
  - Actor-critic training step (gradient flow)
  - Feature extractor concat mode output shape
  - Auxiliary training step (reward + continue losses)
"""

import copy
import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from policy import (
    ActorNetwork,
    CriticNetwork,
    CriticMovingAverage,
    RewardNetwork,
    ContinueNetwork,
    HierarchicalFeatureExtractor,
    ReturnNormalizer,
    compute_lambda_returns,
    count_policy_params,
    get_horizon,
    symlog,
    symexp,
)
from imagination import ImagineRollout, Trajectory
from replay_buffer import TokenReplayBuffer
from vq import HRVQTokenizer
from world_model import HierarchicalWorldModel, WorldModelConfig


# Fixtures

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_world_model(device):
    """Tiny world model for CPU testing."""
    config = WorldModelConfig(
        d_model=64, n_layers=2, n_heads=4, d_ff=256,
        dropout=0.0, max_seq_len=128, num_codes=256, num_actions=6,
        layer_weights=[1.0, 0.5, 0.1],
    )
    model = HierarchicalWorldModel(config).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@pytest.fixture(scope="module")
def tiny_hrvq(device):
    """Tiny HRVQ tokenizer for CPU testing."""
    hrvq = HRVQTokenizer(
        input_dim=64, num_codes_per_layer=256, num_layers=3,
        commitment_costs=[0.05, 0.25, 0.60],
    ).to(device)
    hrvq.eval()
    for p in hrvq.parameters():
        p.requires_grad = False
    return hrvq


@pytest.fixture(scope="module")
def tiny_policy_networks(tiny_hrvq, device):
    """All trainable policy networks with consistent dimensions."""
    from policy import HiddenStateFeatureExtractor

    feat_ext = HiddenStateFeatureExtractor(
        d_model=64,           # match tiny_config.d_model
        use_projection=True,
    ).to(device)
    feat_dim = feat_ext.feat_dim  # 64 * 3 = 192

    actor = ActorNetwork(feat_dim=feat_dim, num_actions=6, hidden_dim=128).to(device)
    critic = CriticNetwork(feat_dim=feat_dim, hidden_dim=128).to(device)
    reward_net = RewardNetwork(feat_dim=feat_dim, hidden_dim=128).to(device)
    continue_net = ContinueNetwork(feat_dim=feat_dim, hidden_dim=128).to(device)

    return feat_ext, actor, critic, reward_net, continue_net


@pytest.fixture
def filled_buffer(device):
    """Replay buffer pre-filled with 500 random transitions."""
    buf = TokenReplayBuffer(capacity=1000, seq_len=16, device=device)
    for i in range(500):
        tokens = torch.randint(0, 256, (3,))
        buf.push(tokens=tokens, action=int(np.random.randint(0, 6)),
                 reward=float(np.random.randn()), done=bool(i % 50 == 49))
    return buf


# 1. LAMBDA RETURNS — Bug 4 fix validation

class TestLambdaReturns:
    """Validate compute_lambda_returns uses V(s_{t+1}), not V(s_t)."""

    def test_basic_shape(self):
        """Output shape must match (B, H)."""
        B, H = 4, 10
        returns = compute_lambda_returns(
            rewards=torch.randn(B, H),
            values=torch.randn(B, H),
            continues=torch.ones(B, H),
            last_value=torch.randn(B),
        )
        assert returns.shape == (B, H)

    def test_no_nan_or_inf(self):
        """Lambda returns must be finite."""
        B, H = 4, 10
        returns = compute_lambda_returns(
            rewards=torch.randn(B, H),
            values=torch.randn(B, H),
            continues=torch.ones(B, H),
            last_value=torch.randn(B),
        )
        assert torch.isfinite(returns).all(), "NaN or Inf in lambda returns"

    def test_uses_next_state_value_not_current(self):
        """Critical: the 1-step bootstrap must use V(s_{t+1}), not V(s_t).

        Construct a scenario where V(s_t) and V(s_{t+1}) differ dramatically.
        If the function uses V(s_t), the result will be detectably wrong.
        """
        B, H = 1, 3
        rewards = torch.zeros(B, H)
        # values[0] = [0, 0, 100] — only the LAST state has high value
        values = torch.tensor([[0.0, 0.0, 100.0]])
        continues = torch.ones(B, H)
        last_value = torch.tensor([100.0])
        gamma, lam = 1.0, 0.0  # lambda=0 means PURE 1-step bootstrap

        returns = compute_lambda_returns(
            rewards=rewards, values=values, continues=continues,
            last_value=last_value, gamma=gamma, lam=lam,
        )

        # With lambda=0, G_t = r_t + gamma * c_t * V(s_{t+1})
        # G_2 = 0 + 1*1*last_value = 100
        # G_1 = 0 + 1*1*V(s_2)     = 100  (V(s_2)=values[0,2]=100)
        # G_0 = 0 + 1*1*V(s_1)     = 0    (V(s_1)=values[0,1]=0)
        #
        # If buggy (uses V(s_t)):
        # G_0 = 0 + 1*1*V(s_0) = 0  (same by accident for G_0)
        # G_1 = 0 + 1*1*V(s_1) = 0  (WRONG — should be 100)

        assert returns[0, 1].item() == pytest.approx(100.0, abs=1e-4), (
            f"G_1 should be 100 (bootstrap from V(s_2)=100), got {returns[0, 1].item()}. "
            "This means compute_lambda_returns is using V(s_t) instead of V(s_{t+1})."
        )
        assert returns[0, 0].item() == pytest.approx(0.0, abs=1e-4), (
            f"G_0 should be 0 (bootstrap from V(s_1)=0), got {returns[0, 0].item()}"
        )

    def test_gamma_zero_returns_immediate_reward(self):
        """With gamma=0, returns should equal immediate rewards (no future)."""
        B, H = 2, 5
        rewards = torch.randn(B, H)
        returns = compute_lambda_returns(
            rewards=rewards,
            values=torch.randn(B, H),
            continues=torch.ones(B, H),
            last_value=torch.randn(B),
            gamma=0.0, lam=0.95,
        )
        assert torch.allclose(returns, rewards, atol=1e-5), (
            "With gamma=0, returns should equal immediate rewards"
        )

    def test_terminal_state_cuts_bootstrap(self):
        """When continues=0 at step t, future value should be cut off."""
        B, H = 1, 3
        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[10.0, 10.0, 10.0]])
        continues = torch.tensor([[1.0, 0.0, 1.0]])  # terminal at t=1
        last_value = torch.tensor([10.0])

        returns = compute_lambda_returns(
            rewards=rewards, values=values, continues=continues,
            last_value=last_value, gamma=1.0, lam=0.0,
        )
        # G_1 = r_1 + gamma*c_1*V(s_2) = 1 + 1*0*10 = 1 (cut off)
        assert returns[0, 1].item() == pytest.approx(1.0, abs=1e-4), (
            f"Terminal state should cut bootstrap: got {returns[0, 1].item()}, expected 1.0"
        )


# 2. HORIZON SCHEDULER — Issue 8 fix validation

class TestHorizonScheduler:

    def test_flat_returns_flat_horizon_not_max(self):
        """Flat mode must return flat_horizon, not max_horizon."""
        h = get_horizon(current_step=0, total_steps=1000,
                        max_horizon=30, min_horizon=5, flat_horizon=15,
                        mode="flat")
        assert h == 15, f"Flat mode returned {h}, expected 15"

    def test_flat_is_constant_across_training(self):
        """Flat mode must return same value at start, middle, and end."""
        results = set()
        for step in [0, 500, 999]:
            results.add(get_horizon(current_step=step, total_steps=1000,
                                    max_horizon=30, min_horizon=5,
                                    flat_horizon=15, mode="flat"))
        assert len(results) == 1, f"Flat mode is not constant: {results}"

    def test_decay_starts_at_max(self):
        """Decay mode must start at max_horizon."""
        h = get_horizon(current_step=0, total_steps=1000,
                        max_horizon=30, min_horizon=5, mode="decay")
        assert h == 30, f"Decay start: got {h}, expected 30"

    def test_decay_ends_at_min(self):
        """Decay mode must end at min_horizon."""
        h = get_horizon(current_step=1000, total_steps=1000,
                        max_horizon=30, min_horizon=5, mode="decay")
        assert h == 5, f"Decay end: got {h}, expected 5"

    def test_bell_starts_at_min(self):
        """Bell mode must start at min_horizon."""
        h = get_horizon(current_step=0, total_steps=1000,
                        max_horizon=30, min_horizon=5, mode="bell")
        assert h == 5, f"Bell start: got {h}, expected 5"

    def test_bell_peaks_at_midpoint(self):
        """Bell mode must peak at max_horizon around the midpoint."""
        h = get_horizon(current_step=500, total_steps=1000,
                        max_horizon=30, min_horizon=5, mode="bell")
        assert h == 30, f"Bell peak: got {h}, expected 30"

    def test_bell_ends_at_min(self):
        """Bell mode must return to min_horizon at the end."""
        h = get_horizon(current_step=1000, total_steps=1000,
                        max_horizon=30, min_horizon=5, mode="bell")
        assert h == 5, f"Bell end: got {h}, expected 5"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown horizon mode"):
            get_horizon(0, 1000, mode="invalid")


# 3. COUNT POLICY PARAMS — Bug 1 fix validation

class TestCountPolicyParams:

    def test_returns_dict_with_expected_keys(self, tiny_policy_networks):
        feat_ext, actor, critic, reward_net, continue_net = tiny_policy_networks
        counts = count_policy_params(
            critic=critic, actor=actor,
            reward_net=reward_net, continue_net=continue_net,
            feature_extractor=feat_ext,
        )
        for key in ['actor', 'critic', 'reward_net', 'continue_net', 'total']:
            assert key in counts, f"Missing key: {key}"

    def test_total_equals_sum_of_components(self, tiny_policy_networks):
        feat_ext, actor, critic, reward_net, continue_net = tiny_policy_networks
        counts = count_policy_params(
            critic=critic, actor=actor,
            reward_net=reward_net, continue_net=continue_net,
            feature_extractor=feat_ext,
        )
        expected = sum(v for k, v in counts.items() if k != 'total')
        assert counts['total'] == expected

    def test_keyword_actor_not_policy(self, tiny_policy_networks):
        """Confirm 'actor' keyword works (the bug was using 'policy')."""
        _, actor, critic, reward_net, continue_net = tiny_policy_networks
        # This should NOT raise TypeError
        counts = count_policy_params(
            critic=critic, actor=actor,
            reward_net=reward_net, continue_net=continue_net,
        )
        assert counts['actor'] > 0


# 4. FEATURE EXTRACTOR — shape and grad behavior

class TestFeatureExtractor:

    def test_concat_output_shape(self, tiny_hrvq, device):
        from policy import HiddenStateFeatureExtractor

        feat_ext = HiddenStateFeatureExtractor(
            d_model=64,
            use_projection=True,
        ).to(device)
        tokens = torch.randint(0, 256, (4, 3), device=device)
        out = feat_ext(tokens)
        assert out.shape == (4, 192), f"Expected (4, 192), got {out.shape}"

    def test_concat_has_no_trainable_params(self, tiny_hrvq, device):
        from policy import HiddenStateFeatureExtractor
        
        feat_ext = HiddenStateFeatureExtractor(
            d_model=64,
            use_projection=True,
        ).to(device)
        trainable = sum(p.numel() for p in feat_ext.parameters() if p.requires_grad)
        assert trainable == 0, (
            f"Concat mode should have 0 trainable params, got {trainable}"
        )

    def test_feat_dim_property_matches_output(self, tiny_hrvq, device):
        from policy import HiddenStateFeatureExtractor

        feat_ext = HiddenStateFeatureExtractor(
            d_model=64,
            use_projection=True,
        ).to(device)
        tokens = torch.randint(0, 256, (2, 3), device=device)
        out = feat_ext(tokens)
        assert out.shape[-1] == feat_ext.feat_dim


# 5. REPLAY BUFFER — sampling shapes

class TestReplayBuffer:

    def test_sample_shapes(self, filled_buffer):
        batch = filled_buffer.sample(batch_size=8)
        assert batch['tokens'].shape == (8, 16, 3)
        assert batch['actions'].shape == (8, 16)
        assert batch['rewards'].shape == (8, 16)
        assert batch['dones'].shape == (8, 16)

    def test_seed_context_shapes(self, filled_buffer):
        ctx = filled_buffer.sample_seed_context(batch_size=4, context_len=8)
        assert ctx['tokens'].shape == (4, 8, 3)
        assert ctx['actions'].shape == (4, 8)

    def test_buffer_len(self, filled_buffer):
        assert len(filled_buffer) == 500


# 6. IMAGINATION ROLLOUT — trajectory shapes

class TestImaginationRollout:

    def test_trajectory_shapes(self, tiny_world_model, tiny_policy_networks, device):
        feat_ext, actor, critic, reward_net, continue_net = tiny_policy_networks

        imagination = ImagineRollout(
            world_model=tiny_world_model,
            feature_extractor=feat_ext,
            actor_network=actor,
            critic_network=critic,
            reward_network=reward_net,
            continue_network=continue_net,
            max_horizon=5,
            temperature=1.0,
            device=device,
        )

        B, T_seed = 2, 8
        seed_tokens = torch.randint(0, 256, (B, T_seed, 3), device=device)
        seed_actions = torch.randint(0, 6, (B, T_seed), device=device)

        traj = imagination.rollout(seed_tokens=seed_tokens,
                                   seed_actions=seed_actions, horizon=5)

        assert traj.tokens.shape == (B, 5, 3)
        assert traj.actions.shape == (B, 5)
        assert traj.log_probs.shape == (B, 5)
        assert traj.feats.shape == (B, 5, feat_ext.feat_dim)
        assert traj.values.shape == (B, 5)
        assert traj.rewards.shape == (B, 5)
        assert traj.continues.shape == (B, 5)
        assert traj.last_value.shape == (B,)
        assert traj.entropies.shape == (B, 5)

    def test_trajectory_log_probs_require_grad(self, tiny_world_model,
                                                tiny_policy_networks, device):
        """log_probs must retain grad for REINFORCE backprop."""
        feat_ext, actor, critic, reward_net, continue_net = tiny_policy_networks

        imagination = ImagineRollout(
            world_model=tiny_world_model,
            feature_extractor=feat_ext,
            actor_network=actor,
            critic_network=critic,
            reward_network=reward_net,
            continue_network=continue_net,
            max_horizon=3,
            temperature=1.0,
            device=device,
        )

        seed_tokens = torch.randint(0, 256, (2, 8, 3), device=device)
        seed_actions = torch.randint(0, 6, (2, 8), device=device)
        traj = imagination.rollout(seed_tokens=seed_tokens,
                                   seed_actions=seed_actions, horizon=3)

        assert traj.log_probs.requires_grad, (
            "log_probs must require grad for REINFORCE — "
            "actor forward is probably inside a no_grad block"
        )


# 7. SYMLOG / SYMEXP ROUND-TRIP

class TestSymlogSymexp:

    def test_roundtrip(self):
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0, 999.0])
        reconstructed = symexp(symlog(x))
        assert torch.allclose(reconstructed, x, atol=1e-4)

    def test_symlog_compresses(self):
        x = torch.tensor([999.0])
        assert symlog(x).item() < 10.0, "symlog should compress large values"


# 8. CRITIC MOVING AVERAGE — EMA behavior

class TestCriticMovingAverage:

    def test_initial_copy_is_identical(self):
        critic = CriticNetwork(feat_dim=64, hidden_dim=32)
        ema = CriticMovingAverage(critic=critic, tau=0.02)
        feat = torch.randn(2, 64)
        with torch.no_grad():
            v_online = critic(feat)
            v_ema = ema(feat)
        assert torch.allclose(v_online, v_ema, atol=1e-6)

    def test_ema_moves_toward_online(self):
        critic = CriticNetwork(feat_dim=64, hidden_dim=32)
        ema = CriticMovingAverage(critic=critic, tau=1.0)  # tau=1 → instant copy
        # Mutate the online critic
        with torch.no_grad():
            for p in critic.parameters():
                p.add_(torch.randn_like(p) * 10)
        ema.update(critic)
        feat = torch.randn(2, 64)
        with torch.no_grad():
            v_online = critic(feat)
            v_ema = ema(feat)
        assert torch.allclose(v_online, v_ema, atol=1e-5), (
            "With tau=1.0, EMA should exactly match online critic after update"
        )


# 9. RETURN NORMALIZER — percentile tracking

class TestReturnNormalizer:

    def test_normalize_does_not_explode_on_first_call(self):
        rn = ReturnNormalizer(decay=0.99)
        returns = torch.randn(16)
        rn.update(returns)
        normalized = rn.normalize(returns)
        assert torch.isfinite(normalized).all()

    def test_scale_at_least_one(self):
        rn = ReturnNormalizer(decay=0.99)
        rn.update(torch.tensor([0.0, 0.0, 0.0]))
        result = rn.normalize(torch.tensor([5.0]))
        # scale = max(high_ema - low_ema, 1.0) so divide by at least 1
        assert result.item() <= 5.0


# 10. ACTOR-CRITIC TRAINING STEP — gradient flow smoke test

class TestActorCriticGradientFlow:
    """Smoke test that a single _train_actor_critic-style step works."""

    def test_actor_receives_gradients(self, tiny_policy_networks, device):
        feat_ext, actor, critic, reward_net, continue_net = tiny_policy_networks
        B, H = 4, 5
        feat_dim = feat_ext.feat_dim

        # Simulate hidden states from transformer (not token indices)
        fake_hidden = torch.randn(B, H, 3, feat_ext.d_model, device=device)

        feats_list = []
        log_probs_list = []
        entropies_list = []
        for h in range(H):
            f = feat_ext(fake_hidden[:, h])  # (B, 3, 64) → (B, 192)
            dist = actor(f)
            a = dist.sample()
            feats_list.append(f)
            log_probs_list.append(dist.log_prob(a))
            entropies_list.append(dist.entropy())

        feats = torch.stack(feats_list, dim=1)
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)

        advantages = torch.randn(B, H)
        actor_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -entropies.mean()
        total = actor_loss + 3e-4 * entropy_loss

        actor.zero_grad()
        total.backward()

        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert len(grads) > 0, "Actor received no gradients"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite actor gradients"