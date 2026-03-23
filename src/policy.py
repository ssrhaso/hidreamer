""" POLICY NETWORKS FOR ACTOR-CRITIC RL """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Optional
import math
import copy


# SYMLOG / SYMEXP TRANSFORMS FOR REWARDS AND VALUES
def symlog(
    x : torch.Tensor
    ) -> torch.Tensor:
    """ COMPRESS LARGE MAGNITUDES: SIGN(X) * LN(|X| + 1) """
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(
    x : torch.Tensor
    ) -> torch.Tensor:
    """ INVERSE OF SYMLOG: SIGN(X) * (EXP(|X|) - 1) """
    return torch.sign(x) * (torch.expm1(torch.abs(x)))

# HORIZON SCHEDULE FOR IMAGINATION
def get_horizon(
    current_step : int,
    total_steps : int,
    max_horizon : int = 30,
    min_horizon : int = 5,
    flat_horizon : int = 15,
    mode : str = "decay",
) -> int:
    """ UNIFIED HORIZON SCHEDULER FOR IMAGINATION ROLLOUTS """

    progress = min(1.0, current_step / max(total_steps, 1))

    # CONSTANT HORIZON
    if mode == "flat":
        return flat_horizon

    # COSINE DECAY FROM MAX TO MIN
    elif mode == "decay":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return int(round(min_horizon + (max_horizon - min_horizon) * cosine))

    # COSINE BELL CURVE FROM MIN TO MAX BACK TO MIN
    elif mode == "bell":
        cosine = 0.5 * (1.0 - math.cos(2.0 * math.pi * progress))
        return int(round(min_horizon + (max_horizon - min_horizon) * cosine))

    else:
        raise ValueError(f"Unknown horizon mode: {mode}. Use 'flat', 'decay', or 'bell'.")

# POLICY NETWORKS
class HierarchicalFeatureExtractor(nn.Module):
    """ CONVERTS HRVQ TOKEN INDICES TO DENSE FEATURE VECTOR VIA CODEBOOK LOOKUP """

    def __init__(
        self,
        hrvq_tokenizer,
        mode : str = 'concat',
        d_model : int = 384,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.hrvq = hrvq_tokenizer  # FROZEN

        if mode == "concat":
            self.feat_dim = d_model * 3  # 1152 DIMENSIONS

        elif mode == "attention":
            self.feat_dim = d_model      # 384 DIMENSIONS

            # 3-TOKEN SELF-ATTENTION AGGREGATION
            self.cross_attn = nn.MultiheadAttention(
                embed_dim = d_model,
                num_heads = 4,
                dropout = 0.0,
                batch_first = True
            )
            self.attn_norm = nn.LayerNorm(d_model)
            self.feat_dim = d_model

        else:
            raise ValueError(f"Unknown Mode : {mode}. Use 'concat' or 'attention'.")

    @torch.no_grad()
    def _lookup_codebooks(
        self,
        tokens : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ LOOK UP CODEBOOK EMBEDDINGS FOR EACH HRVQ LAYER """

        emb_l0 = self.hrvq.vq_layers[0].codebook(tokens[:, 0]) # (B, 384)
        emb_l1 = self.hrvq.vq_layers[1].codebook(tokens[:, 1]) # (B, 384)
        emb_l2 = self.hrvq.vq_layers[2].codebook(tokens[:, 2]) # (B, 384)

        return emb_l0, emb_l1, emb_l2

    def forward(
        self,
        tokens : torch.Tensor,
    )-> torch.Tensor:
        """ FORWARD PASS """

        # LOOKUP CODEBOOK EMBEDDINGS FOR EACH LAYER
        emb_l0, emb_l1, emb_l2 = self._lookup_codebooks(tokens) # (B, 384) each

        # OPTION A - CONCATENATE LAYER EMBEDDINGS
        if self.mode == "concat":
            feat = torch.cat([emb_l0, emb_l1, emb_l2], dim = -1) # (B, 1152)

        # OPTION B - ATTENTION OVER LAYER EMBEDDINGS
        elif self.mode == "attention":
            seq = torch.stack(tensors = [emb_l0, emb_l1, emb_l2], dim = 1)  # (B, 3, 384)
            attended, _ = self.cross_attn(seq, seq, seq)        # (B, 3, 384)
            attended = self.attn_norm(attended + seq)           # RESIDUAL + NORM
            feat = attended.mean(dim = 1)                         # MEAN POOL (B, 384)

        return feat


class HiddenStateFeatureExtractor(nn.Module):
    """ EXTRACTS POLICY FEATURES FROM FROZEN TRANSFORMER HIDDEN STATES """

    def __init__(
        self,
        d_model : int = 384,
        use_projection : bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_projection = use_projection
        self.feat_dim = d_model * 3  # 1152D

        # PROJECTION LAYER
        self.projection = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.feat_dim),
            nn.Linear(in_features = self.feat_dim, out_features = self.feat_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        hidden_states : torch.Tensor,
    ) -> torch.Tensor:
        """ FORWARD PASS """

        if hidden_states.dim() == 3:
            # (B, 3, 384) -> (B, 1152) CONCATENATE HIDDEN STATES
            feat = hidden_states.reshape(shape = (hidden_states.size(0), -1))

        else:
            feat = hidden_states  # ASSUME ALREADY CONCATENATED (B, 1152)

        if self.use_projection:
            feat = self.projection(feat)  # PROJECTED FEATURES (B, 1152)

        return feat

    # DISPATCH FLAG USED BY IMAGINEROLLOUT AND _TRAIN_AUX
    is_visual_mode: bool = False


# VISUAL POLICY COMPONENTS
class VisualEncoder(nn.Module):
    """ CNN ENCODER FOR MULTI-SCALE DECODED ATARI FRAMES """

    def __init__(self, input_channels: int = 3, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),  # 20x20
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),              # 9x9
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),              # 7x7
            nn.ReLU(inplace=True),
            nn.Flatten(),                                                         # 3136
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ ENCODE (B, 3, 84, 84) TO (B, FEAT_DIM) """
        return self.head(self.cnn(x))


class VisualFeatureExtractor(nn.Module):
    """ POLICY FEATURE EXTRACTION VIA HIERARCHICAL PIXEL DECODING """

    is_visual_mode: bool = True   # DISPATCH FLAG FOR IMAGINEROLLOUT AND _TRAIN_AUX

    def __init__(self, hrvq_tokenizer, decoder, visual_encoder: VisualEncoder):
        super().__init__()
        # REGISTER DECODER AS SUBMODULE SO .TO(DEVICE) MOVES IT
        self.decoder        = decoder
        self.visual_encoder = visual_encoder
        # DO NOT REGISTER HRVQ AS SUBMODULE - MANAGED BY POLICY_TRAIN.PY
        self._hrvq = hrvq_tokenizer
        self.feat_dim = visual_encoder.feat_dim

    def get_multi_scale_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """ TOKENS (B, 3) TO (B, 3, 84, 84) MULTI-SCALE DECODED FRAMES """
        # CODEBOOK LOOKUP - ALWAYS NO-GRAD
        with torch.no_grad():
            emb_l0 = self._hrvq.vq_layers[0].codebook(tokens[:, 0])  # (B, 384)
            emb_l1 = self._hrvq.vq_layers[1].codebook(tokens[:, 1])  # (B, 384)
            emb_l2 = self._hrvq.vq_layers[2].codebook(tokens[:, 2])  # (B, 384)

        coarse = self.decoder(emb_l0)                       # (B, 1, 84, 84)
        mid    = self.decoder(emb_l0 + emb_l1)              # (B, 1, 84, 84)
        full   = self.decoder(emb_l0 + emb_l1 + emb_l2)    # (B, 1, 84, 84)

        return torch.cat([coarse, mid, full], dim=1)        # (B, 3, 84, 84)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """ TOKENS (B, 3) TO POLICY FEATURES (B, FEAT_DIM) """
        frames = self.get_multi_scale_frames(tokens)        # (B, 3, 84, 84)
        return self.visual_encoder(frames)                  # (B, feat_dim)


class CPCHead(nn.Module):
    """ ACTOR-CRITIC CONTRASTIVE PREDICTIVE CODING HEAD """

    def __init__(
        self,
        feat_dim    : int,
        proj_dim    : int   = 256,
        k_steps     : int   = 3,
        temperature : float = 0.1,
    ):
        super().__init__()
        self.k_steps     = k_steps
        self.temperature = temperature

        # PREDICTOR: F_T TO PREDICTED FUTURE REPRESENTATION
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, proj_dim),
        )
        # PROJECTOR: F_{T+K} TO TARGET REPRESENTATION
        self.projector = nn.Linear(feat_dim, proj_dim)

    def forward(
        self,
        feats : torch.Tensor,
    ) -> torch.Tensor:
        """ COMPUTE INFONCE LOSS OVER TRAJECTORY FEATURES """
        B, H, D = feats.shape
        k = self.k_steps

        if H <= k:
            return feats.new_zeros(()).requires_grad_(True)

        anchors = feats[:, :H-k, :]   # (B, H-k, D)
        targets = feats[:, k:,   :]   # (B, H-k, D)

        preds = self.predictor(anchors)
        tgts  = self.projector(targets.detach())
        # DETACH TARGETS - ASYMMETRIC LIKE BYOL/SIMSIAM

        # L2-NORMALIZE FOR COSINE SIMILARITY SPACE
        preds = F.normalize(preds, dim=-1)
        tgts  = F.normalize(tgts,  dim=-1)

        # FLATTEN BATCH x TIME DIMENSION
        T          = B * (H - k)
        preds_flat = preds.reshape(T, -1)
        tgts_flat  = tgts.reshape(T, -1)

        # INFONCE: POSITIVE PAIRS ON DIAGONAL
        logits = torch.matmul(preds_flat, tgts_flat.t()) / self.temperature  # (T, T)
        labels = torch.arange(T, device=feats.device)

        return F.cross_entropy(logits, labels)


class ActorNetwork(nn.Module):
    """ THE ACTOR - MAPS FEATURE TO CATEGORICAL DISTRIBUTION OVER ACTIONS """

    def __init__(
        self,
        feat_dim : int,
        num_actions : int,
        hidden_dim : int = 512,
        unimix : float = 0.01,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.unimix = unimix

        # NETWORK ARCHITECTURE
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim),
            nn.Linear(in_features = feat_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = num_actions),
        )

        # ZERO INIT FOR UNBIASED STARTING POLICY
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)

    def forward(
        self,
        feat : torch.Tensor,
    ) -> D.Categorical:
        """ RETURN CATEGORICAL DISTRIBUTION OVER ACTIONS """

        # PASS MLP TO GET ACTION LOGITS
        logits = self.net(feat)

        # UNIMIX BLENDING TO PREVENT COLLAPSE
        if self.unimix > 0:
            probs = F.softmax(input = logits, dim = -1)
            uniform_probs = torch.ones_like(input = probs) / self.num_actions
            probs = (1 - self.unimix) * probs + self.unimix * uniform_probs
            dist = D.Categorical(probs = probs)

        else:
            dist = D.Categorical(logits = logits)

        return dist


class CriticNetwork(nn.Module):
    """ THE CRITIC - MAPS FEATURE TO SCALAR STATE VALUE """

    def __init__(
        self,
        feat_dim : int,
        hidden_dim : int = 512,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim),
            nn.Linear(in_features = feat_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = 1),
        )

        # ZERO INIT FOR STABLE STARTING VALUE PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)

    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ RETURN SCALAR VALUE PREDICTION """

        value = self.net(feat).squeeze(-1)  # (B,) SCALAR VALUE PREDICTION
        return value

class RewardNetwork(nn.Module):
    """ PREDICTS IMMEDIATE REWARD FROM STATE FEATURES IN SYMLOG SPACE """

    def __init__(
        self,
        feat_dim : int,
        hidden_dim : int = 512,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim),
            nn.Linear(in_features = feat_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = 1),
        )

        # ZERO INIT FOR STABLE STARTING REWARD PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)


    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ RETURN SCALAR REWARD PREDICTION """

        return self.net(feat).squeeze(-1)  # (B,) SCALAR REWARD PREDICTION


class ContinueNetwork(nn.Module):
    """ PREDICTS P(EPISODE CONTINUES) AS LOGIT """

    def __init__(
        self,
        feat_dim : int,
        hidden_dim : int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim),
            nn.Linear(in_features = feat_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features = hidden_dim, out_features = 1),
        )

        # ZERO INIT WITH POSITIVE BIAS FOR HIGH INITIAL CONTINUE PROBABILITY
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.constant_(tensor = self.net[-1].bias, val = 2.0)  # SIGMOID(2) ~ 0.88


    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ RETURN RAW CONTINUE LOGIT """

        return self.net(feat).squeeze(-1)  # (B,) SCALAR CONTINUE LOGIT


class CriticMovingAverage(nn.Module):
    """ EMA COPY OF CRITIC FOR STABLE LAMBDA-RETURN TARGETS """
    def __init__(
        self,
        critic : CriticNetwork,
        tau : float = 0.02,
    ):

        super().__init__()
        self.target_net = copy.deepcopy(critic)    # INITIAL COPY OF CRITIC
        self.target_net.requires_grad_(False)      # FROZEN TARGET NETWORK
        self.target_net.eval()
        self.tau = tau

    @torch.no_grad()
    def update(
        self,
        critic : CriticNetwork,
    ):
        """ EMA UPDATE 2% TOWARDS ONLINE CRITIC """

        for parameters_emacritic, parameters_onlinecritic in zip(self.target_net.parameters(), critic.parameters()):
            # EMA UPDATE: target = (1 - tau) * target + tau * online
            parameters_emacritic.data.lerp_(end = parameters_onlinecritic.data, weight = self.tau)

    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ RETURN STABLE VALUE ESTIMATES FROM EMA CRITIC """

        return self.target_net(feat)  # (B,) SCALAR VALUE PREDICTION


def compute_lambda_returns(
    rewards : torch.Tensor,
    values : torch.Tensor,
    continues : torch.Tensor,
    last_value : torch.Tensor,
    gamma : float = 0.997,
    lam : float = 0.95,
)-> torch.Tensor:
    """ BACKWARDS-RECURSIVE LAMBDA-RETURN BLENDING TD AND MONTE CARLO TARGETS """

    B , H = rewards.shape
    targets = torch.zeros_like(rewards)

    # BOOTSTRAP LAST VALUE
    next_val = last_value

    for t in reversed(range(H)):
        discount = gamma * continues[:, t]
        # USE NEXT STATE VALUE FOR ONE-STEP BOOTSTRAP
        next_state_value = values[:, t + 1] if t < H - 1 else last_value
        blend = (1 - lam) * next_state_value + lam * next_val
        targets[:, t] = rewards[:, t] + discount * blend
        next_val = targets[:, t]

    return targets


class ReturnNormalizer:
    """ NORMALIZES ADVANTAGES BY 5TH-95TH PERCENTILE RANGE OF RECENT RETURNS """

    def __init__(
        self,
        decay : float = 0.99,
        low_percentile : float = 5.0,
        high_percentile : float = 95.0,
    ):
        self.decay = decay
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.low_ema = 0.0   # INITIALIZED ON FIRST UPDATE
        self.high_ema = 1.0  # DEFAULT RANGE UNTIL FIRST REAL BATCH

    @torch.no_grad()
    def update(
        self,
        returns : torch.Tensor,
    ):
        """ UPDATE EMA OF RETURN PERCENTILES """

        low = torch.quantile(input = returns, q = self.low_percentile / 100).item()
        high = torch.quantile(input = returns, q = self.high_percentile / 100).item()

        # COLD START: SEED EMAS FROM FIRST REAL BATCH
        if self.low_ema == 0.0 and self.high_ema == 1.0:
            self.low_ema = low
            self.high_ema = high
            return

        # UPDATE LOW AND HIGH EMA
        self.low_ema = self.decay * self.low_ema + (1 - self.decay) * low
        self.high_ema = self.decay * self.high_ema + (1 - self.decay) * high


    def normalize(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """ NORMALIZE X BY EMA-TRACKED PERCENTILE RANGE """

        # EMA RANGE, AVOID DIV BY ZERO
        scale = max(self.high_ema - self.low_ema, 1.0)

        return x / scale

def count_policy_params(
    critic : CriticNetwork,
    actor : ActorNetwork,
    reward_net : RewardNetwork,
    continue_net : ContinueNetwork,
    feature_extractor : Optional[HierarchicalFeatureExtractor] = None,
) -> dict:
    """ COUNT TRAINABLE PARAMETERS ACROSS ALL POLICY NETWORKS """

    counts = {
        'actor' : sum(p.numel() for p in actor.parameters() if p.requires_grad),
        'critic' : sum(p.numel() for p in critic.parameters() if p.requires_grad),
        'reward_net' : sum(p.numel() for p in reward_net.parameters() if p.requires_grad),
        'continue_net' : sum(p.numel() for p in continue_net.parameters() if p.requires_grad),
    }

    # OPTIONAL FEATURE EXTRACTOR PARAM COUNT
    if feature_extractor is not None:
        counts['feature_extractor'] = sum(
            p.numel() for p in feature_extractor.parameters() if p.requires_grad
        )

    counts['total'] = sum(counts.values())

    return counts
