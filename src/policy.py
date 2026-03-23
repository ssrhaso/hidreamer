""" POLICY NETWORKS FOR ACTOR CRITIC

ALL TRAINABLE COMPONENTS FOR IMAGINATION BASED RL

World Model is FROZEN during PPO training - only these networks are updated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Optional
import math
import copy 

""" HELPER FUNCTIONS """
# SYMLOG / SYMEXP TRANSFORMS FOR REWARDS AND VALUES TO HANDLE ATARI'S WIDE REWARD SCALE 
def symlog(
    x : torch.Tensor
    ) -> torch.Tensor:
    """Compress large magnitudes: sign(x) * ln(|x| + 1). 
    
    Keeps small values ~unchanged, squashes 999 -> 6.9.
    Applied to reward targets and critic values so Breakout's large rewards don't dominate Pong's small ones."""
    
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(
    x : torch.Tensor
    ) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1). 
    
    Decompress predictions back to real scale."""
    return torch.sign(x) * (torch.expm1(torch.abs(x)))

# GET HORIZON SCHEDULE FOR IMAGINATION
def get_horizon(
    current_step : int,
    total_steps : int,
    max_horizon : int = 30,
    min_horizon : int = 5,
    flat_horizon : int = 15,
    mode : str = "decay", # "flat" , "decay" , "bell"
) -> int:
    """ Unified Horizon Scheduler for Imagination Rollouts.
    
    mode = "flat" 15: constant horizon (baseline). (15 in DreamerV3, 15 in TWISTER)
    mode = "decay" 30-> 5: cosine decay from max_horizon to min_horizon. (explore more early, exploit more late)
    mode = "bell" 5->30->5: cosine bell curve peaking at mid-training for a ramp-up then compression curriculum.
    """
    
    progress = min(1.0, current_step / max(total_steps, 1))  # Normalize progress to [0, 1]
    
    # 15 STEP CONSTANT HORIZON (DREAMERV3 DEFAULT) [15]
    if mode == "flat":
        return flat_horizon
    
    # COSINE DECAY FROM MAX_HORIZON TO MIN_HORIZON [30 -> 5]
    elif mode == "decay":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine from 1 initial to 0 at end
        return int(round(min_horizon + (max_horizon - min_horizon) * cosine))
    
    # COSINE BELL CURVE FROM MIN TO MAX BACK TO MIN [5 -> 30 -> 5]
    elif mode == "bell":
        cosine = 0.5 * (1.0 - math.cos(2.0 * math.pi * progress))  # Cosine bell from 0 to 1 to 0
        return int(round(min_horizon + (max_horizon - min_horizon) * cosine))

    else:
        raise ValueError(f"Unknown horizon mode: {mode}. Use 'flat', 'decay', or 'bell'.")

# POLICY NETWORKS
class HierarchicalFeatureExtractor(nn.Module):
    """Converts HRVQ token indices -> dense feature vector by looking up frozen codebook embeddings.
    
    OPTION 1 - Concat mode: [codebook_0[L0] | codebook_1[L1] | codebook_2[L2]] → 1152D.
    OPTION 2 - Attention mode: 3-token self-attention over the three layer embeddings -> pooled 384D."""
    
    def __init__(
        self,
        hrvq_tokenizer, # FROZEN - used only for lookups
        mode : str = 'concat', # 'concat' or 'attention'
        d_model : int = 384,   # only used for attention mode
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.hrvq = hrvq_tokenizer # FROZEN 
        
        if mode == "concat":
            self.feat_dim = d_model * 3  # 1152 Dimension (3 layers * 384 each)
        
        elif mode == "attention":
            self.feat_dim = d_model      # 384 Dimension (pooled output)
            
            # 3 TOKEN SELF ATTENTION LAYER AGGREGRATION
            self.cross_attn = nn.MultiheadAttention(
                embed_dim = d_model, # 384 DIM
                num_heads = 4,       # 4 HEADS
                dropout = 0.0,       # 0.0 DROPOUT SINCE FEATURE EXTRACTION
                batch_first = True  
            )
            self.attn_norm = nn.LayerNorm(d_model)  # LAYER NORM FOR ATTENTION OUTPUT
            self.feat_dim = d_model                 # FINAL FEATURE DIMENSION AFTER ATTENTION (384)
            
        else:
            raise ValueError(f"Unknown Mode : {mode}. Use 'concat' or 'attention'.")
    
    @torch.no_grad()
    def _lookup_codebooks(
        self,
        tokens : torch.Tensor,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up codebook embeddings for each HRVQ layer """
        
        emb_l0 = self.hrvq.vq_layers[0].codebook(tokens[:, 0]) # (B, 384)
        emb_l1 = self.hrvq.vq_layers[1].codebook(tokens[:, 1]) # (B, 384)
        emb_l2 = self.hrvq.vq_layers[2].codebook(tokens[:, 2]) # (B, 384)
        
        return emb_l0, emb_l1, emb_l2
    
    def forward(
        self,
        tokens : torch.Tensor, 
    )-> torch.Tensor:
        """ Forward Pass """
        
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
    """ EXTRACTS POLICY FEATURES from FROZEN TRANSFORMER hidden states 
    
    Replaces CODEBOOK LOOKUP with direct use of the world model hidden stats 
    """      
    
    def __init__(
        self,
        d_model : int = 384,          # DIMENSION OF TRANSFORMER HIDDEN STATES
        use_projection : bool = True, # WHETHER TO PROJECT HIDDEN STATES TO A LOWER DIMENSION
    ):
        super().__init__()
        self.d_model = d_model
        self.use_projection = use_projection
        self.feat_dim = d_model * 3 # 1152D (same as codebook concat)
        
        # IF PROJECTION (Lets policy network adapt to frozen WM representation)
        self.projection = nn.Sequential(
            nn.LayerNorm(normalized_shape = self.feat_dim),
            nn.Linear(in_features = self.feat_dim, out_features = self.feat_dim),
            nn.SiLU(),
        )
        
    def forward(
        self,
        hidden_states : torch.Tensor, # (B, 3, d_model) HIDDEN STATES FROM TRANSFORMER
    ) -> torch.Tensor:
        """ Forward Pass """
        
        if hidden_states.dim() == 3:
            # (B, 3, 384) -> (B, 1152) CONCATENATE HIDDEN STATES
            feat = hidden_states.reshape(shape = (hidden_states.size(0), -1))
            
        else:
            feat = hidden_states # ASSUME ALREADY CONCATENATED (B, 1152)
        
        if self.use_projection:
            feat = self.projection(feat) # PROJECTED FEATURES (B, 1152)
        
        return feat
            

class CPCHead(nn.Module):
    """Actor-Critic Contrastive Predictive Coding (AC-CPC) head.

    Trains the feature extractor to produce representations that are predictive
    of future states, using an InfoNCE (contrastive) loss.  This is the core of
    TWISTER's representation improvement (Burchi & Timofte, ICLR 2025, §2.3).

    WHY THIS IS NEEDED
    
    The WM hidden states are trained to predict discrete codebook entries (256-way
    classification). They don't need to encode fine-grained position — just coarse
    cluster membership.  The linear probe showed max per-feature correlation with
    paddle y-position ≈ 0.27, and the feature_extractor projection was never being
    updated (optimizer bug).  Together these explain why learned policies fail to
    distinguish game states that matter for control.

    HOW IT WORKS
    Given a trajectory of features f_0, f_1, ..., f_{H-1}:
        - predictor(f_t) should match projector(f_{t+k}) for the SAME trajectory
        - but NOT match projector(f_{t+k}) from OTHER trajectories in the batch
    The InfoNCE loss (cross-entropy over the similarity matrix) forces f_t to
    encode information that determines f_{t+k}, i.e., the temporal dynamics —
    ball/paddle position and velocity.

    GRADIENT FLOW
    cpc_loss → CPCHead params (via cpc_optimizer)
               feature_extractor params (via actor_optimizer, since
               trajectory.feats = feature_extractor(WM_hidden_states) is in graph)
    """

    def __init__(
        self,
        feat_dim    : int,
        proj_dim    : int   = 256,   # projection dim (< feat_dim = info bottleneck)
        k_steps     : int   = 3,     # how many steps ahead to predict
        temperature : float = 0.1,   # InfoNCE temperature (0.07 SimCLR, 0.1 TWISTER)
    ):
        super().__init__()
        self.k_steps     = k_steps
        self.temperature = temperature

        # Predictor: f_t → predicted future representation
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, proj_dim),
        )
        # Projector: f_{t+k} → target representation (stops gradient from future)
        self.projector = nn.Linear(feat_dim, proj_dim)

    def forward(
        self,
        feats : torch.Tensor,   # (B, H, feat_dim) — trajectory features
    ) -> torch.Tensor:
        """Compute InfoNCE loss: predictor(f_t) should identify projector(f_{t+k}).

        Returns scalar contrastive loss.
        """
        B, H, D = feats.shape
        k = self.k_steps

        if H <= k:
            return feats.new_zeros(()).requires_grad_(True)

        anchors = feats[:, :H-k, :]   # (B, H-k, D) — current features
        targets = feats[:, k:,   :]   # (B, H-k, D) — future features k steps ahead

        preds = self.predictor(anchors)                    # (B, H-k, proj_dim)
        tgts  = self.projector(targets.detach())           # (B, H-k, proj_dim)
        # detach targets: we only want to pull anchors toward targets,
        # not push targets toward anchors (asymmetric, like BYOL/SimSiam)

        # L2-normalise → cosine similarity space
        preds = F.normalize(preds, dim=-1)
        tgts  = F.normalize(tgts,  dim=-1)

        # Flatten batch × time dimension: (T, proj_dim)
        T          = B * (H - k)
        preds_flat = preds.reshape(T, -1)
        tgts_flat  = tgts.reshape(T, -1)

        # InfoNCE: logit[i,j] = similarity(pred_i, target_j)
        # Positive pairs lie on the diagonal (i == j)
        logits = torch.matmul(preds_flat, tgts_flat.t()) / self.temperature  # (T, T)
        labels = torch.arange(T, device=feats.device)

        return F.cross_entropy(logits, labels)


class ActorNetwork(nn.Module):
    """ The Actor. 
    
    Maps 1152D feature -> categorical distribution over Atari actions.
    LayerNorm -> 2-layer MLP (ELU) -> softmax with 1% uniform mix to prevent collapse.
    Zero-init final layer -> uniform initial policy -> unbiased exploration at start."""

    def __init__(
        self,
        feat_dim : int,                 # CONCAT = 1152, ATTENTION = 384
        num_actions : int,              # GAME SPECIFIC (PONG = 6, BREAKOUT = 4, MsPACMAN = 9)
        hidden_dim : int = 512,         # SIZE OF HIDDEN LAYER IN MLP
        unimix : float = 0.01,          # DreamerV3 = 1%
    ):
        # SETUP 
        super().__init__()
        self.num_actions = num_actions
        self.unimix = unimix
        
        # NETWORK ARCHITECTURE, DreamerV3
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = num_actions) ,
        )
        
        # ZERO INITIALISATION - UNBIASED STARTING POLICY CONTRARY TO KAIMING INIT (PYTORCH DEFAULT)
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        
    def forward(
        self,
        feat : torch.Tensor,
    ) -> D.Categorical: 
        """ Forward Pass, returns Categorical Distribution over actions in given state """
        
        # PASS MLP TO GET ACTION LOGITS
        logits = self.net(feat)
        
        # UNIMIX BLENDING TO PREVENT COLLAPSE EARLY IN TRAINING
        
        if self.unimix > 0:
            
            probs = F.softmax(input = logits, dim = -1)                        # SOFTMAX - PROBABILITY DISTRIBUTION
            uniform_probs = torch.ones_like(input = probs) / self.num_actions  # UNIFORM DISTRIBUTION OVER ACTIONS
            probs = (1 - self.unimix) * probs + self.unimix * uniform_probs    # BLEND WITH UNIFORM
            dist = D.Categorical(probs = probs)                                # CATEGORICAL DISTRIBUTION
        
        else:
            dist = D.Categorical(logits = logits)                              # NO UNIMIX, STANDARD CATEGORICAL
        
        return dist
    

class CriticNetwork(nn.Module):
    """ The Critic. 
    
    Maps 1152D feature -> scalar 'how good is this state?'
    Same architecture as actor but outputs 1 value instead of num_actions logits.
    Separate from actor to avoid gradient interference (MSE vs REINFORCE scales differ)."""
    
    def __init__(
        self,
        feat_dim : int,         # CONCAT = 1152, ATTENTION = 384
        hidden_dim : int = 512, # SIZE OF HIDDEN LAYER IN MLP
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = 1) ,
        )
        
        # ZERO INITIALISATION FOR STABLE STARTING VALUE PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        
    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns scalar value prediction for given state """
        
        value = self.net(feat).squeeze(-1) # (B,) SCALAR VALUE PREDICTION
        return value

class RewardNetwork(nn.Module):
    """Predicts immediate reward from state features in symlog space.
    
    Trained supervised on real transitions (where true rewards exist).
    Used during imagination to provide reward signal when no real env is available."""
    
    def __init__(
        self,
        feat_dim : int,         # CONCAT = 1152, ATTENTION = 384
        hidden_dim : int = 512, # SIZE OF HIDDEN LAYER IN MLP
    ):
        super().__init__()
        
        # SAME ARCHITECTURE AS CRITIC BUT OUTPUTS 1 VALUE (REWARD PREDICTION) INSTEAD OF VALUE PREDICTION
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = 1) ,
        )
        
        # ZERO INITIALISATION FOR STABLE STARTING REWARD PREDICTIONS
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.zeros_(tensor = self.net[-1].bias)
        

    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns scalar reward prediction for given state features """

        return self.net(feat).squeeze(-1) # (B,) SCALAR REWARD PREDICTION
    

class ContinueNetwork(nn.Module):
    """Predicts p(episode continues) as a logit -> sigmoid for probability.
    
    Bias-initialized positive (sigmoid(2)≈0.88) because 99% of Atari steps aren't terminal.
    During imagination: effective_discount = gamma * p(continue), soft-truncating near game-over states."""

    def __init__(
        self,
        feat_dim : int,         # CONCAT = 1152, ATTENTION = 384
        hidden_dim : int = 512, # SIZE OF HIDDEN LAYER IN MLP
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape = feat_dim), 
            nn.Linear(in_features = feat_dim, out_features = hidden_dim), 
            nn.SiLU(), 
            nn.Linear(in_features = hidden_dim, out_features = hidden_dim), 
            nn.SiLU() ,
            nn.Linear(in_features = hidden_dim, out_features = 1) ,
        )
        
        # ZERO INITIALISATION WITH POSITIVE BIAS FOR HIGH INITIAL CONTINUE PROBABILITY (ATARI IS 99% NON-TERMINAL STEPS)
        nn.init.zeros_(tensor = self.net[-1].weight)
        nn.init.constant_(tensor = self.net[-1].bias, val = 2.0) # SIGMOID(2) ≈ 0.88, CLOSE TO REAL 0.99 TO PREVENT EARLY IMAGINATION COLLAPSE
        

    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns raw continue logit for given state features, 
        indicating how likely the episode is to continue from this state (game over). """
        
        return self.net(feat).squeeze(-1) # (B,) SCALAR CONTINUE LOGIT
    
    
class CriticMovingAverage(nn.Module):
    """Expontential Moving Average copy of the critic updated at τ=0.02 per step for stable λ-return targets.
    
    Solves the moving-target problem: critic can't train on its own rapidly-changing predictions.
    Same technique as DDPG, SAC, DreamerV3."""
    def __init__(
        self,
        critic : CriticNetwork,    # CRITIC NETWORK TO COPY
        tau : float = 0.02,        # EMA UPDATE RATE
    ):

        super().__init__()
        self.target_net = copy.deepcopy(critic)    # INITIAL COPY OF CRITIC
        self.target_net.requires_grad_(False)      # FROZEN TARGET NETWORK
        self.target_net.eval()                     # EVAL MODE FOR STABILITY
        self.tau = tau
        
    @torch.no_grad()
    def update(
        self,
        critic : CriticNetwork, # CRITIC NETWORK TO TRACK
    ):
        """ EMA Update 2% towards Online Critic """
        
        # ZIP TOGETHER TARGET NET AND ONLINE CRITIC PARAMETERS AND UPDATE IN PLACE
        for parameters_emacritic, parameters_onlinecritic in zip(self.target_net.parameters(), critic.parameters()):
            
            # EMA UPDATE : target = (1 - tau) * target + tau * online
            parameters_emacritic.data.lerp_(end = parameters_onlinecritic.data, weight = self.tau)
    
    def forward(
        self,
        feat : torch.Tensor,
    ) -> torch.Tensor:
        """ Forward Pass, returns stable value estimates for λ-return targets """
        
        return self.target_net(feat) # (B,) SCALAR VALUE PREDICTION FROM EMA CRITIC
    
    
    
def compute_lambda_returns(
    rewards : torch.Tensor,          # (B, H) PREDICTED  REWARD
    values : torch.Tensor,           # (B, H) PREDICTED VALUE FROM EMA CRITIC
    continues : torch.Tensor,        # (B, H) PREDICTED CONTINUE LOGITS FROM CONTINUE NETWORK
    last_value : torch.Tensor,       # (B,) VALUE PREDICTION FOR LAST STATE FROM EMA CRITIC 
    gamma : float = 0.997,           # DISCOUNT FACTOR
    lam : float = 0.95,              # LAMBDA FOR BLENDING TD AND MONTE CARLO TARGETS
)-> torch.Tensor:
    """Backwards-recursive λ-return: blends 1-step TD (low variance) with Monte Carlo (low bias).
    G_t = r_t + Y * c_t * [(1-λ)*V(s_{t+1}) + λ*G_{t+1}]. 
    
    λ = How much you trust the critic vs the reward network and future returns.
    
    λ=0.95 uses 95% of long-horizon info. 
    Returns (B, H) targets that the critic is trained to predict."""
    
    # (B, H) INITIALIZE TARGETS TENSOR
    B , H = rewards.shape
    targets = torch.zeros_like(rewards) 
    
    # BOOTSTRAP LAST VALUE FOR G_{H-1}
    next_val = last_value 
    
    for t in reversed(range(H)):
        discount = gamma * continues[:, t]
        # Use NEXT state's value for one-step bootstrap
        next_state_value = values[:, t + 1] if t < H - 1 else last_value
        blend = (1 - lam) * next_state_value + lam * next_val
        targets[:, t] = rewards[:, t] + discount * blend
        next_val = targets[:, t]
    
    return targets
    

class ReturnNormalizer:
    """Normalizes advantages by the 5th-95th percentile range of recent returns.
    
    Robust under sparse rewards where std≈0 would cause division-by-zero explosion.
    EMA-tracked percentiles (decay=0.99) adapt smoothly as training progresses."""
    
    def __init__(
        self,
        decay : float = 0.99,               # EMA DECAY FOR TRACKING RETURN PERCENTILES
        low_percentile : float = 5.0,       # LOW PERCENTILE FOR NORMALIZATION
        high_percentile : float = 95.0,     # HIGH PERCENTILE FOR NORMALIZATION
    ):
        self.decay = decay
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.low_ema = 0.0   # initialized on first update via cold-start branch
        self.high_ema = 1.0  # default range of 1 until first real batch arrives
    
    @torch.no_grad()
    def update(
        self,
        returns : torch.Tensor,             # (B,) RETURNS FROM RECENT BATCH
    ):
        """ Update EMA of return percentiles based on recent returns. 
        (since Atari rewards can be sparse, 
        we use percentiles instead of mean/std for robustness) """
        
        low = torch.quantile(input = returns, q = self.low_percentile / 100).item()
        high = torch.quantile(input = returns, q = self.high_percentile / 100).item()
        
        # COLD START: seed EMAs from first real batch instead of multiplying None
        if self.low_ema == 0.0 and self.high_ema == 1.0:
            self.low_ema = low
            self.high_ema = high
            return
        
        # UPDATE LOW AND HIGH EMA
        self.low_ema = self.decay * self.low_ema + (1 - self.decay) * low
        self.high_ema = self.decay * self.high_ema + (1 - self.decay) * high
        
        
    def normalize(
        self,
        x : torch.Tensor,                     # (B,) ADVANTAGES TO NORMALIZE
    ) -> torch.Tensor:
        """ Normalize x by the EMA-tracked percentile range of returns.
        Land in a consistent, sensible magintude for stable PPO training. """
        
        # EMA RANGE , AVOID DIV BY ZERO
        scale = max(self.high_ema - self.low_ema, 1.0) 
        
        return x / scale
        
def count_policy_params(
    critic : CriticNetwork,
    actor : ActorNetwork,
    reward_net : RewardNetwork,
    continue_net : ContinueNetwork,
    feature_extractor : Optional[HierarchicalFeatureExtractor] = None,
) -> dict:
    """Counts trainable parameters across all four networks. Sanity check: should be ~1.5-3M total"""
    
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