"""
Local World Model v3 (LWMv3) — Transformer-based architecture.

Uses a prefix-LM attention pattern:
- 3 observation tokens (PCD, agent state, affordance) with bidirectional attention
  (with pcd_tokens=K > 1: K local point-cloud patch tokens + state + affordance)
- 6 action tokens with causal attention (action i sees obs + actions 1..i)
- Per-step prediction heads on each action token output

Input (at timestep t=0):
- Current agent state [B, agent_state_dim]
- Current point cloud [B, N, 3]
- Current affordance point [B, 3]
- Action horizon [B, n_future_steps, action_dim]

Output:
- Agent state delta [B, n_future_steps, agent_state_dim]
- Affordance delta [B, n_future_steps, affordance_output_dim]
- contact_logit [B, n_future_steps, 1]  (optional, if predict_contact is set)

Contact prediction modes (predict_contact):
- None / "none": no contact head (default, backward compatible)
- "bce": auxiliary BCE head; affordance always predicted, contact is an extra output
- "gumbel_gate": contact gate that multiplies affordance output;
    training: Gumbel-Sigmoid straight-through; inference: hard 0/1
"""

import torch
from torch import nn

from lwm.world_model.pointnet2 import PointNetSetAbstraction
from lwm.world_model.pointnet import PointNetEncoderXYZ
from lwm.world_model.targets import check_frames, shift_observation_to_handle_frame


class PointNet2Encoder(nn.Module):
    """
    PointNet++ encoder that outputs a global feature vector.
    Uses set abstraction layers with hierarchical point cloud processing.

    3-layer hierarchy: 1280 → 256 → 64 → global pool.
    """
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # Hierarchical set abstraction layers
        # Each layer: npoint, radius, nsample, in_channel, mlp, group_all
        self.sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.1,
            nsample=32,
            in_channel=in_channels,
            mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.2,
            nsample=32,
            in_channel=128 + 3,  # +3 for xyz
            mlp=[128, 128, 256],
            group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,  # +3 for xyz
            mlp=[256, 512, out_channels],
            group_all=True  # Global aggregation
        )

    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3] point cloud
        Returns:
            features: [B, out_channels] global features
        """
        # PointNet++ expects [B, C, N] format
        xyz_t = xyz.permute(0, 2, 1)  # [B, 3, N]

        # Hierarchical feature learning
        l1_xyz, l1_points = self.sa1(xyz_t, None)    # [B, 3, 256], [B, 128, 256]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 64], [B, 256, 64]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 1], [B, out_channels, 1]

        # Global feature
        features = l3_points.squeeze(-1)  # [B, out_channels]
        return features


@torch.no_grad()
def farthest_point_sample(xyz, k):
    """Batched farthest point sampling.

    Deterministic: starts from the point closest to the cloud centroid (independent of
    the point order and equivariant to translations), then repeatedly adds the point
    farthest from the points selected so far.

    Args:
        xyz: [B, N, 3]
        k: number of points to select (<= N)
    Returns:
        idx: [B, k] long indices into N
    """
    xyz = xyz.float()
    B, N, _ = xyz.shape
    batch = torch.arange(B, device=xyz.device)
    idx = torch.empty(B, k, dtype=torch.long, device=xyz.device)
    centroid = xyz.mean(dim=1, keepdim=True)                       # [B, 1, 3]
    cur = ((xyz - centroid) ** 2).sum(-1).argmin(dim=1)            # [B]
    min_dist = torch.full((B, N), float("inf"), device=xyz.device)
    for i in range(k):
        idx[:, i] = cur
        d = ((xyz - xyz[batch, cur][:, None, :]) ** 2).sum(-1)     # [B, N]
        min_dist = torch.minimum(min_dist, d)
        cur = min_dist.argmax(dim=1)
    return idx


def group_points(xyz, n_groups, group_size):
    """Split a point cloud into local patches (FPS centres + k nearest neighbours).

    Args:
        xyz: [B, N, 3]
        n_groups: number of patches K
        group_size: points per patch G (the centre itself is included)
    Returns:
        centers: [B, K, 3] patch centres
        patches: [B, K, G, 3] neighbours relative to their centre
    """
    B, N, _ = xyz.shape
    if n_groups > N:
        raise ValueError(f"cannot pick {n_groups} patch centres from {N} points")
    g = min(group_size, N)
    batch = torch.arange(B, device=xyz.device)[:, None]
    centers = xyz[batch, farthest_point_sample(xyz, n_groups)]    # [B, K, 3]
    with torch.no_grad():
        # Exact (non-matmul) distances, so the grouping does not depend on the absolute
        # position of the cloud through rounding.
        dist = torch.cdist(centers.float(), xyz.float(),
                           compute_mode="donot_use_mm_for_euclid_dist")  # [B, K, N]
        knn = dist.topk(g, dim=-1, largest=False).indices          # [B, K, G]
    patches = xyz[batch[:, :, None], knn]                          # [B, K, G, 3]
    return centers, patches - centers[:, :, None, :]


class LWMv3(nn.Module):
    def __init__(self,
                 agent_state_dim: int = 10,
                 action_dim: int = 10,
                 predict_agent_state: bool = True,
                 n_future_steps: int = 6,
                 affordance_output_dim: int = 3,
                 # Transformer params
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 # PCD encoder
                 pointcloud_encoder_size: int = 256,
                 use_pointnet2: bool = True,
                 # Observation history
                 n_obs_steps: int = 1,
                 # Contact prediction
                 predict_contact: str = "none",
                 gumbel_tau: float = 0.5,
                 # Point-cloud tokens: 1 = one global token (paper models); K > 1 = K local
                 # patch tokens (FPS centres + pcd_group_size nearest neighbours each)
                 pcd_tokens: int = 1,
                 pcd_group_size: int = 32,
                 # Frames (see lwm.world_model.targets)
                 affordance_target_frame: str = "world",
                 input_frame: str = "world"):
        super().__init__()

        check_frames(affordance_target_frame, input_frame)
        # Frame of the 9-D part-motion output; consumers decode with
        # targets.decode_affordance_delta(..., frame=model.affordance_target_frame).
        self.affordance_target_frame = affordance_target_frame
        # "handle": point cloud and EE position are shifted by the affordance point in forward();
        # "handle_rel": the affordance token is shifted as well (no absolute position left).
        self.input_frame = input_frame

        self.agent_state_dim = agent_state_dim
        self.action_dim = action_dim
        self.predict_agent_state = predict_agent_state
        self.n_future_steps = n_future_steps
        self.affordance_output_dim = affordance_output_dim
        self.d_model = d_model
        self.n_obs_steps = n_obs_steps
        self.predict_contact = predict_contact if predict_contact else "none"
        self.gumbel_tau = gumbel_tau

        self.pcd_tokens = int(pcd_tokens)
        self.pcd_group_size = int(pcd_group_size)
        if self.pcd_tokens < 1:
            raise ValueError(f"pcd_tokens must be >= 1, got {pcd_tokens}")
        if self.pcd_tokens > 1:
            if n_obs_steps > 1:
                raise ValueError("pcd_tokens > 1 supports n_obs_steps == 1 only "
                                 f"(got n_obs_steps={n_obs_steps})")
            if use_pointnet2:
                raise ValueError("pcd_tokens > 1 uses the PointNet patch encoder; set use_pointnet2=false")

        n_obs_tokens = (self.pcd_tokens + 2) * n_obs_steps  # PCD (x K) + state + aff per timestep
        self.n_obs_tokens = n_obs_tokens
        total_tokens = n_obs_tokens + n_future_steps

        # --- Point cloud encoder ---
        if use_pointnet2:
            self.pcd_encoder = PointNet2Encoder(
                in_channels=3,
                out_channels=pointcloud_encoder_size,
            )
        else:
            self.pcd_encoder = PointNetEncoderXYZ(
                in_channels=3,
                out_channels=pointcloud_encoder_size,
                use_layernorm=False,
                final_norm="none",
            )
        self.pcd_proj = nn.Linear(pointcloud_encoder_size, d_model)
        if self.pcd_tokens > 1:
            # Patch mode: the same PointNet encodes each local patch (centre-relative
            # points); the patch centre enters through a learned positional embedding.
            self.pcd_center_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, d_model),
            )

        # --- Observation token encoders ---
        self.state_encoder = nn.Linear(agent_state_dim, d_model)
        self.affordance_encoder = nn.Linear(3, d_model)

        # --- Action token encoder + step embeddings ---
        self.action_encoder = nn.Linear(action_dim, d_model)
        self.step_embedding = nn.Embedding(n_future_steps, d_model)

        # --- Token type embeddings ---
        # For n_obs_steps=1: [1, 1, d_model] broadcasts to all 3 obs tokens (backward compatible)
        # For n_obs_steps>1: [1, 3, d_model] gives per-type embedding (PCD=0, state=1, aff=2)
        # For pcd_tokens>1: [1, 3, d_model] as well; the PCD embedding is shared by all K patch tokens
        if n_obs_steps > 1 or self.pcd_tokens > 1:
            self.obs_type_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
            self.obs_step_embed = nn.Embedding(n_obs_steps, d_model) if n_obs_steps > 1 else None
        else:
            self.obs_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.obs_step_embed = None
        self.action_type_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Input LayerNorm ---
        self.input_norm = nn.LayerNorm(d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Pre-computed attention mask (prefix-LM pattern) ---
        # PyTorch convention: True = blocked, False = allowed
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)
        # Obs tokens attend to all obs tokens (bidirectional)
        mask[:n_obs_tokens, :n_obs_tokens] = False
        # Action tokens: causal within actions, full attention to obs
        for i in range(n_future_steps):
            mask[n_obs_tokens + i, :n_obs_tokens] = False          # action i → all obs
            mask[n_obs_tokens + i, n_obs_tokens:n_obs_tokens + i + 1] = False  # action i → actions 0..i
        self.register_buffer('attn_mask', mask)

        # --- Prediction heads ---
        self.affordance_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, affordance_output_dim),
        )
        self.agent_state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, agent_state_dim),
        )

        # --- Contact prediction head (optional) ---
        if self.predict_contact != "none":
            self.contact_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

        # --- Normalized identity for gumbel_gate (set by training script) ---
        if self.predict_contact == "gumbel_gate":
            # Default to zeros (≈ correct when mean ≈ physical identity).
            # Must be overwritten via set_normalized_identity() before training.
            self.register_buffer(
                'identity_normalized',
                torch.zeros(affordance_output_dim)
            )

    def set_normalized_identity(self, mean: torch.Tensor, std: torch.Tensor):
        """Compute and register the normalized identity transform.

        For 9-DoF: physical identity = [1,0,0, 0,1,0, 0,0,0].
        Normalized identity = (physical_identity - mean) / std.

        Args:
            mean: [affordance_output_dim] normalization mean
            std: [affordance_output_dim] normalization std
        """
        physical_identity = torch.zeros(self.affordance_output_dim)
        if self.affordance_output_dim == 9:
            physical_identity[0] = 1.0  # rot_6d col1 = [1, 0, 0]
            physical_identity[4] = 1.0  # rot_6d col2 = [0, 1, 0]
        normalized = (physical_identity - mean.cpu()) / std.cpu()
        self.identity_normalized = normalized.to(self.identity_normalized.device)

    def forward(self, observation):
        """
        Forward pass for LWMv3.

        Args:
            observation: dict with keys:
                - agent_state: [B, agent_state_dim] or [B, n_obs_steps, agent_state_dim]
                - static_affordance: [B, 3] or [B, n_obs_steps, 3]
                - pointcloud: [B, N, 3] or [B, n_obs_steps, N, 3]
                - action_horizon: [B, n_future_steps, action_dim]

        Returns:
            dict with keys:
                - agent_state_delta: [B, n_future_steps, agent_state_dim]
                - affordance_delta: [B, n_future_steps, affordance_output_dim]
        """
        if self.input_frame in ("handle", "handle_rel"):
            observation = shift_observation_to_handle_frame(
                observation, shift_affordance=self.input_frame == "handle_rel")
        agent_state = observation["agent_state"]
        affordance = observation["static_affordance"]
        pointcloud = observation["pointcloud"]
        actions = observation["action_horizon"]

        B = agent_state.shape[0]

        # --- Encode observation tokens ---
        if self.pcd_tokens > 1:
            if agent_state.ndim == 3:
                if agent_state.shape[1] != 1:
                    raise ValueError("pcd_tokens > 1 supports a single observation step only")
                agent_state, affordance, pointcloud = agent_state[:, 0], affordance[:, 0], pointcloud[:, 0]
            centers, patches = group_points(pointcloud, self.pcd_tokens, self.pcd_group_size)
            K, G = patches.shape[1], patches.shape[2]
            patch_feat = self.pcd_encoder(patches.reshape(B * K, G, 3))   # [B*K, pointcloud_encoder_size]
            pcd_tokens = self.pcd_proj(patch_feat).reshape(B, K, -1)      # [B, K, d_model]
            pcd_tokens = pcd_tokens + self.pcd_center_embed(centers)
            state_token = self.state_encoder(agent_state)                 # [B, d_model]
            aff_token = self.affordance_encoder(affordance)               # [B, d_model]

            type_embed = self.obs_type_embed                              # [1, 3, d_model]
            obs_tokens = torch.cat([
                pcd_tokens + type_embed[:, :1],
                torch.stack([state_token, aff_token], dim=1) + type_embed[:, 1:],
            ], dim=1)                                                     # [B, K + 2, d_model]
        elif agent_state.ndim == 3:
            # Multi-obs: [B, T, ...] where T = n_obs_steps
            T = agent_state.shape[1]

            # Encode each timestep through shared encoders via reshape
            pcd_feat = self.pcd_encoder(pointcloud.reshape(B * T, *pointcloud.shape[2:]))
            pcd_tokens = self.pcd_proj(pcd_feat).reshape(B, T, -1)           # [B, T, d_model]
            state_tokens = self.state_encoder(
                agent_state.reshape(B * T, -1)
            ).reshape(B, T, -1)                                               # [B, T, d_model]
            aff_tokens = self.affordance_encoder(
                affordance.reshape(B * T, -1)
            ).reshape(B, T, -1)                                               # [B, T, d_model]

            # Interleave: [pcd_0, state_0, aff_0, pcd_1, state_1, aff_1, ...]
            obs_tokens = torch.stack(
                [pcd_tokens, state_tokens, aff_tokens], dim=2
            )  # [B, T, 3, d_model]
            obs_tokens = obs_tokens.reshape(B, T * 3, -1)  # [B, T*3, d_model]

            # Add per-type embedding (tiled across timesteps)
            type_embed = self.obs_type_embed.repeat(1, T, 1)  # [1, T*3, d_model]
            obs_tokens = obs_tokens + type_embed

            # Add temporal step embedding
            if self.obs_step_embed is not None:
                step_ids = torch.arange(T, device=agent_state.device)
                step_emb = self.obs_step_embed(step_ids)  # [T, d_model]
                # Repeat for each of the 3 token types per timestep
                step_emb = step_emb.unsqueeze(1).expand(-1, 3, -1).reshape(T * 3, -1)
                obs_tokens = obs_tokens + step_emb.unsqueeze(0)
        else:
            # Single-obs: [B, ...] — original path
            pcd_feat = self.pcd_encoder(pointcloud)            # [B, pointcloud_encoder_size]
            pcd_token = self.pcd_proj(pcd_feat)                # [B, d_model]
            state_token = self.state_encoder(agent_state)      # [B, d_model]
            aff_token = self.affordance_encoder(affordance)    # [B, d_model]

            obs_tokens = torch.stack([pcd_token, state_token, aff_token], dim=1)  # [B, 3, d_model]
            obs_tokens = obs_tokens + self.obs_type_embed

        # --- Encode action tokens (supports variable-length action horizons for evaluation) ---
        n_obs_tokens = obs_tokens.shape[1]
        n_actions = actions.shape[1]
        action_tokens = self.action_encoder(actions)       # [B, n_actions, d_model]
        step_ids = torch.arange(n_actions, device=actions.device)
        action_tokens = action_tokens + self.step_embedding(step_ids)  # broadcast over batch
        action_tokens = action_tokens + self.action_type_embed

        # --- Concatenate and normalize ---
        n_tokens = n_obs_tokens + n_actions
        tokens = torch.cat([obs_tokens, action_tokens], dim=1)  # [B, n_tokens, d_model]
        tokens = self.input_norm(tokens)

        # --- Transformer with prefix-LM mask (slice for variable-length actions) ---
        mask = self.attn_mask[:n_tokens, :n_tokens]
        out = self.transformer(tokens, mask=mask)  # [B, n_tokens, d_model]

        # --- Extract action token outputs (skip all obs tokens) ---
        action_out = out[:, n_obs_tokens:, :]  # [B, n_actions, d_model]

        # --- Per-step predictions ---
        affordance_delta = self.affordance_head(action_out)  # [B, n_actions, affordance_output_dim]

        # --- Contact prediction (optional) ---
        if self.predict_contact != "none":
            contact_logit = self.contact_head(action_out)  # [B, n_actions, 1]

            if self.predict_contact == "gumbel_gate":
                # Gumbel-Sigmoid gate: blend affordance toward normalized identity
                # when no contact predicted. Uses straight-through estimator for
                # hard gate in forward, soft gradients in backward.
                #
                # identity_normalized must be set via set_normalized_identity()
                # before training (contains (physical_identity - mean) / std).
                if self.training:
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(contact_logit).clamp(1e-8, 1 - 1e-8)
                    ) + 1e-8)
                    soft_gate = torch.sigmoid((contact_logit + gumbel_noise) / self.gumbel_tau)
                    hard_gate = (soft_gate > 0.5).float()
                    gate = hard_gate - soft_gate.detach() + soft_gate  # straight-through
                else:
                    gate = (contact_logit > 0.0).float()

                # Gate toward normalized identity (not physical identity or zeros)
                identity = self.identity_normalized.expand_as(affordance_delta)
                affordance_delta = gate * affordance_delta + (1 - gate) * identity

        result = {"affordance_delta": affordance_delta}
        if self.predict_contact != "none":
            result["contact_logit"] = contact_logit
        if self.predict_agent_state:
            agent_state_delta = self.agent_state_head(action_out)  # [B, n_actions, agent_state_dim]
            result["agent_state_delta"] = agent_state_delta
        else:
            result["agent_state_delta"] = torch.zeros(
                B, n_actions, self.agent_state_dim, device=actions.device
            )

        return result
