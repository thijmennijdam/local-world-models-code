"""World Model inference wrapper for MPC planning.

Provides interface between MPC controller and trained World Model checkpoints,
handling batched autoregressive trajectory prediction with proper normalization.
"""

import torch
import numpy as np
from typing import Dict, Optional
from termcolor import cprint


class WorldModelPredictor:
    """Wrapper for batched World Model trajectory prediction.

    Loads a trained WM checkpoint and provides efficient batched inference
    for MPC planning, handling normalization, denormalization, and proper
    state/affordance composition.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 affordance_proximity_gate_scale: float = None,
                 finger_close_threshold: float = None,
                 use_gt_agent_state: bool = False,
                 cumulative_agent_state_delta: bool = True):
        """Initialize predictor by loading WM checkpoint.

        Args:
            checkpoint_path: Path to trained WM checkpoint (.pth file)
            device: Device for inference ("cuda" or "cpu")
            affordance_proximity_gate_scale: Hard distance threshold (meters).
                Zero out affordance deltas when EE is farther than this from
                the affordance. None = disabled. Typical value: 0.04 (4cm).
            finger_close_threshold: Hard finger threshold (meters).
                Zero out affordance deltas when finger state is above this
                (i.e. gripper too open). finger_state: 0=closed, 0.04=open.
                None = disabled. Typical value: 0.02 (half closed).
            use_gt_agent_state: If True, use forward kinematics (action applied
                as state delta) instead of the WM's agent state prediction.
                Only the affordance prediction from the WM is used.
            cumulative_agent_state_delta: The LWM's agent-state target for step k
                is s_{t+k+1} - s_t (relative to the chunk's input state), so each
                predicted delta is composed onto the chunk's input state. False
                composes it onto the previous predicted step instead.
        """
        from lwm.world_model.analysis.checkpoint import load_checkpoint
        from lwm.world_model.targets import decode_affordance_delta
        from lwm.world_model.evaluation_utils import (
            compose_state_with_delta_batched,
            compose_affordance_with_9dof_delta_batched,
            rotation_transfer_6D_to_matrix_torch,
        )

        self.device = device
        self.compose_state_fn = compose_state_with_delta_batched
        self.compose_affordance_fn = compose_affordance_with_9dof_delta_batched
        self.rotation_6d_to_matrix = rotation_transfer_6D_to_matrix_torch
        self.decode_affordance_delta = decode_affordance_delta
        self.affordance_proximity_gate_scale = affordance_proximity_gate_scale
        self.finger_close_threshold = finger_close_threshold
        self.use_gt_agent_state = use_gt_agent_state
        self.cumulative_agent_state_delta = cumulative_agent_state_delta

        # Load checkpoint
        cprint(f"Loading WM checkpoint from {checkpoint_path}", "yellow")
        self.model, self.ckpt, self.norm_stats, self.config = load_checkpoint(
            checkpoint_path, device
        )
        self.model.eval()

        # Extract configuration
        self.n_obs_steps = self.config.model.get("n_obs_steps", 1)
        self.horizon = self.config.model.n_future_steps
        self.affordance_output_dim = self.config.model.get("affordance_output_dim", 3)
        self.agent_state_dim = self.config.model.agent_state_dim
        self.action_dim = self.config.model.action_dim
        self.predict_agent_state = self.config.model.get("predict_agent_state", True)
        # Frame of the 9-D part-motion output (missing in old configs = "world")
        self.affordance_target_frame = getattr(self.model, "affordance_target_frame", "world")
        self.input_frame = getattr(self.model, "input_frame", "world")

        # Extract normalization stats
        if self.norm_stats is None:
            cprint("WARNING: No normalization stats in checkpoint, using zeros", "yellow")
            self.action_mean = torch.zeros(self.action_dim, device=device)
            self.action_std = torch.ones(self.action_dim, device=device)
            self.agent_state_delta_mean = torch.zeros(self.agent_state_dim, device=device)
            self.agent_state_delta_std = torch.ones(self.agent_state_dim, device=device)
            self.affordance_delta_mean = torch.zeros(self.affordance_output_dim, device=device)
            self.affordance_delta_std = torch.ones(self.affordance_output_dim, device=device)
        else:
            self.action_mean = torch.from_numpy(self.norm_stats["action_mean"]).float().to(device)
            self.action_std = torch.from_numpy(self.norm_stats["action_std"]).float().to(device)
            self.agent_state_delta_mean = torch.from_numpy(
                self.norm_stats["agent_state_delta_mean"]
            ).float().to(device)
            self.agent_state_delta_std = torch.from_numpy(
                self.norm_stats["agent_state_delta_std"]
            ).float().to(device)
            self.affordance_delta_mean = torch.from_numpy(
                self.norm_stats["affordance_delta_mean"]
            ).float().to(device)
            self.affordance_delta_std = torch.from_numpy(
                self.norm_stats["affordance_delta_std"]
            ).float().to(device)

            # Action bounds (available in checkpoints trained after this change)
            if "action_min" in self.norm_stats:
                self.action_min = torch.from_numpy(self.norm_stats["action_min"]).float().to(device)
                self.action_max = torch.from_numpy(self.norm_stats["action_max"]).float().to(device)
                cprint(f"Action bounds from checkpoint: min={self.action_min.tolist()}, max={self.action_max.tolist()}", "cyan")
            else:
                self.action_min = None
                self.action_max = None
                cprint("NOTE: No action_min/action_max in checkpoint (older checkpoint)", "yellow")

        cprint(
            f"WM loaded: n_obs_steps={self.n_obs_steps}, horizon={self.horizon}, "
            f"affordance_dim={self.affordance_output_dim}, predict_agent_state={self.predict_agent_state}, "
            f"affordance_target_frame={self.affordance_target_frame}, input_frame={self.input_frame}, "
            f"use_gt_agent_state={self.use_gt_agent_state}, "
            f"aff_proximity_gate={'OFF' if self.affordance_proximity_gate_scale is None else f'{self.affordance_proximity_gate_scale:.3f}m'}, "
            f"finger_gate={'OFF' if self.finger_close_threshold is None else f'{self.finger_close_threshold:.3f}m'}",
            "green"
        )
        if self.use_gt_agent_state:
            cprint("  use_gt_agent_state=True: ignoring WM agent state prediction, using forward kinematics (action as state delta)", "yellow")
        elif not self.predict_agent_state:
            cprint("  When predict_agent_state=False, next agent state = current state + action (action applied as state delta)", "cyan")

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions using checkpoint stats.

        Args:
            actions: [..., action_dim] unnormalized actions

        Returns:
            [..., action_dim] normalized actions
        """
        return (actions - self.action_mean) / (self.action_std + 1e-8)

    def denormalize_delta(
        self,
        normalized_delta: torch.Tensor,
        delta_type: str
    ) -> torch.Tensor:
        """Denormalize predicted deltas.

        Args:
            normalized_delta: Normalized delta predictions
            delta_type: "agent_state" or "affordance"

        Returns:
            Denormalized delta
        """
        if delta_type == "agent_state":
            mean = self.agent_state_delta_mean
            std = self.agent_state_delta_std
        elif delta_type == "affordance":
            mean = self.affordance_delta_mean
            std = self.affordance_delta_std
        else:
            raise ValueError(f"Unknown delta_type: {delta_type}")

        return normalized_delta * std + mean

    def predict_trajectory(
        self,
        agent_state: torch.Tensor,      # [batch, agent_state_dim]
        affordance: torch.Tensor,       # [batch, 3]
        point_cloud: torch.Tensor,      # [batch, N_points, 3]
        actions: torch.Tensor,          # [batch, planning_horizon, action_dim]
        wm_batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressively predict state and affordance trajectory.

        The planning horizon is covered by chaining calls to the WM, each
        predicting up to ``self.horizon`` (H) steps.  For example, with H=6
        and a planning horizon of 12, two WM forward passes are made: the
        first predicts steps 1-6 from the initial observation, and its
        composed output seeds the second call for steps 7-12.  Within each
        chunk the model uses its full causal attention across action tokens,
        matching the training-time sequence structure.

        When wm_batch_size is set and smaller than batch, the samples are
        chunked into mini-batches to avoid GPU OOM while allowing large
        num_samples in the planner.

        Args:
            agent_state: Initial agent state [batch, 10]
            affordance: Initial affordance position [batch, 3]
            point_cloud: Point cloud observation [batch, N, 3]
            actions: Action sequence to execute [batch, planning_horizon, action_dim]
            wm_batch_size: Max samples per WM forward pass (None = all at once)

        Returns:
            Dictionary containing:
                - predicted_states: [batch, planning_horizon, agent_state_dim]
                - predicted_affordances: [batch, planning_horizon, 3] (always 3D
                    position, even when using 9-DoF deltas internally — the rigid
                    transform is composed back into a 3D position)
        """
        batch_size = agent_state.shape[0]

        # Chunk across the sample dimension if needed (GPU memory)
        if wm_batch_size is not None and batch_size > wm_batch_size:
            return self._predict_trajectory_chunked(
                agent_state, affordance, point_cloud, actions, wm_batch_size
            )

        planning_horizon = actions.shape[1]
        model_horizon = self.horizon  # H from model checkpoint (e.g. 6)

        # Storage for predictions
        predicted_states = []
        predicted_affordances = []
        contact_logits = []

        # Current state tracking
        cur_state = agent_state  # [batch, agent_state_dim]
        cur_aff = affordance     # [batch, 3]

        # Normalize actions
        actions_normalized = self.normalize_actions(actions)  # [batch, planning_horizon, action_dim]

        # Autoregressive rollout, chunked by model horizon
        with torch.no_grad():
            for chunk_start in range(0, planning_horizon, model_horizon):
                chunk_end = min(chunk_start + model_horizon, planning_horizon)
                chunk_actions = actions_normalized[:, chunk_start:chunk_end, :]  # [batch, chunk_len, action_dim]

                # Single model forward pass for the full chunk — the model
                # uses causal attention across action tokens, matching training.
                obs = self._build_obs(cur_state, cur_aff, point_cloud, chunk_actions)
                output = self.model(obs)
                chunk_start_state = cur_state

                # Process each step's prediction within the chunk to compose
                # states, affordances, and point clouds sequentially.
                for k in range(chunk_end - chunk_start):
                    # --- Affordance delta ---
                    affordance_delta_normalized = output["affordance_delta"][:, k, :]
                    affordance_delta = self.denormalize_delta(
                        affordance_delta_normalized, "affordance"
                    )
                    if self.affordance_target_frame != "world" and self.affordance_output_dim == 9:
                        # Decode step by step about the centroid of the cloud as it is now
                        # (the chunk's input cloud at k=0, then the transformed cloud), so a
                        # gated (identity) step leaves the later steps consistent.
                        affordance_delta = self.decode_affordance_delta(
                            affordance_delta.unsqueeze(1), point_cloud.mean(dim=1),
                            self.affordance_target_frame,
                        ).squeeze(1)

                    # --- Agent state delta ---
                    if self.predict_agent_state and not self.use_gt_agent_state:
                        agent_state_delta_normalized = output["agent_state_delta"][:, k, :]
                        agent_state_delta = self.denormalize_delta(
                            agent_state_delta_normalized, "agent_state"
                        )
                        base_state = chunk_start_state if self.cumulative_agent_state_delta else cur_state
                        next_state = self.compose_state_fn(base_state, agent_state_delta)
                    else:
                        action_for_step = actions[:, chunk_start + k, :]
                        next_state = self.compose_state_fn(cur_state, action_for_step)
                        # The finger joint is bounded (0 = closed, 0.04 m = open); the
                        # commanded delta alone can drive it outside that range.
                        next_state[:, 9] = next_state[:, 9].clamp(0.0, 0.04)

                    # --- Contact / proximity gating ---
                    has_proximity_gate = (
                        self.affordance_proximity_gate_scale is not None
                        or self.finger_close_threshold is not None
                    )
                    has_contact_gate = "contact_logit" in output
                    if has_proximity_gate or has_contact_gate:
                        keep = torch.ones(cur_aff.shape[0], 1, dtype=torch.bool, device=cur_aff.device)
                        if self.affordance_proximity_gate_scale is not None:
                            ee_pos = next_state[:, :3]
                            dist = torch.norm(ee_pos - cur_aff, dim=-1, keepdim=True)
                            keep = keep & (dist < self.affordance_proximity_gate_scale)
                        if self.finger_close_threshold is not None:
                            finger_state = next_state[:, 9:10]
                            keep = keep & (finger_state < self.finger_close_threshold)
                        if has_contact_gate:
                            contact_pred = (output["contact_logit"][:, k, :] > 0.0)
                            keep = keep & contact_pred
                        identity = torch.zeros_like(affordance_delta)
                        if self.affordance_output_dim == 9:
                            identity[:, 0] = 1.0
                            identity[:, 4] = 1.0
                        affordance_delta = torch.where(keep, affordance_delta, identity)

                    # --- Compose next affordance & update point cloud ---
                    if self.affordance_output_dim == 9:
                        next_aff = self.compose_affordance_fn(cur_aff, affordance_delta)
                        R = self.rotation_6d_to_matrix(affordance_delta[:, :6])
                        t = affordance_delta[:, 6:]
                        point_cloud = (
                            torch.bmm(R, point_cloud.transpose(1, 2)).transpose(1, 2)
                            + t.unsqueeze(1)
                        )
                    elif self.affordance_output_dim == 3:
                        next_aff = cur_aff + affordance_delta
                    else:
                        raise ValueError(
                            f"Unsupported affordance_output_dim: {self.affordance_output_dim}"
                        )

                    # --- Store predictions ---
                    if "contact_logit" in output:
                        contact_logits.append(output["contact_logit"][:, k, :])

                    predicted_states.append(next_state)
                    predicted_affordances.append(next_aff)

                    # Update current state for next step
                    cur_state = next_state
                    cur_aff = next_aff

        # Stack predictions
        predicted_states = torch.stack(predicted_states, dim=1)  # [batch, horizon, agent_state_dim]
        predicted_affordances = torch.stack(predicted_affordances, dim=1)  # [batch, horizon, 3]

        result = {
            "predicted_states": predicted_states,
            "predicted_affordances": predicted_affordances,
            "predicted_point_cloud": point_cloud,  # PCD after all rigid body transforms
        }
        if contact_logits:
            result["contact_logits"] = torch.stack(contact_logits, dim=1)  # [batch, horizon, 1]
        return result

    def _predict_trajectory_chunked(
        self,
        agent_state: torch.Tensor,
        affordance: torch.Tensor,
        point_cloud: torch.Tensor,
        actions: torch.Tensor,
        wm_batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Process samples in mini-batches and concatenate results."""
        batch_size = agent_state.shape[0]
        all_states = []
        all_affs = []
        all_contact_logits = []

        for start in range(0, batch_size, wm_batch_size):
            end = min(start + wm_batch_size, batch_size)
            chunk_result = self.predict_trajectory(
                agent_state[start:end],
                affordance[start:end],
                point_cloud[start:end],
                actions[start:end],
                wm_batch_size=None,  # no further chunking
            )
            all_states.append(chunk_result["predicted_states"])
            all_affs.append(chunk_result["predicted_affordances"])
            if "contact_logits" in chunk_result:
                all_contact_logits.append(chunk_result["contact_logits"])

        result = {
            "predicted_states": torch.cat(all_states, dim=0),
            "predicted_affordances": torch.cat(all_affs, dim=0),
        }
        if all_contact_logits:
            result["contact_logits"] = torch.cat(all_contact_logits, dim=0)
        return result

    def _build_obs(
        self,
        agent_state: torch.Tensor,      # [batch, agent_state_dim]
        affordance: torch.Tensor,       # [batch, 3]
        point_cloud: torch.Tensor,      # [batch, N, 3]
        action_horizon: torch.Tensor,   # [batch, horizon, action_dim]
    ) -> Dict[str, torch.Tensor]:
        """Build observation dictionary for model input.

        Args:
            agent_state: Current agent state [batch, agent_state_dim]
            affordance: Current affordance [batch, 3]
            point_cloud: Point cloud [batch, N, 3]
            action_horizon: Action sequence [batch, horizon, action_dim]

        Returns:
            Observation dict with proper shape for model input
        """
        if self.n_obs_steps == 1:
            # Single observation step: model expects [B, D] (ndim=2).
            # LWMv3.forward() dispatches on ndim: 2 → single-obs path, 3 → multi-obs path.
            # Matches evaluation_utils.build_observation() for n_obs_steps=1.
            obs = {
                "agent_state": agent_state,            # [batch, agent_state_dim]
                "pointcloud": point_cloud,              # [batch, N, 3]
                "static_affordance": affordance,        # [batch, 3]
                "action_horizon": action_horizon,       # [batch, horizon, action_dim]
            }
        else:
            # Multiple observation steps: model expects [B, T, D] (ndim=3).
            # In MPC we don't have real history, so repeat the current obs.
            # Use expand (no memory copy) instead of repeat.
            obs = {
                "agent_state": agent_state.unsqueeze(1).expand(-1, self.n_obs_steps, -1),
                "pointcloud": point_cloud.unsqueeze(1).expand(-1, self.n_obs_steps, -1, -1),
                "static_affordance": affordance.unsqueeze(1).expand(-1, self.n_obs_steps, -1),
                "action_horizon": action_horizon,
            }

        return obs

    def predict_single_step(
        self,
        agent_state: torch.Tensor,
        affordance: torch.Tensor,
        point_cloud: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict single-step dynamics (convenience wrapper).

        Args:
            agent_state: Current state [batch, agent_state_dim]
            affordance: Current affordance [batch, 3]
            point_cloud: Point cloud [batch, N, 3]
            action: Single action [batch, action_dim]

        Returns:
            Dict with next_state and next_affordance
        """
        action_horizon = action.unsqueeze(1)  # [batch, 1, action_dim]
        result = self.predict_trajectory(
            agent_state, affordance, point_cloud, action_horizon
        )

        out = {
            "next_state": result["predicted_states"][:, 0, :],
            "next_affordance": result["predicted_affordances"][:, 0, :],
            "predicted_point_cloud": result["predicted_point_cloud"],  # [batch, N, 3]
        }
        if "contact_logits" in result:
            out["contact_logit"] = result["contact_logits"][:, 0, :]  # [batch, 1]
        return out
